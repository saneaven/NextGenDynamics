import torch
import torch.nn as nn

# Assuming these are imported from your RL library (e.g., skrl)
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space


class SharedRecurrentModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs, init_log_std=0.0, gain=1.0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
            role="policy",
        )
        DeterministicMixin.__init__(self, 
            clip_actions=False, 
            role="value"
        )

        base_dim = self.observation_space["base_obs"].shape[0]
        leg_dims = self.observation_space["leg_obs"].shape
        self.num_legs = leg_dims[0]
        self.num_leg_joints = action_space.shape[1]
        height_dim = self.observation_space["height_data"].shape
        assert height_dim == (16, 16), "Expected height_data to be of shape (16, 16)"
        self.num_envs = num_envs


        # Observation encoder
        self.base_encoder = nn.Sequential(
            nn.Linear(base_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )

        # Height_data encoder CNN (16x16 to 128)
        self.height_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0),   # 16 → 14
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),  # 14 → 12
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0), # 12 → 5
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0), # 5 → 3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((2, 2)),  # <- new: compress to (B, 32, 2, 2)
            nn.Flatten(), # flatten to (B, 128)
        )
        
        # Leg encoder (shared across legs)
        self.leg_encoder = nn.Sequential(
            nn.Linear(leg_dims[1], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
        )


        # ----- Policy Network -----

        leg_input_size = 64 + 128 + 32 # 224 (fused_base + leg_encoded)
        
        self.leg_pre_net = nn.Sequential(
            nn.Linear(leg_input_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128), # Compress to 128
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # Leg gru (weights shared, memory per leg)
        self.sequence_length = 64
        self.leg_num_layers = 2
        self.leg_hidden_size = 128 # fused base+height + leg encoding
        self.leg_gru = nn.GRU(
            input_size=self.leg_hidden_size,
            num_layers=self.leg_num_layers,
            hidden_size=self.leg_hidden_size,
            batch_first=True,
        )

        # Policy Head
        self.leg_policy_head = nn.Sequential(
            nn.Linear(self.leg_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_leg_joints),
        )


        # ----- Value Network -----
        value_input_size = 192 + (self.num_legs * 32) # 192 + 192 = 384

        self.value_pre_net = nn.Sequential(
            nn.Linear(value_input_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128), # 256 -> 128
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        self.value_num_layers = 1
        self.value_hidden_size = 128
        self.value_gru = nn.GRU(input_size=128,
                        num_layers=self.value_num_layers,
                        hidden_size=self.value_hidden_size, 
                        batch_first=True)
        
        self.value_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_leg_joints,), fill_value=init_log_std), requires_grad=True)

        self._policy_out = None
        self._value_out = None

        self.debug_i = 0 

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0)


    
    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.leg_num_layers, self.num_envs, self.leg_hidden_size) for _ in range(self.num_legs)
                ] +
                [
                    (self.value_num_layers, self.num_envs, self.value_hidden_size)
                ]
            }
        }
    
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        rnn_data = inputs["rnn"]

        if self._value_out is None:
            states = unflatten_tensorized_space(self.observation_space, inputs["states"])
            base_obs = states["base_obs"]
            leg_obs_all = states["leg_obs"]  # [B, num_legs, leg_dim]
            height_data = states["height_data"]

            terminated = inputs.get("terminated", None)
            rnn_dict = {"rnn": []}

            # --- Base and height encoders ---
            base_encoded = self.base_encoder(base_obs)
            height_encoded = self.height_encoder(height_data.unsqueeze(1))  # add channel dim

            # --- Per-leg encoding and GRU ---
            # Flatten so each leg acts like different env
            leg_obs_flat = leg_obs_all.reshape(-1, leg_obs_all.shape[-1])  # [B * num_legs, leg_dim]
            # Fuse and expand base to match leg count
            fused_base = torch.cat([base_encoded, height_encoded], dim=-1)
            fused_base_exp = fused_base.unsqueeze(1).expand(-1, self.num_legs, -1).contiguous()
            fused_base_flat = fused_base_exp.reshape(-1, fused_base_exp.shape[-1])  # [B * num_legs, 128]
            # expand terminated to match leg count
            terminated_flat = terminated.expand(-1, self.num_legs).reshape(-1) if terminated is not None else None
            # Flatten rnn_data for legs
            leg_data_flat = torch.cat(rnn_data[:self.num_legs], dim=1)

            leg_encoded = self.leg_encoder(leg_obs_flat)
            
            # --- Policy SPECIFIC ---
            leg_input = torch.cat([fused_base_flat, leg_encoded], dim=-1)
            
            leg_gru_input = self.leg_pre_net(leg_input)
            # rnn_data[0]-rnn_data[5]
            leg_rnn_output, leg_rnn_hidden = self.gru_rollout(
                self.leg_gru,
                self.leg_num_layers,
                leg_gru_input,
                terminated_flat,
                leg_data_flat,
            )

            leg_action_mean = self.leg_policy_head(leg_rnn_output)  # [B * num_legs, action_dim_per_leg]
            actions_flat = leg_action_mean.view(-1, self.num_legs * self.num_leg_joints)  # [B, num_legs, action_dim]

            # Reshape leg hidden states back to per-leg
            for i in range(self.num_legs):
                rnn_dict["rnn"].append(leg_rnn_hidden[:, i::self.num_legs, :])


            # ----- Value Specific -----
            value_input = torch.cat(
                [fused_base, leg_encoded.view(-1, self.num_legs * 32)],
                dim=-1,
            )  # base + height + leg_encoded
            value_gru_input = self.value_pre_net(value_input)

            value_rnn_output, value_rnn_hidden = self.gru_rollout(
                self.value_gru,
                self.value_num_layers,
                value_gru_input,
                terminated,
                rnn_data[-1],  # last slot for value GRU
            )
            rnn_dict["rnn"].append(value_rnn_hidden)

            # Final value prediction
            value = self.value_layer(value_rnn_output)  # [B, 1]

            log_flat = self.log_std_parameter.repeat(self.num_legs)
            self._policy_out = actions_flat, log_flat, rnn_dict
            self._value_out = value, rnn_dict
            
            # ----- UPDATED DEBUG BLOCK -----
            self.debug_i += 1
            if self.debug_i % 128 == 1:
                with torch.no_grad():
                    # Prepare copies for ablation
                    base_obs_zero = base_obs.clone()
                    height_data_zero = height_data.clone()
                    leg_obs_zero = leg_obs_all.clone()

                    # Zero out specific parts (adjust indices if needed)
                    base_obs_zero[..., -6:] = 0       # zero target info
                    height_data_zero[:] = 0           # zero height map
                    leg_obs_zero[..., :6] = 0         # zero leg velocities

                    ablation_dict = {
                        "zero_target": base_obs_zero,
                        "zero_height": height_data_zero,
                        "zero_leg_vel": leg_obs_zero
                    }
                    
                    # Get the current hidden state for the per-leg GRU
                    # This tensor was prepared for gru_rollout
                    current_leg_hidden_state = leg_data_flat.contiguous()

                    for name, obs_mod in ablation_dict.items():
                        # Encode base
                        base_enc = self.base_encoder(obs_mod if "target" in name else base_obs)
                        # Encode height
                        height_enc = self.height_encoder(obs_mod.unsqueeze(1) if "height" in name else height_data.unsqueeze(1))
                        # Fuse base + height
                        fused = torch.cat([base_enc, height_enc], dim=-1)
                        fused_exp = fused.unsqueeze(1).expand(-1, self.num_legs, -1).reshape(-1, fused.shape[-1])
                        
                        # Encode legs
                        leg_input_obs = obs_mod.reshape(-1, obs_mod.shape[-1]) if "leg" in name else leg_obs_flat
                        leg_enc = self.leg_encoder(leg_input_obs)
                        
                        # --- FIXED ABLATION PATH ---
                        # 1. Replicate the *full* policy input path, including the pre-net
                        leg_input_raw_mod = torch.cat([fused_exp, leg_enc], dim=-1)
                        leg_gru_input_mod = self.leg_pre_net(leg_input_raw_mod)

                        # 2. Call the GRU with a single time-step (unsqueeze(1))
                        #    AND pass the *correct* hidden state.
                        leg_rnn_output_mod, _ = self.leg_gru(
                            leg_gru_input_mod.unsqueeze(1),  # [B*num_legs, 1, H_in]
                            current_leg_hidden_state
                        )
                        
                        # 3. Squeeze the time-step dimension
                        leg_rnn_output_mod = leg_rnn_output_mod.squeeze(1) # [B*num_legs, H_out]
                        
                        # Policy head
                        leg_actions_mod = self.leg_policy_head(leg_rnn_output_mod).view(-1, self.num_legs * self.num_leg_joints)
                        
                        diff = (leg_actions_mod - actions_flat).abs().mean().item()
                        print(f"[Ablation] {name} change in mean action magnitude: {diff:.4f}")
            # ----- END OF DEBUG BLOCK -----

        if role == "policy":
            out = self._policy_out
            self._policy_out = None
            return out
        elif role == "value":
            out = self._value_out
            self._value_out = None
            return out

    def gru_rollout(self, model, num_layers, states, terminated, hidden_state):
        if self.training:
            # reshape to (batch, seq, features)
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])
            

            hidden_state = hidden_state.view(
                num_layers, -1, self.sequence_length, hidden_state.shape[-1]
            )[:, :, 0, :].contiguous()

            if terminated is not None and torch.any(terminated):
                # handle resets within the sequence
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_state = model(
                        rnn_input[:, i0:i1, :], hidden_state
                    )
                    hidden_state[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_state = model(rnn_input, hidden_state)
        else:
            # evaluation mode: one step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            # Make h contiguous
            hidden_state = hidden_state.contiguous()
            rnn_output, hidden_state = model(rnn_input, hidden_state)
        # flatten batch + sequence
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, hidden_state