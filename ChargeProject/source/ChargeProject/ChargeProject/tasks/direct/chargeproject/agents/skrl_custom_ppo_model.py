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
            nn.Linear(base_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Height_data encoder CNN (16x16 to 256)
        self.height_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),  # 32 * 2 * 2 = 128
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # For fusion of both encoders
        self.fusion = nn.Sequential(
            nn.Linear(128+128, 128), # 256 -> 128
            nn.ReLU(),
        )
        
        # Leg encoder (shared across legs)
        self.leg_encoder = nn.Sequential(
            nn.Linear(leg_dims[1], 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Leg gru (weights shared, memory per leg)
        self.sequence_length = 32
        self.leg_num_layers = 2
        self.leg_hidden_size = 128
        #self.leg_gru = nn.GRU(
        #    input_size=128+32,
        #    num_layers=self.leg_num_layers,
        #    hidden_size=self.leg_hidden_size,
        #    batch_first=True,
        #)

        # leg decoder
        self.leg_decoder = nn.Sequential(
            nn.Linear(128+32, 128),
            nn.ReLU(),
            nn.Linear(self.leg_hidden_size, 64), # 128 -> 64
            nn.ReLU(),
            nn.Linear(64, 32), # 64 -> 32
            nn.ReLU(),
        )
        
        # Policy Head
        self.leg_policy_head = nn.Linear(32, self.num_leg_joints) # 32 -> action_dim (4)
        # Value Head
        self.value_num_layers = 1
        self.value_hidden_size = 128
        self.value_gru = nn.GRU(input_size=128 + 32*2, # fused + legs_mean + legs_std 
                        num_layers=self.value_num_layers,
                        hidden_size=self.value_hidden_size, # 192 -> 128
                        batch_first=True)
        self.value_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_leg_joints,), fill_value=init_log_std), requires_grad=True)

        self._policy_out = None
        self._value_out = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0)

        def _make_grad_logger(name):
            def hook(grad):
                # grad is a tensor -> print its L2 norm
                print(f"[HOOK] grad on {name} norm: {grad.norm().item():.6f}")
            return hook

        # 1️⃣ Register hooks for all base_encoder parameters
        #for name, p in self.base_encoder.named_parameters():
        #    if not hasattr(p, "_grad_hook_registered"):
        #        p.register_hook(_make_grad_logger(f"base_encoder.{name}"))
        #        p._grad_hook_registered = True

        # 2️⃣ Hook on policy head (to see if actions get gradients)
        #self.leg_policy_head.weight.register_hook(_make_grad_logger("leg_policy_head.weight"))
        #self.leg_policy_head.bias.register_hook(_make_grad_logger("leg_policy_head.bias"))

        # 3️⃣ Hook on log_std parameter
        #self.log_std_parameter.register_hook(_make_grad_logger("log_std_parameter"))


    
    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    #(self.leg_num_layers, self.num_envs, self.leg_hidden_size) for _ in range(self.num_legs)
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
            """
            # Optional one-time hook on the output activation itself
            if torch.is_grad_enabled() and base_encoded.requires_grad and not hasattr(self, "_base_encoded_hook_registered"):
                
                # Base encoder hooks
                for name, p in self.base_encoder.named_parameters():
                    p.register_hook(lambda g, n=name: print(f"[HOOK] base_encoder.{n} grad norm: {g.norm().item()}"))

                # Height encoder hooks
                for name, p in self.height_encoder.named_parameters():
                    p.register_hook(lambda g, n=name: print(f"[HOOK] height_encoder.{n} grad norm: {g.norm().item()}"))

                # Leg encoder hooks
                for name, p in self.leg_encoder.named_parameters():
                    p.register_hook(lambda g, n=name: print(f"[HOOK] leg_encoder.{n} grad norm: {g.norm().item()}"))
                
            log_std_vals = self.log_std_parameter.data.tolist()
            std_vals = self.log_std_parameter.data.exp().tolist()
            print("[DBG] log_std:", log_std_vals)
            print("[DBG] std:", std_vals)
            """
            height_encoded = self.height_encoder(height_data.unsqueeze(1))  # add channel dim
            fused_global = self.fusion(torch.cat([base_encoded, height_encoded], dim=-1))  # [B, 128]

            # --- Per-leg encoding and GRU ---
            leg_obs_flat = leg_obs_all.reshape(-1, leg_obs_all.shape[-1])  # [B * num_legs, leg_dim]
            leg_encoded = self.leg_encoder(leg_obs_flat)
            fused_global_exp = fused_global.unsqueeze(1).expand(-1, self.num_legs, -1).contiguous()
            fused_global_flat = fused_global_exp.reshape(-1, fused_global_exp.shape[-1])  # [B * num_legs, 128]
            leg_gru_input = torch.cat([fused_global_flat, leg_encoded], dim=-1)
            #if terminated is not None:
            #    terminated_flat = terminated.expand(-1, self.num_legs).reshape(-1)
            #else:
            #    terminated_flat = None

            # rnn_data[0]-rnn_data[5]
            #leg_data_flat = torch.cat(rnn_data[:self.num_legs], dim=1)
            #leg_rnn_output, leg_rnn_hidden = self.gru_rollout(
            #    self.leg_gru,
            #    self.leg_num_layers,
            #    leg_gru_input,
            #    terminated_flat,
            #    leg_data_flat,
            #)
            #for i in range(self.num_legs):
                #rnn_dict["rnn"].append(leg_rnn_hidden[:, i::self.num_legs, :])  # split hidden states per leg
            leg_decoded_flat = self.leg_decoder(leg_gru_input)  # [B * num_legs, 32]
            leg_decoded_all = leg_decoded_flat.view(-1, self.num_legs, leg_decoded_flat.shape[-1])  # [B, num_legs, 32]

            # --- Aggregate leg features for value ---
            leg_mean = leg_decoded_all.mean(dim=1)
            leg_std = leg_decoded_all.std(dim=1)
            leg_agg = torch.cat([leg_mean, leg_std], dim=-1)  # [B, 64]

            # --- Policy Head ---
            leg_action_mean = self.leg_policy_head(leg_decoded_flat)  # [B * num_legs, action_dim_per_leg]
            mean_flat = leg_action_mean.view(-1, self.num_legs * self.num_leg_joints)  # [B, num_legs, action_dim]


            # Calc value output
            # Value GRU (global)
            value_input = torch.cat([fused_global, leg_agg], dim=-1)  # [B, 192]
            value_rnn_output, value_rnn_hidden = self.gru_rollout(
                self.value_gru,
                self.value_num_layers,
                value_input,
                terminated,
                rnn_data[-1],  # last slot for value GRU
            )
            rnn_dict["rnn"].append(value_rnn_hidden)

            # Final value prediction
            value = self.value_layer(value_rnn_output)  # [B, 1]

            log_flat = self.log_std_parameter.repeat(self.num_legs)
            self._policy_out = mean_flat, log_flat, rnn_dict
            self._value_out = value, rnn_dict

            """
            # Only if you can access advantages/returns here — otherwise skip
            with torch.no_grad():
                base_zero = torch.zeros_like(base_obs)
                base_zero_encoded = self.base_encoder(base_zero)
                height_encoded_ = self.height_encoder(height_data.unsqueeze(1))
                fused_zero = self.fusion(torch.cat([base_zero_encoded, height_encoded_], dim=-1))
                fused_zero_exp = fused_zero.unsqueeze(1).expand(-1, self.num_legs, -1).contiguous()
                fused_zero_flat = fused_zero_exp.reshape(-1, fused_zero_exp.shape[-1])
                leg_gru_input_zero = torch.cat([fused_zero_flat, leg_encoded], dim=-1)

                # Feed through leg GRU & decoder (single step)
                leg_rnn_output_z, _ = self.leg_gru(leg_gru_input_zero.unsqueeze(1))
                leg_decoded_z_flat = self.leg_decoder(leg_rnn_output_z.squeeze(1))

                # Policy head output
                mean_flat_z = self.leg_policy_head(leg_decoded_z_flat).view(-1, self.num_legs * self.num_leg_joints)

                # Compare to original mean
                diff = (mean_flat - mean_flat_z).abs().mean().item()
                print("[ABLATION] mean difference when base=0 (avg abs):", diff)

            with torch.no_grad():
                height_zero = torch.zeros_like(height_data)
                height_zero_encoded = self.height_encoder(height_zero.unsqueeze(1))  # keep channel dim
                fused_height_zero = self.fusion(torch.cat([base_encoded, height_zero_encoded], dim=-1))
                fused_height_zero_exp = fused_height_zero.unsqueeze(1).expand(-1, self.num_legs, -1).contiguous()
                fused_height_zero_flat = fused_height_zero_exp.reshape(-1, fused_height_zero_exp.shape[-1])
                leg_gru_input_height_zero = torch.cat([fused_height_zero_flat, leg_encoded], dim=-1)

                # Feed through leg GRU & decoder
                leg_rnn_output_hz, _ = self.leg_gru(leg_gru_input_height_zero.unsqueeze(1))
                leg_decoded_hz_flat = self.leg_decoder(leg_rnn_output_hz.squeeze(1))

                # Policy head output
                mean_flat_hz = self.leg_policy_head(leg_decoded_hz_flat).view(-1, self.num_legs * self.num_leg_joints)

                # Compare to original mean
                diff_height = (mean_flat - mean_flat_hz).abs().mean().item()
                print("[ABLATION] mean difference when height_data=0 (avg abs):", diff_height)
            
            with torch.no_grad():
                leg_rnn_output_zero = torch.zeros_like(leg_rnn_output)  # zero leg RNN features
                leg_decoded_zero = self.leg_decoder(leg_rnn_output_zero)
                mean_flat_legs_zero = self.leg_policy_head(leg_decoded_zero).view(-1, self.num_legs * self.num_leg_joints)
                diff_legs = (mean_flat - mean_flat_legs_zero).abs().mean().item()
                print("[ABLATION] mean difference when legs=0 (avg abs):", diff_legs)"""

        if role == "policy":
            out = self._policy_out
            self._policy_out = None
            return out
        elif role == "value":
            out = self._value_out
            self._value_out = None
            return out

    def gru_rollout(self, model, num_layers, states, terminated, hidden_state):
        #print(f"states shape: {states.shape}, hidden_states shapes: {[h.shape for h in hidden_states]}")
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