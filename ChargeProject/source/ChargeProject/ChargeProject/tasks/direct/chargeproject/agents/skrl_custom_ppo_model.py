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
            nn.Linear(base_dim, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
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
            nn.AdaptiveAvgPool2d((2, 2)),  # <- new: compress to (B, 32, 2, 2)
            nn.Flatten(),
        )

        # For fusion of both encoders
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 192), # 256 -> 192
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
        )
        
        # Leg encoder (shared across legs)
        self.leg_encoder = nn.Sequential(
            nn.Linear(leg_dims[1], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.leg_net = nn.Sequential(
            nn.Linear(128 + 64, 128), # fused + leg_encoded
            nn.ReLU(),
        )

        # Leg gru (weights shared, memory per leg)
        self.sequence_length = 32
        self.leg_num_layers = 1
        self.leg_hidden_size = 128
        self.leg_gru = nn.GRU(
            input_size=128,
            num_layers=self.leg_num_layers,
            hidden_size=self.leg_hidden_size,
            batch_first=True,
        )

        # leg decoder
        self.leg_decoder = nn.Sequential(
            nn.Linear(self.leg_hidden_size, 64), # 128 -> 64
            nn.ReLU(),
        )
        
        # Policy Head
        self.leg_policy_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_leg_joints),
        )

        # Value Head
        self.value_num_layers = 1
        self.value_hidden_size = 128
        self.value_gru = nn.GRU(input_size=64*6, # fused + legs_mean + legs_std 
                        num_layers=self.value_num_layers,
                        hidden_size=self.value_hidden_size, # 192 -> 128
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
            fused_global = self.fusion(torch.cat([base_encoded, height_encoded], dim=-1))  # [B, 128]

            # --- Per-leg encoding and GRU ---
            # Flatten so each leg acts like different env
            leg_obs_flat = leg_obs_all.reshape(-1, leg_obs_all.shape[-1])  # [B * num_legs, leg_dim]
            # Expand fused to match leg count
            fused_global_exp = fused_global.unsqueeze(1).expand(-1, self.num_legs, -1).contiguous()
            fused_global_flat = fused_global_exp.reshape(-1, fused_global_exp.shape[-1])  # [B * num_legs, 128]
            # expand terminated to match leg count
            terminated_flat = terminated.expand(-1, self.num_legs).reshape(-1) if terminated is not None else None
            # Flatten rnn_data for legs
            leg_data_flat = torch.cat(rnn_data[:self.num_legs], dim=1)

            # --- Running leg models ---
            leg_encoded = self.leg_encoder(leg_obs_flat)
            
            leg_net_input = torch.cat([fused_global_flat, leg_encoded], dim=-1)

            leg_gru_input = self.leg_net(leg_net_input)  # [B * num_legs, 128]

            # rnn_data[0]-rnn_data[5]
            leg_rnn_output, leg_rnn_hidden = self.gru_rollout(
                self.leg_gru,
                self.leg_num_layers,
                leg_gru_input,
                terminated_flat,
                leg_data_flat,
            )
            
            leg_decoded_flat = self.leg_decoder(leg_rnn_output)  # [B * num_legs, 32]

            # Reshape leg hidden states back to per-leg
            for i in range(self.num_legs):
                rnn_dict["rnn"].append(leg_rnn_hidden[:, i::self.num_legs, :])
            # reshape leg decoded back to per-leg
            leg_decoded_all = leg_decoded_flat.view(-1, self.num_legs, leg_decoded_flat.shape[-1])  # [B, num_legs, 32]

            # --- Aggregate leg features for value ---
            leg_mean = leg_decoded_all.mean(dim=1)
            leg_std = leg_decoded_all.std(dim=1)
            leg_agg = torch.cat([leg_mean, leg_std], dim=-1)  # [B, 64]

            # --- Policy Head ---
            leg_action_mean = self.leg_policy_head(leg_decoded_flat)  # [B * num_legs, action_dim_per_leg]
            actions_flat = leg_action_mean.view(-1, self.num_legs * self.num_leg_joints)  # [B, num_legs, action_dim]


            # Calc value output
            # Value GRU (global)
            # Now cat all leg_decoded_all instead of leg_agg and dont include fused_global
            value_input = leg_decoded_all.view(-1, self.num_legs * leg_decoded_all.shape[-1])
            #torch.cat([fused_global, leg_agg], dim=-1)  # [B, 192]
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
            self._policy_out = actions_flat, log_flat, rnn_dict
            self._value_out = value, rnn_dict

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