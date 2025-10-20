import torch
import torch.nn as nn

# Assuming these are imported from your RL library (e.g., skrl)
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space


class SharedRecurrentModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs, init_log_std=0.0):
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

        obs_dim = self.observation_space["observations"].shape[0]
        height_dim = self.observation_space["height_data"].shape
        assert height_dim == (16, 16), "Expected height_data to be of shape (16, 16)"
        act_dim = self.num_actions
        self.num_envs = num_envs


        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 192),
            nn.ReLU(),
        )

        # Height_data encoder CNN (16x16 to 256)
        self.height_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 4x4 -> 2x2
            nn.ReLU(),
            nn.Flatten(),  # 32 * 2 * 2 = 128
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # For fusion of both encoders
        self.fusion = nn.Sequential(
            nn.Linear(320, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
        )


        self.num_layers = 1
        self.input_size = 256
        self.hidden_size = 128
        self.sequence_length = 32
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,    # Input/output tensors are (batch, seq, feature)
        )

        self.net = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
        )

        self.policy_layer = nn.Linear(128, act_dim)
        self.value_layer = nn.Linear(128, 1)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=init_log_std), requires_grad=True)

        self._shared_output = None
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.6)
                nn.init.constant_(m.bias, 0)

    
    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),  # gru memory
                ]
            }
        }
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),  # hx
                    (self.num_layers, self.num_envs, self.hidden_size),  # cx
                ]
            }
        }
    
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        if self._shared_output is None:
            states = unflatten_tensorized_space(self.observation_space, inputs["states"])
            observations = states["observations"]
            height_data = states["height_data"]
            
            #terminated = inputs.get("terminated", None)
            rnn_dict = {}
            
            # Encode observations
            obs_encoded = self.obs_encoder(observations)
            height_encoded = self.height_encoder(height_data.unsqueeze(1))  # Add channel dimension
            encoded = torch.cat([obs_encoded, height_encoded], dim=-1)

            # Fusion
            fused = self.fusion(encoded)

            # LSTM
            #rnn_output, rnn_dict = self.lstm_rollout(self.lstm, fused, terminated, inputs["rnn"])
            # GRU
            rnn_output, rnn_dict = self.gru_rollout(self.gru, fused, None, inputs["rnn"])

            # Final layers
            net = self.net(rnn_output)

            self._shared_output = net, rnn_dict

        if role == "policy":
            mean = self.policy_layer(net)
            return mean, self.log_std_parameter, rnn_dict

        elif role == "value":
            net, rnn_dict = self._shared_output
            self._shared_output = None
            output = self.value_layer(net)
            return output, rnn_dict

    # === LSTM rollout logic ===
    def lstm_rollout(self, model, states, terminated, hidden_states):
        #print(f"states shape: {states.shape}, hidden_states shapes: {[h.shape for h in hidden_states]}")
        if self.training:
            # reshape to (batch, seq, features)
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])
            
            h, c = hidden_states
            h = h.view(
                self.num_layers, -1, self.sequence_length, h.shape[-1]
            )[:, :, 0, :].contiguous()
            c = c.view(
                self.num_layers, -1, self.sequence_length, c.shape[-1]
            )[:, :, 0, :].contiguous()
            hidden_states = (h.contiguous(), c.contiguous())

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
                    rnn_output, (h, c) = model(
                        rnn_input[:, i0:i1, :], hidden_states
                    )
                    h[:, (terminated[:, i1 - 1]), :] = 0
                    c[:, (terminated[:, i1 - 1]), :] = 0
                    hidden_states = (h, c)
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states = model(rnn_input, hidden_states)
        else:
            # evaluation mode: one step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            # Make h, c contiguous
            h, c = hidden_states
            h = h.contiguous()
            c = c.contiguous()
            hidden_states = (h, c)
            rnn_output, hidden_states = model(rnn_input, hidden_states)

        # flatten batch + sequence
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, {"rnn": hidden_states}

    def gru_rollout(self, model, states, terminated, hidden_states):
        #print(f"states shape: {states.shape}, hidden_states shapes: {[h.shape for h in hidden_states]}")
        if self.training:
            # reshape to (batch, seq, features)
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])
            

            hidden_states[0] = hidden_states[0].view(
                self.num_layers, -1, self.sequence_length, hidden_states[0].shape[-1]
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
                    rnn_output, hidden_states[0] = model(
                        rnn_input[:, i0:i1, :], hidden_states[0]
                    )
                    hidden_states[0][:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states[0] = model(rnn_input, hidden_states[0])
        else:
            # evaluation mode: one step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            # Make h contiguous
            hidden_states[0] = hidden_states[0].contiguous()
            rnn_output, hidden_states[0] = model(rnn_input, hidden_states[0])
        # flatten batch + sequence
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, {"rnn": hidden_states}