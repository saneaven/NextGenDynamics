import torch
import torch.nn as nn

# Assuming these are imported from your RL library (e.g., skrl)
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


class SharedRecurrentModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs):
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

        self.hidden_size = 1024
        self.sequence_length = 128
        self.num_layers = 2

        self.num_envs = num_envs

        self.lstm = nn.LSTM(
            input_size=self.observation_space.shape[0],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,  # Using a single LSTM layer
            batch_first=True,    # Input/output tensors are (batch, seq, feature)
        )

        self.net = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=512),
            nn.ELU(),
            nn.LazyLinear(out_features=256),
            nn.ELU(),
            nn.LazyLinear(out_features=128),
            nn.ELU(),
        )

        self.policy_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        self.value_layer = nn.LazyLinear(out_features=1)

    
    def get_specification(self):
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
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"]

        # hidden_states = [hx, cx]
        rnn_output, rnn_dict = self.rnn_rollout(states, terminated, hidden_states)

        net = self.net(rnn_output)
        self._shared_output = net

        if role == "policy":
            mean = self.policy_layer(net)
            return mean, self.log_std_parameter, rnn_dict

        elif role == "value":
            output = self.value_layer(net)
            return output, rnn_dict

    # === RNN rollout logic ===
    def rnn_rollout(self, states, terminated, hidden_states):
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
                    + (
                        terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1
                    ).tolist()
                    + [self.sequence_length]
                )
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (h, c) = self.lstm(
                        rnn_input[:, i0:i1, :], hidden_states
                    )
                    h[:, (terminated[:, i1 - 1]), :] = 0
                    c[:, (terminated[:, i1 - 1]), :] = 0
                    hidden_states = (h, c)
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states = self.lstm(rnn_input, hidden_states)
        else:
            # evaluation mode: one step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            rnn_output, hidden_states = self.lstm(rnn_input, hidden_states)

        # flatten batch + sequence
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, {"rnn": hidden_states}
