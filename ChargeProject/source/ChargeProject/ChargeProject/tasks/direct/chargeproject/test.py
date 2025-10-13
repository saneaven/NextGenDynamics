import torch

# -----------------------------
# CONFIG
# -----------------------------
checkpoint_path = r"G:\Code\NextGenDynamics\ChargeProject\logs\skrl\spiderbot\2025-10-12_18-08-46_ppo_torch\checkpoints\agent_1000.pt"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# -----------------------------
# LOAD CHECKPOINT
# -----------------------------
checkpoint = torch.load(checkpoint_path, map_location=device)
print("Checkpoint keys:", checkpoint.keys())

# -----------------------------
# FUNCTION TO INSPECT A STATE DICT
# -----------------------------
def inspect_state_dict(state_dict, name):
    print(f"\n{name} state dict:")
    lstm_detected = False
    for param_name, tensor in state_dict.items():
        print(f"{param_name}: {tensor.shape}, device={tensor.device}")
        # Detect if this is an LSTM layer
        if "weight_ih" in param_name or "weight_hh" in param_name:
            lstm_detected = True
    if lstm_detected:
        print(f"✅ LSTM detected in {name}")
    else:
        print(f"⚠️ No LSTM detected in {name}")

# -----------------------------
# INSPECT POLICY AND VALUE
# -----------------------------
policy_state = checkpoint['policy']
value_state = checkpoint['value']

inspect_state_dict(policy_state, "Policy model")
inspect_state_dict(value_state, "Value model")
