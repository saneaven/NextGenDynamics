import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


# Will auto select the latest run if None
log_dir = None
#log_dir = "logs\\skrl\\quadre\\2025-09-23_10-04-10_ppo_torch_Load both next dist"



# Parses tfevents into a DataFrame
def parse_tfevents_to_dataframe(tfevents_file):
    print(f"Reading data from: {tfevents_file}")
    
    # Initialize the accumulator
    acc = event_accumulator.EventAccumulator(tfevents_file)
    acc.Reload()  # Load all data from the file
    
    # Get all scalar tags
    tags = acc.Tags()['scalars']
    
    all_data = []
    for tag in tags:
        events = acc.Scalars(tag)
        for event in events:
            all_data.append({
                'tag': tag,
                'step': event.step,
                'value': event.value
            })
            
    print(f"Found {len(all_data)} data points across {len(tags)} tags.")
    return pd.DataFrame(all_data)

# If log_dir is not set get the latest folder in logs/skrl/quadre
if log_dir is None:
    base_log_dir = "logs/skrl/quadre"
    all_runs = glob.glob(os.path.join(base_log_dir, "*"))
    if not all_runs:
        print(f"No runs found in '{base_log_dir}'")
        exit()
    log_dir = max(all_runs, key=os.path.getmtime)
    print(f"No log_dir specified. Using the latest run: {log_dir}")

# Find the .tfevents file automatically
try:
    event_file_path = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))[0]
except IndexError:
    print(f"Error: Could not find a .tfevents file in '{log_dir}'")
    exit()


# Read all scalar data from the log file
full_df = parse_tfevents_to_dataframe(event_file_path)

# Filter for only the reward-related tags
# This captures the total reward and all individual components
reward_tags_df = full_df[
    full_df['tag'].str.startswith('Episode_Reward/')
].copy() # Use .copy() to avoid SettingWithCopyWarning


# Plotting
fig, ax = plt.subplots(figsize=(16, 9))

# Plot each reward component
for tag_name in reward_tags_df['tag'].unique():
    # Get the data for the current reward component
    subset_df = reward_tags_df[reward_tags_df['tag'] == tag_name]
    
    # Clean up the label name for the legend
    clean_label = tag_name.replace('Episode_Reward/', '')
    
    ax.plot(subset_df['step'], subset_df['value'], label=clean_label, alpha=0.8)

# Customize the plot
ax.set_title("Reward Components Over Training Steps", fontsize=18)
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Reward Value", fontsize=12)
ax.legend(title="Reward Components", bbox_to_anchor=(1.02, 1), loc='upper left')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
fig.tight_layout()

# if looking at small rewards

#ax.set_ylim(-0.01, 0.01)

# Save the figure to a file
#plt.savefig("reward_overlap_plot.png", dpi=300)
#print("\nPlot saved as 'reward_overlap_plot.png'")

# Display the plot
plt.show()