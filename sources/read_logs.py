from tensorboard.backend.event_processing import event_accumulator
import os

directory_path = "../outputs/logs/lightning_logs/version_1"
try:
    directory_contents = os.listdir(directory_path)
    print(directory_contents)
except Exception as e:
    print("Exception", str(e))

event_file = "../outputs/logs/lightning_logs/version_4/events.out.tfevents.1736261988.dgxa100-ncit-wn02.grid.pub.ro.1635384.0"

# Load the event file
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()  # Load the data

print(ea.Tags()['scalars'])

# Extract scalar data (e.g., training loss)

perplexities = ea.Scalars('val_perplexity')
for step in perplexities:
    print(f"Step: {step.step}, Value: {step.value}")

# losses = ea.Scalars('train_loss')
# for step in losses:
#     print(f"Step: {step.step}, Value: {step.value}")
