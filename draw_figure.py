import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('./loss/loss_step5000_lambda0.1.csv')  # Replace with your actual file path

# Plot the losses
plt.figure(figsize=(10, 6))
# plt.plot(data['step'], data['initial_loss'], label='Initial Loss')
# plt.plot(data['step'], data['aesthetic_loss'], label='Aesthetic Loss')
# plt.plot(data['step'], data['total_loss'], label='Total Loss')
plt.plot(data['initial_loss'], label='Initial Loss')
plt.plot(data['aesthetic_loss'], label='Aesthetic Loss')
plt.plot(data['total_loss'], label='Total Loss')
plt.xlabel('Step')
plt.ylabel('Loss Value')
plt.title('Loss Trends over Training Steps')
plt.legend()

# Save the figure
current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
plt.savefig(f'./data_analysis/loss_trends_{current_time}.png')  # Replace with your desired save location and filename
plt.show()