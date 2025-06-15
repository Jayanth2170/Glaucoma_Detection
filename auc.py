import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch

# Load the training losses from the Excel file
losses_df = pd.read_excel('training_losses.xlsx')

# Check available columns
print("Available columns in training_losses.xlsx:", losses_df.columns)

# Use 'Loss' for the loss values, and 'Epoch' or 'Batch' for plotting
train_losses = losses_df['Loss']

# If you want to plot against epochs or batches, choose one:
epochs = losses_df['Epoch']
batches = losses_df['Batch']

# Plot Loss vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# If you have validation losses as well, you might need to adjust the code accordingly
# Replace with the correct data if you have them
# val_losses = losses_df['Validation Loss'] # if available in your data
