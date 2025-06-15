import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Replace these with your actual true labels and predicted labels
true_labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])  # True labels
predicted_labels = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])  # Predicted labels

# Generate the classification report
report = classification_report(true_labels, predicted_labels, target_names=['Negative', 'Positive'], output_dict=True)

# Convert report to DataFrame for better visualization
report_df = pd.DataFrame(report).transpose()
print("Classification Report:")
print(report_df)

# Create a confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# Performance metrics summary
performance_metrics = {
    "Metric": ["Accuracy", "Sensitivity", "Specificity", "F1 Score"],
    "Value": [report['accuracy'], report['Positive']['recall'], report['Negative']['recall'], report['Positive']['f1-score']]
}

performance_df = pd.DataFrame(performance_metrics)
print("Performance Metrics:")
print(performance_df)

# If you have results from standard machine learning techniques, you can create a comparison table like this:
comparison_data = {
    "Model": ["Proposed Model", "Model A", "Model B", "Model C"],
    "Accuracy": [report['accuracy'], 0.85, 0.82, 0.78],  # Replace with actual accuracies
    "F1 Score": [report['Positive']['f1-score'], 0.80, 0.75, 0.70]  # Replace with actual F1 scores
}

comparison_df = pd.DataFrame(comparison_data)
print("Standard Machine Learning Techniques Comparison:")
print(comparison_df)

# You can also visualize train vs val loss and accuracy if you have that data
# For example:
epochs = np.arange(1, 11)  # Replace with actual epoch numbers
train_loss = np.random.rand(10)  # Replace with actual training loss values
val_loss = np.random.rand(10)  # Replace with actual validation loss values
train_acc = np.random.rand(10)  # Replace with actual training accuracy values
val_acc = np.random.rand(10)  # Replace with actual validation accuracy values

# Plotting loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('Train vs Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Val Accuracy')
plt.title('Train vs Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
