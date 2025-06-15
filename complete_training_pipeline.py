from setup_paths import get_paths
from data_loader import load_data
from fpd_student_model import StudentModelFPD
from distillation_loss import distillation_loss
from fine_tuning_strategies import fine_tune_model

import os
import csv
import torch
import torch.optim as optim
import torch.nn as nn  # Import for neural network operations
from torchvision import models
from tqdm import tqdm  # Import tqdm for progress bar
import logging  # Import logging for better logging capabilities

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check paths before loading data
train_glaucoma_pos, train_glaucoma_neg, val_glaucoma_pos, val_glaucoma_neg = get_paths()

# Print paths and check if they exist
logging.info("Checking dataset paths...")
for path in [train_glaucoma_pos, train_glaucoma_neg, val_glaucoma_pos, val_glaucoma_neg]:
    if not os.path.exists(path):
        logging.warning(f"Path does not exist: {path}")
    else:
        logging.info(f"Path exists: {path}")
        logging.info("Contents: %s", os.listdir(path))  # List the contents of the directory

# Load data
logging.info("Loading data...")
train_loader, val_loader = load_data(train_glaucoma_pos, train_glaucoma_neg, val_glaucoma_pos, val_glaucoma_neg)

# Define the student model with FPD
logging.info("Defining the student model...")
student_model = StudentModelFPD().to(device)

# Load a pre-trained teacher model (ResNet50)
logging.info("Starting model loading...")
teacher_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)  # Load weights correctly
logging.info("Model loaded successfully, modifying output layer...")

# Modify the teacher model's final layer to match the number of classes (2 for binary classification)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 2).to(device)  # Change output features to 2
logging.info("Output layer modified.")

# Define optimizer
logging.info("Defining optimizer...")
optimizer = optim.AdamW(student_model.parameters(), lr=1e-4)

# Create checkpoints directory if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)

# Define the file path to save the trained model
model_save_path = 'checkpoints/glaucoma_student_model_trained.pth'

# Check if a trained model already exists
if os.path.exists(model_save_path):
    # Load the trained model if it already exists
    student_model.load_state_dict(torch.load(model_save_path))
    logging.info(f"Trained model loaded from {model_save_path}. No need to retrain.")
else:
    # If the trained model doesn't exist, you can continue training if needed
    logging.info("No pre-trained model found. Training from scratch.")

    # Train the model using the chosen fine-tuning strategy
    logging.info("Starting training process...")
    epochs = 10  # Define the number of epochs
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        logging.info(f"Starting Epoch {epoch + 1}/{epochs}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass through student model
            outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)  # Forward pass through teacher model

            # Calculate weights for the current batch
            weights = torch.tensor([0.7, 0.3], device=device)  # Adjust these weights as necessary
            batch_weights = weights[labels]  # Select weights based on the labels for the current batch
            batch_weights = batch_weights.view(-1)  # Ensure batch_weights matches the batch size

            # Calculate loss with weighted loss and L2 regularization
            loss = distillation_loss(outputs, teacher_outputs, labels, student_model, weight=batch_weights)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save the model after every batch to 'glaucoma_student_model_finetuned.pth'
            torch.save(student_model.state_dict(), 'checkpoints/glaucoma_student_model_finetuned.pth')
            logging.info(f"Model updated in glaucoma_student_model_finetuned.pth after batch {batch_idx + 1} of epoch {epoch + 1}.")

        # Log the completion of the epoch
        logging.info(f"Epoch {epoch + 1} completed.")

    # Save the fully trained model after all epochs are completed
    torch.save(student_model.state_dict(), model_save_path)
    logging.info(f"Fully trained model saved at {model_save_path}.")
