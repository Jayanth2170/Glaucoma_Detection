import os
import torch
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch.nn as nn
from data_loader import load_data  # Ensure you have a data loader
from fpd_student_model import StudentModelFPD  # Import your student model
from distillation_loss import distillation_loss  # Import the loss function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load paths (make sure to set the correct paths)
train_glaucoma_pos = 'C:\\Users\\mohan\\PycharmProjects\\practice\\Fundus_Train_Val_Data\\Fundus_Scanes_Sorted\\Train\\Glaucoma_Positive'
train_glaucoma_neg = 'C:\\Users\\mohan\\PycharmProjects\\practice\\Fundus_Train_Val_Data\\Fundus_Scanes_Sorted\\Train\\Glaucoma_Negative'
val_glaucoma_pos = 'C:\\Users\\mohan\\PycharmProjects\\practice\\Fundus_Train_Val_Data\\Fundus_Scanes_Sorted\\Validation\\Glaucoma_Positive'
val_glaucoma_neg = 'C:\\Users\\mohan\\PycharmProjects\\practice\\Fundus_Train_Val_Data\\Fundus_Scanes_Sorted\\Validation\\Glaucoma_Negative'

# Load data
logging.info("Loading validation data...")
_, val_loader = load_data(train_glaucoma_pos, train_glaucoma_neg, val_glaucoma_pos, val_glaucoma_neg)

# Load the trained student model
model_save_path = 'checkpoints/glaucoma_student_model_finetuned.pth'
student_model = StudentModelFPD().to(device)

if os.path.exists(model_save_path):
    student_model.load_state_dict(torch.load(model_save_path))
    logging.info(f"Trained student model loaded from {model_save_path}.")
else:
    logging.error("Trained model not found, please ensure the path is correct.")

# Load the pre-trained teacher model (ResNet50)
teacher_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 2).to(device)  # Modify output layer

# Function to calculate VFD performance
def calculate_vfd_performance(student_outputs, teacher_outputs):
    student_probs = torch.softmax(student_outputs, dim=1)
    teacher_probs = torch.softmax(teacher_outputs, dim=1)

    # Debug statements to check outputs
    logging.info(f"Raw Student Outputs: {student_outputs}")
    logging.info(f"Raw Teacher Outputs: {teacher_outputs}")
    logging.info(f"Softmax Student Probabilities: {student_probs}")
    logging.info(f"Softmax Teacher Probabilities: {teacher_probs}")

    # Calculate VFD Loss
    vfd_loss = nn.BCELoss()(student_probs, teacher_probs.detach())
    logging.info(f"VFD Loss with BCELoss: {vfd_loss.item()}")

    return vfd_loss

def compare_vfd_fpd(student_model, teacher_model, val_loader):
    student_model.eval()
    teacher_model.eval()

    vfd_losses = []
    fpd_losses = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            student_outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)

            vfd_loss = calculate_vfd_performance(student_outputs, teacher_outputs)

            fpd_loss = distillation_loss(student_outputs, teacher_outputs, labels, student_model)
            if fpd_loss.numel() > 1:  # Check if the tensor has more than one element
                fpd_loss = fpd_loss.mean()  # Take the mean to convert it to a scalar

            vfd_losses.append(vfd_loss)
            fpd_losses.append(fpd_loss.item())  # This should now be a scalar

    logging.info(f"Average VFD Loss: {np.mean(vfd_losses)}, Average FPD Loss: {np.mean(fpd_losses)}")
    return np.mean(vfd_losses), np.mean(fpd_losses)

# Function to visualize the comparison
def visualize_comparison(vfd_loss, fpd_loss):
    plt.bar(['VFD', 'FPD'], [vfd_loss, fpd_loss], color=['blue', 'orange'])
    plt.ylabel('Loss')
    plt.title('Comparison of Distillation Techniques')
    plt.show()

# Main execution
if __name__ == "__main__":
    logging.info("Starting comparison of VFD and FPD...")
    vfd_loss, fpd_loss = compare_vfd_fpd(student_model, teacher_model, val_loader)

    logging.info(f"VFD Loss: {vfd_loss}, FPD Loss: {fpd_loss}")
    visualize_comparison(vfd_loss, fpd_loss)