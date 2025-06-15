import torch
import torch.nn.functional as F

def distillation_loss(student_output, teacher_output, labels, student_model, temperature=2.0, weight=None, lambda_l2=0.01):
    # Apply softmax to outputs if they are logits
    student_output = F.log_softmax(student_output / temperature, dim=1)
    teacher_output = F.softmax(teacher_output / temperature, dim=1)

    # Calculate mean squared error between student and teacher outputs
    mse_loss = F.mse_loss(student_output, teacher_output, reduction='none')  # 'none' to keep the same shape

    # Apply weighted loss if weights are provided
    if weight is not None:
        # Ensure weight matches the batch size
        mse_loss = mse_loss * weight.view(-1, 1)  # Expand weight to match mse_loss shape
        mse_loss = mse_loss.mean()  # Average over the batch

    # Add L2 regularization on student model parameters
    l2_reg = sum(torch.norm(param) ** 2 for param in student_model.parameters())  # L2 norm of the weights
    total_loss = mse_loss + lambda_l2 * l2_reg  # Combine losses

    return total_loss
