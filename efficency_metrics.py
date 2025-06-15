import torch
import torch.nn as nn
from fpd_student_model import StudentModelFPD  # Import your model from fpd_student_model.py
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to calculate model parameters
def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to calculate GFLOPS
def calculate_gflops(model, input_size):
    model.eval()
    input_tensor = torch.randn(*input_size).to(next(model.parameters()).device)

    # Measure the time taken for a forward pass
    flops = 0

    def count_flops(layer):
        nonlocal flops
        if isinstance(layer, nn.Conv2d):
            output_height = (input_size[2] - layer.kernel_size[0] + 2 * layer.padding[0]) // layer.stride[0] + 1
            output_width = (input_size[3] - layer.kernel_size[1] + 2 * layer.padding[1]) // layer.stride[1] + 1
            flops += layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * output_height * output_width
        for child in layer.children():
            count_flops(child)

    count_flops(model)
    return flops / 1e9  # Convert to GFLOPS

# Main function
if __name__ == "__main__":
    model = StudentModelFPD().to("cuda" if torch.cuda.is_available() else "cpu")  # Load model
    model_params = get_model_params(model)
    gflops = calculate_gflops(model, (1, 3, 224, 224))  # Example input size for an image

    logging.info(f"Model Parameters: {model_params}")
    logging.info(f"Model GFLOPS: {gflops}")
