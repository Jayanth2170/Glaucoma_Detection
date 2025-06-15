import torch
from torchvision import transforms
from PIL import Image
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from fpd_student_model import StudentModelFPD  # Make sure to import your model class

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
    model = StudentModelFPD()  # Initialize your model class
    model.load_state_dict(torch.load(model_path))  # Load the trained model
    model.eval()  # Set the model to evaluation mode
    return model

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Load the image
    return image

def calculate_fps(model, image_path, image_size, num_iterations=100):
    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(image_size),  # Resize the image to the specified input size
        transforms.ToTensor(),  # Convert image to tensor
    ])

    # Load and preprocess the image
    image = preprocess(load_image(image_path)).unsqueeze(0)  # Add batch dimension
    model.eval()  # Ensure model is in evaluation mode

    # Warm up the model
    with torch.no_grad():
        for _ in range(10):  # Warm-up iterations
            _ = model(image)

    # Measure the time for inference over num_iterations
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(image)  # Run inference
    end_time = time.time()

    # Calculate FPS
    fps = num_iterations / (end_time - start_time)
    return fps

def analyze_performance(model, image_path, sizes):
    fps_results = []

    for size in sizes:
        logging.info(f"Testing with input size: {size}")
        fps = calculate_fps(model, image_path, size)
        fps_results.append(fps)
        logging.info(f"FPS for size {size}: {fps:.2f}")

    return fps_results

def plot_results(sizes, fps_results):
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, fps_results, marker='o')
    plt.title('FPS vs Input Image Size')
    plt.xlabel('Input Image Size (Height, Width)')
    plt.ylabel('Frames Per Second (FPS)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.savefig('fps_analysis.png')  # Save the plot as an image
    plt.show()  # Display the plot

if __name__ == "__main__":
    model_path = 'checkpoints/glaucoma_student_model_finetuned.pth'  # Path to your model
    image_path = 'C:\\Users\\mohan\\PycharmProjects\\practice\\Test_Image\\5.jpg'  # Path to the test image
    image_path = 'C:\\Users\\mohan\\PycharmProjects\\practice\\Test_Image\\036.jpg'  # Path to the test image

    logging.info("Loading model...")
    model = load_model(model_path)  # Load the model
    logging.info("Model loaded successfully.")

    # Define a list of input sizes to test (width, height)
    sizes = [(224, 224), (256, 256), (300, 300), (320, 320), (384, 384), (512, 512)]

    # Analyze performance and plot results
    fps_results = analyze_performance(model, image_path, sizes)
    plot_results(sizes, fps_results)
