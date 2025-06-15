import torch
from torchvision import transforms
from PIL import Image
import time
import logging
import matplotlib.pyplot as plt
from fpd_student_model import StudentModelFPD  # Make sure to import your model class

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    model = StudentModelFPD()  # Initialize your model class
    model.load_state_dict(torch.load(model_path))  # Load the trained model
    model.eval()  # Set the model to evaluation mode
    return model.to(device)  # Move model to device


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Load and convert to RGB
    return image


def evaluate_accuracy(model, dataset):
    correct = 0
    total = len(dataset)

    for image_path, label in dataset:
        image = load_image(image_path)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the input size
            transforms.ToTensor(),  # Convert to tensor
        ])
        image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        with torch.no_grad():
            output = model(image)  # Run the model
            _, predicted = torch.max(output, 1)  # Get the predicted class

        correct += (predicted.item() == label)  # Count correct predictions

    accuracy = correct / total
    return accuracy


def plot_results(labels, accuracies):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, accuracies, color=['blue', 'orange'])
    plt.ylim(0, 1)  # Set y-axis limit to 1 for accuracy
    plt.title('Classification Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.grid(axis='y')
    plt.savefig('classification_accuracy.png')  # Save the plot as an image
    plt.show()  # Display the plot


if __name__ == "__main__":
    model_path = 'checkpoints/glaucoma_student_model_finetuned.pth'  # Path to your model

    logging.info("Loading model...")
    model = load_model(model_path)  # Load the model
    logging.info("Model loaded successfully.")

    # Slide-Level Dataset
    slide_level_dataset = [
        (r'C:\Users\mohan\PycharmProjects\practice\Test_Image\5.jpg',0),  # Healthy
        (r'C:\Users\mohan\PycharmProjects\practice\Test_Image\036.jpg',1),  # Glaucoma positive
        (r'C:\Users\mohan\PycharmProjects\practice\Test_Image\160.jpg',1)  # Glaucoma positive
    ]

    # Patch-Level Dataset (if applicable, use similar labeling)
    patch_level_dataset = [
        (r'C:\Users\mohan\PycharmProjects\practice\Test_Image\5.jpg',0),  # Healthy
        (r'C:\Users\mohan\PycharmProjects\practice\Test_Image\036.jpg',0),  # Glaucoma positive
        (r'C:\Users\mohan\PycharmProjects\practice\Test_Image\160.jpg',1)  # Glaucoma positive
        # Add more patches if you have them
    ]

    # Evaluate slide-level accuracy
    slide_accuracy = evaluate_accuracy(model, slide_level_dataset)
    logging.info(f"Slide-Level Classification Accuracy: {slide_accuracy:.2f}")

    # Evaluate patch-level accuracy (if applicable)
    patch_accuracy = evaluate_accuracy(model, patch_level_dataset)
    logging.info(f"Patch-Level Classification Accuracy: {patch_accuracy:.2f}")

    # Plot the results
    labels = ['Slide-Level', 'Patch-Level']
    accuracies = [slide_accuracy, patch_accuracy]
    plot_results(labels, accuracies)
