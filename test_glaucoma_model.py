import os
import torch
import matplotlib.pyplot as plt  # Ensure you import matplotlib
from torchvision import transforms
from PIL import Image
from fpd_student_model import StudentModelFPD  # Ensure this import matches your model file

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained student model
model_path = 'checkpoints/glaucoma_student_model_finetuned.pth'  # Path to your trained model
student_model = StudentModelFPD().to(device)
student_model.load_state_dict(torch.load(model_path))
student_model.eval()  # Set the model to evaluation mode

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size of your model
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])


def load_image(image_path):
    """Load and preprocess the image."""
    if os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')  # Open the image and convert to RGB
        return transform(image).unsqueeze(0).to(device)  # Transform and add batch dimension
    else:
        raise FileNotFoundError(f"Image not found at: {image_path}")


def apply_activation(outputs):
    """Apply activation functions to the model outputs."""
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()
    tanh = torch.nn.Tanh()

    activated_relu = relu(outputs)
    activated_sigmoid = sigmoid(outputs)
    activated_tanh = tanh(outputs)

    return activated_relu, activated_sigmoid, activated_tanh


def predict_glaucoma(image_tensor):
    """Make a prediction on the input image tensor."""
    with torch.no_grad():  # Disable gradient calculation
        outputs = student_model(image_tensor)  # Forward pass

        # Apply activation functions
        activated_relu, activated_sigmoid, activated_tanh = apply_activation(outputs)

        # Softmax for final classification
        probabilities = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        predicted_class = torch.argmax(probabilities, dim=1)  # Get the predicted class

        return predicted_class, probabilities, activated_relu, activated_sigmoid, activated_tanh


def plot_activations(activations, title):
    """Plot activation function outputs."""
    activations = activations.squeeze().cpu().numpy()  # Convert to NumPy array

    plt.figure(figsize=(10, 5))
    plt.plot(activations, marker='o')
    plt.title(f'{title} Activation Outputs')
    plt.xlabel('Output Index')
    plt.ylabel('Activation Value')
    plt.grid(True)
    plt.show()  # Display the figure


def main(test_image_path):
    """Main function to load image and predict."""
    try:
        image_tensor = load_image(test_image_path)
        predicted_class, probabilities, activated_relu, activated_sigmoid, activated_tanh = predict_glaucoma(
            image_tensor)

        # Map predicted class to label
        result = "Glaucoma Positive" if predicted_class.item() == 1 else "Glaucoma Negative"

        # Get the confidence level
        confidence = probabilities[0][predicted_class].item() * 100  # Convert to percentage

        # Print results
        print(f"The prediction for the image '{test_image_path}' is: {result}")
        print(f"Confidence level: {confidence:.2f}%")

        # Plot activation graphs
        plot_activations(activated_relu, "ReLU")
        plot_activations(activated_sigmoid, "Sigmoid")
        plot_activations(activated_tanh, "Tanh")

    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    test_image_path = 'Test_Image/036.jpg'  # Path to your test image
    test_image_path = 'Test_Image/5.jpg'  # Path to your test image

    main(test_image_path)