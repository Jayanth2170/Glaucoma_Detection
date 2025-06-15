import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the saved model
model_save_path = 'checkpoints/glaucoma_student_model_finetuned.pth'

# Check if the trained model exists
if os.path.exists(model_save_path):
    logging.info(f"Trained model found at {model_save_path}.")
else:
    logging.warning(f"Trained model not found at {model_save_path}. You may need to train the model.")
