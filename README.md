👁️ Glaucoma Detection Using Deep Learning on OCT Images

Glaucoma is a serious eye condition that can lead to irreversible blindness if not detected early. Traditional detection methods are often slow or lack the sensitivity needed for early diagnosis. This project presents an automated deep learning-based solution using **Optical Coherence Tomography (OCT)** and **fundus images** to detect glaucoma with high accuracy.


🎯 Objective

To develop a lightweight and accurate deep learning model that can assist in the **early diagnosis of glaucoma** by analyzing OCT and fundus images. The project utilizes C
convolutional Neural Networks with knowledge distillation to enable fast and efficient inference, making it suitable for real-world medical applications.


🧠 Model Architecture

- Teacher Model**: Pretrained ResNet50
- Student Model: Custom lightweight CNN called `StudentModelFPD`
- Technique: Knowledge distillation from teacher to student for better performance with fewer resources


🏋️ Training and Evaluation

- Loss Function: Custom loss combining classification and distillation objectives
- Optimizer: Adam
- Performance Tracking: Training and validation loss plotted across epochs
- Evaluation Metrics:
  - Accuracy
  - F1-Score
  - False Positive Detection (FPD)
  - Variable Feature Distillation (VFD) Loss


🔬 Distillation Strategy Comparison

We compare two knowledge distillation strategies:

- VFD (Variational Feature Distillation) 
- FPD (Feature Projection Distillation)

The student model's performance is analyzed for both in terms of:
- Convergence speed  
- Accuracy  
- Feature learning


📊 Visualization and Reporting

- Visual comparisons of training losses
- Accuracy graphs
- Confusion matrix


🔍 Fine-Tuning Strategies

To further improve model accuracy and generalization, we explore:
- Reuse CLAM
- Retrain CLAM
- End-to-End Train CLAM 


💡 Key Outcomes

- Improved early detection of glaucoma using AI
- Efficient and deployable CNN model for real-time diagnosis
- Enhanced explainability through visual activations and confidence outputs


