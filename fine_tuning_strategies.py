import torch
from distillation_loss import distillation_loss
def fine_tune_model(strategy, model, teacher_model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    # Freeze layers based on strategy
    if strategy == 'Reuse CLAM':
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = False

    elif strategy == 'Retrain CLAM':
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    elif strategy == 'End2End Train CLAM (ETC)':
        pass  # No freezing; fine-tune both feature extractor and classifier

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
            optimizer.zero_grad()

            # Forward pass through the student model
            student_outputs = model(inputs)
            with torch.no_grad():
                # Forward pass through the teacher model for distillation
                teacher_outputs = teacher_model(inputs)

            # Calculate distillation loss
            loss = distillation_loss(student_outputs, teacher_outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

        # Optionally validate after every epoch
        validate_model(model, val_loader)

def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total}%')
