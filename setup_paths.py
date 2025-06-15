# setup_paths.py
import os

def get_paths():
    base_dir = r'C:\Users\mohan\PycharmProjects\practice\Fundus_Train_Val_Data\Fundus_Scanes_Sorted'  # Use raw string to handle backslashes correctly

    train_dir = os.path.join(base_dir, 'Train')
    val_dir = os.path.join(base_dir, 'Validation')

    train_glaucoma_pos = os.path.join(train_dir, 'Glaucoma_Positive')
    train_glaucoma_neg = os.path.join(train_dir, 'Glaucoma_Negative')

    val_glaucoma_pos = os.path.join(val_dir, 'Glaucoma_Positive')
    val_glaucoma_neg = os.path.join(val_dir, 'Glaucoma_Negative')

    return train_glaucoma_pos, train_glaucoma_neg, val_glaucoma_pos, val_glaucoma_neg
