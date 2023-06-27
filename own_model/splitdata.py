import os
import random
import shutil

# Set the path to your dataset
dataset_path = 'train/'

# Set the path for the training and validation sets
train_path = 'dataset/train/'
validation_path = 'dataset/validation/'

# Set the split ratio
split_ratio = 0.7

# Create directories for training and validation sets
os.makedirs(train_path, exist_ok=True)
os.makedirs(validation_path, exist_ok=True)
os.makedirs(os.path.join(train_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_path, 'labels'), exist_ok=True)
os.makedirs(os.path.join(validation_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(validation_path, 'labels'), exist_ok=True)

# Get the list of image files in the dataset folder
image_files = [f for f in os.listdir(os.path.join(dataset_path, 'images')) if (f.endswith('.jpg') or (f.endswith('.jpeg')) or (f.endswith('.png')))]

# Randomly shuffle the image files
random.shuffle(image_files)

# Calculate the split index
split_index = int(split_ratio * len(image_files))

# Split the dataset into training and validation sets
train_files = image_files[:split_index]
validation_files = image_files[split_index:]

# Copy images and labels to the training set
for file in train_files:
    file_name = os.path.splitext(file)[0]
    shutil.copyfile(os.path.join(dataset_path, 'images', file), os.path.join(train_path, 'images', file))
    shutil.copyfile(os.path.join(dataset_path, 'labels', file_name + '.txt'), os.path.join(train_path, 'labels', file_name + '.txt'))

# Copy images and labels to the validation set
for file in validation_files:
    file_name = os.path.splitext(file)[0]
    shutil.copyfile(os.path.join(dataset_path, 'images', file), os.path.join(validation_path, 'images', file))
    shutil.copyfile(os.path.join(dataset_path, 'labels', file_name + '.txt'), os.path.join(validation_path, 'labels', file_name + '.txt'))

print('Dataset split into training and validation sets.')
