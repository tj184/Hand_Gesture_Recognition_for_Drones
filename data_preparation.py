import os
import shutil
import cv2
import random

image_folder = 'palm_images'
train_folder = 'dataset/images/train'
val_folder = 'dataset/images/val'
train_labels_folder = 'dataset/labels/train'
val_labels_folder = 'dataset/labels/val'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

print(f"Image folder: {image_folder}")
print(f"Training images folder: {train_folder}")
print(f"Validation images folder: {val_folder}")
print(f"Training labels folder: {train_labels_folder}")
print(f"Validation labels folder: {val_labels_folder}")

bbox_width = 0.4
bbox_height = 0.4
bbox_center_x = 0.5
bbox_center_y = 0.5

def generate_yolo_label(image_path, label_folder, image_name):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    x_center = bbox_center_x
    y_center = bbox_center_y
    w = bbox_width
    h = bbox_height
    x_center_norm = x_center * width
    y_center_norm = y_center * height
    w_norm = w * width
    h_norm = h * height
    label = f"0 {x_center_norm / width} {y_center_norm / height} {w_norm / width} {h_norm / height}\n"
    label_path = os.path.join(label_folder, image_name.replace('.jpg', '.txt'))
    with open(label_path, 'w') as label_file:
        label_file.write(label)

image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

random.shuffle(image_files)

train_images = image_files[:int(0.8 * len(image_files))]
val_images = image_files[int(0.8 * len(image_files)):]

for img_file in train_images:
    img_path = os.path.join(image_folder, img_file)
    shutil.copy(img_path, train_folder)
    generate_yolo_label(img_path, train_labels_folder, img_file)
    print(f"Copied {img_file} to train folder and generated label.")

for img_file in val_images:
    img_path = os.path.join(image_folder, img_file)
    shutil.copy(img_path, val_folder)
    generate_yolo_label(img_path, val_labels_folder, img_file)
    print(f"Copied {img_file} to val folder and generated label.")

print(f"Dataset preparation complete: {len(train_images)} train images, {len(val_images)} validation images.")
