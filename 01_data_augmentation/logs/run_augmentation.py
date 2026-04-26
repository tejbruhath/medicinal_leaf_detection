import sys
import os
import random
import uuid
from pathlib import Path
from PIL import Image

# Redirect all print to file for our bash log requirement
os.makedirs("logs/bash_logs", exist_ok=True)
log_file = open("logs/bash_logs/augmentation_run.md", "a")
sys.stdout = log_file
sys.stderr = log_file

print("Starting custom PyTorch-equivalent augmentation script.")

try:
    import torchvision.transforms.v2 as v2
except ImportError:
    print("torchvision is missing. Please ensure PyTorch is fully installed.")
    sys.exit(1)

# Torchvision equivalent for augmentation
augment_transform = v2.Compose([
    v2.RandomAffine(
        degrees=30,
        translate=(0.2, 0.2),
        scale=(0.8, 1.2),
        shear=11.45,
        interpolation=v2.InterpolationMode.NEAREST
    ),
    v2.RandomHorizontalFlip(p=0.5),
])

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

def list_images(directory):
    return [
        f for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

# Function to randomly select an input image and generate augmented samples
def augment_images_from_random_file(input_class_dir, output_class_dir, num_samples=5):
    os.makedirs(output_class_dir, exist_ok=True)
    image_files = [f for f in list_images(input_class_dir) if not f.startswith('aug_')]
    if not image_files:
        image_files = list_images(input_class_dir)
    
    if not image_files:
        raise ValueError(f"No image files found in the directory: {input_class_dir}")
    
    # Randomly choose an image
    random_image = random.choice(image_files)
    
    # Load the image
    img_path = os.path.join(input_class_dir, random_image)
    img = Image.open(img_path).convert('RGB')
    
    # Generate and save 'num_samples' augmented images
    for _ in range(num_samples):
        aug_img = augment_transform(img)
        unique_id = uuid.uuid4().hex[:8]
        out_filename = f"aug_{unique_id}.jpeg"
        out_path = os.path.join(output_class_dir, out_filename)
        aug_img.save(out_path, format='JPEG', quality=95)
    
    print(f"Generated {num_samples} augmented images from {random_image} to {output_class_dir}")

# Main augmentation loop
base_dir = './Medicinal_Leaves'
output_dir = './result'
target_image_count = 500
os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(base_dir):
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    current_image_count = len(list_images(class_dir)) + len(list_images(output_class_dir))
    
    if current_image_count < target_image_count:
        images_needed = target_image_count - current_image_count
        while images_needed > 0:
            batch_size = min(5, images_needed)
            augment_images_from_random_file(class_dir, output_class_dir, num_samples=batch_size)
            current_image_count = len(list_images(class_dir)) + len(list_images(output_class_dir))
            images_needed = target_image_count - current_image_count
            
    print(f"{class_name}: {len(list_images(output_class_dir))} generated images in {output_class_dir}")

print("Augmentation completed!")

log_file.close()
