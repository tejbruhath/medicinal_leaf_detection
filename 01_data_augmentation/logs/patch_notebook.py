import json

# Provide the content replacing the TF cell
new_source = [
    "import os\n",
    "import random\n",
    "import uuid\n",
    "from pathlib import Path\n",
    "from PIL import Image\n",
    "try:\n",
    "    import torchvision.transforms.v2 as v2\n",
    "except ImportError:\n",
    "    v2 = None\n",
    "\n",
    "if v2 is None:\n",
    "    raise ImportError(\"torchvision is required. Please 'pip install torchvision' as part of the PyTorch installation.\")\n",
    "\n",
    "# Torchvision equivalent for augmentation\n",
    "augment_transform = v2.Compose([\n",
    "    v2.RandomAffine(\n",
    "        degrees=30,\n",
    "        translate=(0.2, 0.2),\n",
    "        scale=(0.8, 1.2),\n",
    "        shear=11.45,\n",
    "        interpolation=v2.InterpolationMode.NEAREST\n",
    "    ),\n",
    "    v2.RandomHorizontalFlip(p=0.5),\n",
    "])\n",
    "\n",
    "IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')\n",
    "\n",
    "def list_images(directory):\n",
    "    return [\n",
    "        f for f in os.listdir(directory)\n",
    "        if f.lower().endswith(IMAGE_EXTENSIONS)\n",
    "    ]\n",
    "\n",
    "# Function to randomly select an input image and generate augmented samples\n",
    "def augment_images_from_random_file(input_class_dir, output_class_dir, num_samples=5):\n",
    "    os.makedirs(output_class_dir, exist_ok=True)\n",
    "    image_files = [f for f in list_images(input_class_dir) if not f.startswith('aug_')]\n",
    "    if not image_files:\n",
    "        image_files = list_images(input_class_dir)\n",
    "    \n",
    "    if not image_files:\n",
    "        raise ValueError(f\"No image files found in the directory: {input_class_dir}\")\n",
    "    \n",
    "    # Randomly choose an image\n",
    "    random_image = random.choice(image_files)\n",
    "    \n",
    "    # Load the image\n",
    "    img_path = os.path.join(input_class_dir, random_image)\n",
    "    img = Image.open(img_path).convert('RGB')\n",
    "    \n",
    "    # Generate and save 'num_samples' augmented images\n",
    "    for _ in range(num_samples):\n",
    "        aug_img = augment_transform(img)\n",
    "        unique_id = uuid.uuid4().hex[:8]\n",
    "        out_filename = f\"aug_{unique_id}.jpeg\"\n",
    "        out_path = os.path.join(output_class_dir, out_filename)\n",
    "        aug_img.save(out_path, format='JPEG', quality=95)\n",
    "    \n",
    "    print(f\"Generated {num_samples} augmented images from {random_image}\")\n",
    "\n",
    "# Main augmentation loop\n",
    "base_dir = './Medicinal_Leaves'  # Input folder containing class subfolders\n",
    "output_dir = './result'  # Output folder for generated augmented images\n",
    "target_image_count = 500  # Target number of images per class\n",
    "os.makedirs(output_dir, exist_ok=True)\n",
    "\n",
    "for class_name in os.listdir(base_dir):\n",
    "    class_dir = os.path.join(base_dir, class_name)\n",
    "    if not os.path.isdir(class_dir):\n",
    "        continue\n",
    "\n",
    "    output_class_dir = os.path.join(output_dir, class_name)\n",
    "    os.makedirs(output_class_dir, exist_ok=True)\n",
    "\n",
    "    # Count input images plus any generated images already present in result/<class_name>\n",
    "    current_image_count = len(list_images(class_dir)) + len(list_images(output_class_dir))\n",
    "    \n",
    "    if current_image_count < target_image_count:\n",
    "        # Calculate how many images need to be generated\n",
    "        images_needed = target_image_count - current_image_count\n",
    "        \n",
    "        while images_needed > 0:\n",
    "            # Generate only the remaining number needed, up to 5 at a time\n",
    "            batch_size = min(5, images_needed)\n",
    "            augment_images_from_random_file(class_dir, output_class_dir, num_samples=batch_size)\n",
    "            \n",
    "            # Recount actual files written before deciding whether to continue\n",
    "            current_image_count = len(list_images(class_dir)) + len(list_images(output_class_dir))\n",
    "            images_needed = target_image_count - current_image_count\n",
    "            \n",
    "    print(f\"{class_name}: {len(list_images(output_class_dir))} generated images in {output_class_dir}\")\n",
    "\n",
    "print(\"Augmentation completed!\")\n"
]

files = [
    '/home/tej/code/tobacco/Copy of Integration of Traditional Knowledge and Modern Science/Integration of Traditional Knowledge and Modern Science/Integration of Traditional Knowledge M1/Agumentation.ipynb',
    '/home/tej/code/tobacco/Copy of Integration of Traditional Knowledge and Modern Science/Integration of Traditional Knowledge and Modern Science/Integration of Traditional Knowledge M1/Agumentation-Copy1.ipynb'
]

for notebook_path in files:
    try:
        with open(notebook_path, 'r') as f:
            data = json.load(f)
            
        for cell in data['cells']:
            if cell['cell_type'] == 'code' and 'import os\\n' in cell['source']:
                if any('tensorflow' in s for s in cell['source']):
                    cell['source'] = new_source
                    cell['outputs'] = []
                    
        with open(notebook_path, 'w') as f:
            json.dump(data, f, indent=1)
            
        print(f"Patched {notebook_path}")
    except Exception as e:
        print(f"Failed to patch {notebook_path}: {e}")
