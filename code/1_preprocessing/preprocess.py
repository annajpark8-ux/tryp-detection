import os
import shutil
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
import random

# Paths
base_dir = "tog"
positive_dir = os.path.join(base_dir, "positive_images")
negative_dir = os.path.join(base_dir, "negative_images")

output_dir = "preprocessed"
os.makedirs(output_dir, exist_ok=True)

splits = ["train", "val", "test"]
#img_size = (416, 416)
'''
def augment_image(img):
    """Randomly augment the image"""
    # Flip horizontally
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Flip vertically
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # Rotate randomly by -15 to 15 degrees
    #angle = random.uniform(-15, 15)
    #img = img.rotate(angle)
    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    return img
'''
# --- Helper to create folders ---
def create_split_dirs(split_name):
    img_path = os.path.join(output_dir, "images", split_name)
    lbl_path = os.path.join(output_dir, "labels", split_name)
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(lbl_path, exist_ok=True)
    return img_path, lbl_path

# --- Step 1: Handle negative images ---
neg_imgs = [f for f in os.listdir(negative_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
train_neg, test_neg = train_test_split(neg_imgs, test_size=0.4, random_state=42)
val_neg, test_neg = train_test_split(test_neg, test_size=0.5, random_state=42)
neg_split = {"train": train_neg, "val": val_neg, "test": test_neg}

for split_name, img_list in neg_split.items():
    img_out, lbl_out = create_split_dirs(split_name)
    for img_file in img_list:
        src = os.path.join(negative_dir, img_file)
        # Open original image
        img = Image.open(src).convert("RGB")
        orig_img_dst = os.path.join(img_out, img_file)
        img.save(orig_img_dst)

        orig_lbl_dst = os.path.join(lbl_out, os.path.splitext(img_file)[0] + ".txt")
        open(orig_lbl_dst, "w").close()
        '''
        # Create 7 augmented copies
        for i in range(7):
            aug_img = augment_image(img)
            dst = os.path.join(img_out, f"{os.path.splitext(img_file)[0]}_aug{i}.jpg")
            aug_img.save(dst)
            # Create empty label for each copy
            open(os.path.join(lbl_out, f"{os.path.splitext(img_file)[0]}_aug{i}.txt"), "w").close()
        '''

# --- Step 2: Handle positive images ---
for split_name in splits:
    pos_img_dir = os.path.join(positive_dir, split_name, "images")
    pos_lbl_dir = os.path.join(positive_dir, split_name, "labels")
    out_img_dir, out_lbl_dir = create_split_dirs(split_name)
    
    pos_imgs = [f for f in os.listdir(pos_img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    for img_file in pos_imgs:
        # Copy & resize image
        src_img = os.path.join(pos_img_dir, img_file)
        dst_img = os.path.join(out_img_dir, img_file)
        img = Image.open(src_img).convert("RGB")
        #img = img.resize(img_size)
        img.save(dst_img)

        # Copy label
        src_lbl = os.path.join(pos_lbl_dir, os.path.splitext(img_file)[0] + ".txt")
        dst_lbl = os.path.join(out_lbl_dir, os.path.splitext(img_file)[0] + ".txt")
        shutil.copy(src_lbl, dst_lbl)
