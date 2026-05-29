import json
import os

def coco_to_yolo(coco_json_path, output_dir):
    """
    Convert COCO-format JSON annotations to YOLOv7 label files.
    """

    os.makedirs(output_dir, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Map image_id -> image info
    images = {img["id"]: img for img in coco["images"]}

    # Map category_id -> 0-based class index
    categories = sorted(coco["categories"], key=lambda x: x["id"])
    category_map = {cat["id"]: idx for idx, cat in enumerate(categories)}

    # Prepare empty label lists per image
    labels = {img_id: [] for img_id in images}

    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        bbox = ann["bbox"]  # [x_min, y_min, width, height]

        img = images[image_id]
        img_w = img["width"]
        img_h = img["height"]

        x_min, y_min, bw, bh = bbox

        # Convert to YOLO format
        x_center = (x_min + bw / 2) / img_w
        y_center = (y_min + bh / 2) / img_h
        w = bw / img_w
        h = bh / img_h

        class_id = category_map[category_id]

        labels[image_id].append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        )

    # Write YOLO label files
    for image_id, lines in labels.items():
        img = images[image_id]
        image_name = os.path.splitext(img["file_name"])[0]
        label_path = os.path.join(output_dir, image_name + ".txt")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Conversion complete. Labels saved to: {output_dir}")


# --------- USAGE EXAMPLE ----------
if __name__ == "__main__":
    coco_json = "test.json"      # path to COCO annotation file
    yolo_labels = "labels"              # output directory

    coco_to_yolo(coco_json, yolo_labels)
