import cv2
import matplotlib.pyplot as plt

# === Change these paths to your actual image and label file ===
image_path = "preprocessed/images/test/positive_video_032_00000292.jpg"
label_path = "preprocessed/labels/test/positive_video_032_00000292.txt"

# Read the image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# Read YOLO label file
with open(label_path, "r") as f:
    boxes = [line.strip().split() for line in f.readlines()]

# Draw boxes
for box in boxes:
    cls, x_center, y_center, box_w, box_h = map(float, box)
    
    # Convert from normalized to pixel coordinates
    x_center, y_center, box_w, box_h = x_center * w, y_center * h, box_w * w, box_h * h
    
    # Calculate top-left and bottom-right corners
    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)
    
    # Draw rectangle and class label
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.putText(img, f"{int(cls)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# Show the image
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis("off")
plt.show()