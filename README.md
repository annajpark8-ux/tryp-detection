# Lightweight Trypanosome Parasite Detection

A YOLOv7-tiny based detection system for *Trypanosoma brucei brucei* parasites in unstained thick blood smear microscopy images. Designed to run on standard laptop CPUs without GPU hardware.

**Author:** Anna Park
**Course:** Computer Systems Lab, Period 7 — Yilmaz

---

## Project Overview

This project trains a lightweight object detection model on the Tryp dataset (Anzaku et al., 2023) and exports it to ONNX format for CPU-only deployment. The final model is ~12 MB and runs on any laptop with three pip packages installed.

**Key Results:**
- 74.1% AP@0.5
- 74.9% Precision
- 70.2% Recall
- ~12 MB ONNX model size
- 6M parameters (6× smaller than full YOLOv7)

---

## Folder Structure

```
code/
├── 1_preprocessing/
│   ├── coco_to_yolo.py            # Convert COCO JSON to YOLO format labels
│   └── preprocess.py              # Organize dataset into train/val/test folders
├── 2_training/
│   ├── YOLOv7_Tiny_Trypanosome_Training.ipynb
│   └── tryp.yaml                  # Data config used by YOLOv7
├── 3_evaluation/
│   └── CPU_evaluation.ipynb       # Re-run evaluation on CPU to generate plots
├── tryp_yolov7_results/           # Deployment package
│   ├── weights/
│   │   ├── best.onnx              # Exported ONNX model for CPU inference
│   │   ├── best.pt                # Trained PyTorch weights
│   │   ├── best.torchscript.pt   
│   │   ├── init.pt                # Initial  weights
│   │   └── last.pt                # Most recent epoch weights
│   ├── tryp_detect.py             # CLI inference script (CPU, ONNX)
│   ├── app.py                     # Streamlit web interface
│   └── export_onnx.py             # ONNX exporter
└── utils/
    └── visualize_labels.py        # Visualize YOLO labels overlaid on images
```

---

## Pipeline Overview

```
Raw Tryp Dataset (COCO format)
        │
        ▼
[1_preprocessing/coco_to_yolo.py]  -> Convert annotations to YOLO format
        │
        ▼
[1_preprocessing/preprocess.py]    -> Split into train/val/test (3:1:1)
        │
        ▼
[2_training/...ipynb]              -> Train YOLOv7-tiny on Colab T4 GPU (100 epochs)
        │
        ▼
[2_training/...ipynb]              -> Export trained model to ONNX format
        │
        ▼
[3_evaluation/...ipynb]            -> Generate confusion matrix, F1 curve, PR curve
        │
        ▼
[tryp_yolov7_results/tryp_detect.py] -> Run on individual images via command line
[tryp_yolov7_results/app.py]         -> Run via Streamlit web interface
```

---

## Run Order

### 1. Preprocessing (one-time)

```bash
# Convert COCO JSON annotations to YOLO format
python 1_preprocessing/coco_to_yolo.py

# Organize images and labels into train/val/test folders
python 1_preprocessing/preprocess.py
```

This produces a `preprocessed/` folder with the structure:
```
preprocessed/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

### 2. Training (on Google Colab with GPU)

Open `2_training/YOLOv7_Tiny_Trypanosome_Training.ipynb` in Google Colab.

1. Set Runtime → GPU (T4)
2. Upload `preprocessed.zip` to Colab via the sidebar
3. Run all cells

Training takes ~2–3 hours for 100 epochs. Weights and the ONNX export are saved to Google Drive at `/MyDrive/tryp_yolov7_results/`.

### 3. Evaluation (CPU is sufficient)

Open `3_evaluation/CPU_evaluation.ipynb` in Colab. Re-runs test.py to regenerate the confusion matrix, F1 curve, and precision/recall curves. Takes 10–20 minutes on CPU.

### 4. Deployment (on local laptop)

Download the `tryp_yolov7_results/` folder from Google Drive to your laptop.

**One-time setup:**
```bash
pip install onnxruntime opencv-python numpy streamlit Pillow
```

**If you only have `best.pt` and need to export to ONNX:**
```bash
cd tryp_yolov7_results
python export_onnx.py
```

**Run detection on a single image:**
```bash
cd tryp_yolov7_results
python tryp_detect.py --source path/to/image.jpg
```

**Run detection on a folder of images:**
```bash
cd tryp_yolov7_results
python tryp_detect.py --source path/to/folder/
```

**Launch the web interface:**
```bash
cd tryp_yolov7_results
streamlit run app.py
```

The Streamlit app opens at `http://localhost:8501` in browser.

---

## File Descriptions

### Preprocessing
- **`coco_to_yolo.py`** — Parses the Tryp dataset's COCO-format JSON files and produces YOLO-style text label files (one per image, with normalized bounding box coordinates).
- **`preprocess.py`** — Organizes images and corresponding labels into `train/`, `val/`, and `test/` subdirectories.

### Training
- **`YOLOv7_Tiny_Trypanosome_Training.ipynb`** — Main training notebook. Clones YOLOv7, patches PyTorch 2.6+ compatibility issues, trains for 100 epochs, evaluates on the test set, exports to ONNX, and benchmarks CPU inference speed.
- **`tryp.yaml`** — YOLOv7 data config specifying paths and class names.
- **`yolov7-tiny-tryp.yaml`** — Modified YOLOv7-tiny model config (changed `nc: 80` → `nc: 1` for single-class detection).

### Evaluation
- **`CPU_evaluation.ipynb`** — Standalone notebook for regenerating evaluation plots (confusion matrix, F1 curve, PR curve) by re-running `test.py` on CPU.

### Deployment (`tryp_yolov7_results/`)
This entire folder is the self-contained deployment package. Copy it to any laptop to run inference.

- **`weights/best.pt`** — Trained PyTorch model weights.
- **`weights/best.onnx`** — ONNX-exported model for fast CPU inference.
- **`export_onnx.py`** — Self-contained script that exports `best.pt` to `best.onnx` on a CPU-only laptop. Clones YOLOv7 and handles all dependencies automatically.
- **`tryp_detect.py`** — Command-line inference tool. Loads the ONNX model, runs detection on an image or folder, draws bounding boxes, and saves annotated results.
- **`app.py`** — Streamlit web interface. Lets users upload images via a browser, adjust the confidence threshold, and view detection results side-by-side with the original image.

### Utilities
- **`visualize_labels.py`** — Helper script that overlays YOLO bounding box labels on an image. Used during dataset preparation to verify label correctness.

---

## Requirements

**For training (Colab):**
- Python 3.10+
- PyTorch (auto-installed via YOLOv7's requirements.txt)
- A free Google Colab T4 GPU runtime

**For deployment (any laptop):**
```
onnxruntime
opencv-python
numpy
streamlit       (for web interface only)
Pillow          (for web interface only)
```

No GPU required for deployment.

---

## Dataset

This project uses the **Tryp dataset** by Anzaku et al. (2023):
- 3,115 positive and 93 negative microscopy images
- Unstained thick blood smears
- Single class: *Trypanosoma brucei brucei*
- Original format: MS COCO JSON annotations
- Available at: https://doi.org/10.6084/m9.figshare.22825787

---

## References

1. E. T. Anzaku et al., "Tryp: A dataset of microscopy images of unstained thick blood smears for trypanosome detection," *Sci. Data*, vol. 10, p. 716, 2023.
2. C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors," arXiv:2207.02696, 2022.

---

## Acknowledgments

Built using the [YOLOv7](https://github.com/WongKinYiu/yolov7) framework by Wang et al. and the Tryp dataset by Anzaku et al. Special thanks to Mr. Yilmaz for project guidance.
