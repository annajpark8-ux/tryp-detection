import os, subprocess, sys, pathlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "tryp_yolov7_results", "weights")
BEST_PT = os.path.join(WEIGHTS_DIR, "best.pt")
BEST_ONNX = os.path.join(WEIGHTS_DIR, "best.onnx")
YOLOV7_DIR = os.path.join(SCRIPT_DIR, "yolov7")

# Check best.pt exists
if not os.path.exists(BEST_PT):
    print(f"ERROR: best.pt not found at {BEST_PT}")
    print("Make sure tryp_yolov7_results/weights/best.pt exists next to this script.")
    sys.exit(1)

if os.path.exists(BEST_ONNX):
    print(f"best.onnx already exists at {BEST_ONNX}")
    print("Delete it first if you want to re-export.")
    sys.exit(0)

# Clone YOLOv7 if needed
if not os.path.exists(YOLOV7_DIR):
    print("Cloning YOLOv7...")
    subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov7.git", YOLOV7_DIR], check=True)
else:
    print("YOLOv7 already cloned")

# Patch torch.load for PyTorch 2.6+ compatibility
print("Patching YOLOv7 for PyTorch 2.6+ compatibility...")
patched = 0
for f in pathlib.Path(YOLOV7_DIR).rglob("*.py"):
    txt = f.read_text()
    if "torch.load(" in txt and "weights_only" not in txt:
        patch = (
            "import torch as _torch\n"
            "_orig_load = _torch.load\n"
            '_torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})\n\n'
        )
        f.write_text(patch + txt)
        patched += 1
print(f"Patched {patched} files")

# Install requirements
print("Installing YOLOv7 requirements...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "matplotlib", "numpy", "opencv-python", "Pillow",
                "PyYAML", "scipy", "tqdm", "seaborn",
                "onnx", "onnxsim"], check=True)

# Run export
print(f"\nExporting {BEST_PT} to ONNX...")
result = subprocess.run(
    [sys.executable, "export.py",
     "--weights", BEST_PT,
     "--grid",
     "--simplify",
     "--img-size", "640", "640"],
    cwd=YOLOV7_DIR
)

# Check if ONNX was created next to best.pt
if os.path.exists(BEST_ONNX):
    size_mb = os.path.getsize(BEST_ONNX) / (1024 * 1024)
    print(f"\nDone! ONNX model saved to: {BEST_ONNX} ({size_mb:.1f} MB)")
    print(f"\nYou can now run:")
    print(f"  cd tryp_yolov7_results")
    print(f"  python tryp_detect.py --source your_image.jpg")
else:
    # Sometimes export saves it in the yolov7 dir or with a different path
    # Search for it
    for candidate in [
        os.path.join(YOLOV7_DIR, "best.onnx"),
        BEST_PT.replace(".pt", ".onnx"),
    ]:
        if os.path.exists(candidate):
            import shutil
            shutil.move(candidate, BEST_ONNX)
            size_mb = os.path.getsize(BEST_ONNX) / (1024 * 1024)
            print(f"\nDone! ONNX model saved to: {BEST_ONNX} ({size_mb:.1f} MB)")
            print(f"\nYou can now run:")
            print(f"  cd tryp_yolov7_results")
            print(f"  python tryp_detect.py --source your_image.jpg")
            sys.exit(0)

    print("\nERROR: ONNX export may have failed. Check the output above for errors.")
