import argparse, os, time, cv2, numpy as np, onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "weights", "best.onnx")

def letterbox(img, new_shape=640, color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, (dw, dh)


def nms(boxes, scores, iou_threshold=0.45):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return np.array(keep)


def detect(session, img_path, img_size=640, conf_thresh=0.25, iou_thresh=0.45):
    img0 = cv2.imread(img_path)
    if img0 is None:
        return None, [], 0
    img, ratio, (dw, dh) = letterbox(img0, img_size)
    inp = np.expand_dims(img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0, 0)
    t0 = time.time()
    pred = session.run(None, {session.get_inputs()[0].name: inp})[0]
    ms = (time.time() - t0) * 1000
    if pred.ndim == 3:
        pred = pred[0]
    scores = pred[:, 4] * (pred[:, 5:].max(1) if pred.shape[1] > 5 else 1)
    mask = scores > conf_thresh
    pred, scores = pred[mask], scores[mask]
    if len(pred) == 0:
        return img0, [], ms
    boxes = np.column_stack([pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2,
                             pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2])
    keep = nms(boxes, scores, iou_thresh)
    boxes, scores = boxes[keep], scores[keep]
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dw) / ratio, 0, img0.shape[1])
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dh) / ratio, 0, img0.shape[0])
    return img0, [(b[0], b[1], b[2], b[3], s) for b, s in zip(boxes, scores)], ms


def main():
    p = argparse.ArgumentParser(description="Trypanosome detection (CPU)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to ONNX model (auto-detected if not specified)")
    p.add_argument("--source", required=True, help="Image file or folder")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--output", default="results", help="Output directory")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        print(f"Make sure best.onnx is in the weights/ folder next to this script.")
        return

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    images = ([os.path.join(args.source, f) for f in sorted(os.listdir(args.source))
               if f.lower().endswith(exts)] if os.path.isdir(args.source) else [args.source])

    print(f"Processing {len(images)} image(s)...\n")
    total_ms, total_det = 0, 0
    for path in images:
        img, dets, ms = detect(session, path, args.img_size, args.conf, args.iou)
        if img is None:
            continue
        total_ms += ms
        total_det += len(dets)
        for (x1, y1, x2, y2, c) in dets:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            label = f"trypanosome {c:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (int(x1), int(y1) - th - 6), (int(x1) + tw, int(y1)), (0, 0, 255), -1)
            cv2.putText(img, label, (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(args.output, os.path.basename(path)), img)
        print(f"  {os.path.basename(path)}: {len(dets)} detections, {ms:.0f}ms")

    print(f"\n{total_det} detections across {len(images)} images.")
    print(f"Avg inference: {total_ms / max(len(images), 1):.0f}ms/image")
    print(f"Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
