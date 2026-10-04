"""SafetyVision - Helmet Detection using YOLOv8.

Usage:
    python detect.py --source test.jpg            # image
    python detect.py --source traffic.mp4         # video
    python detect.py --source 0                   # webcam (press q to quit)
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Class names that count as a violation (person WITHOUT helmet).
# If your dataset uses different names, pass --violation-classes "name1" "name2"
DEFAULT_KEYS = ("without", "no helmet", "no_helmet", "no-helmet", "nohelmet", "head")


def is_violation(name, custom=None):
    n = name.lower()
    if custom:
        return n in [c.lower() for c in custom]
    return any(k in n for k in DEFAULT_KEYS)


def annotate(frame, result, custom=None):
    """Draw boxes on frame. Returns (frame, safe_count, violation_count)."""
    safe = bad_count = 0
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        name = result.names[int(box.cls[0])]
        conf = float(box.conf[0])
        bad = is_violation(name, custom)
        color = (0, 0, 255) if bad else (0, 200, 0)  # BGR: red / green
        bad_count += bad
        safe += not bad
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(y1 - h - 8, 0)), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(y1 - 4, h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    summary = f"Safe: {safe}  Violations: {bad_count}"
    cv2.putText(frame, summary, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, summary, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame, safe, bad_count


def main():
    ap = argparse.ArgumentParser(description="SafetyVision helmet detection")
    ap.add_argument("--source", required=True, help="image/video path, or 0 for webcam")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--show", action="store_true", help="show live window")
    ap.add_argument("--violation-classes", nargs="*", default=None)
    a = ap.parse_args()

    model = YOLO(a.weights)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- webcam / video ----
    is_cam = a.source.isdigit()
    is_image = (not is_cam) and Path(a.source).suffix.lower() in IMAGE_EXT

    if is_image:
        frame = cv2.imread(a.source)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {a.source}")
        res = model.predict(frame, conf=a.conf, verbose=False)[0]
        frame, s, v = annotate(frame, res, a.violation_classes)
        path = out_dir / f"{Path(a.source).stem}_detected.jpg"
        cv2.imwrite(str(path), frame)
        print(f"Saved: {path}  | Safe: {s}  Violations: {v}")
        if a.show:
            cv2.imshow("SafetyVision", frame)
            cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(int(a.source) if is_cam else a.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    name = "webcam" if is_cam else Path(a.source).stem
    path = out_dir / f"{name}_detected.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    total_v = frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=a.conf, verbose=False)[0]
        frame, s, v = annotate(frame, res, a.violation_classes)
        total_v += v
        frames += 1
        writer.write(frame)
        if a.show or is_cam:
            cv2.imshow("SafetyVision (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {path}  | Frames: {frames}  Total violation detections: {total_v}")


if __name__ == "__main__":
    main()
