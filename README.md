# SafetyVision: Helmet Detection using YOLOv8

Real-time helmet detection system that identifies whether two-wheeler riders are wearing helmets, built with YOLOv8 and OpenCV. Works on images, videos and live webcam, and counts safety violations per frame.

![demo](results/demo.jpg)

## Features
- Custom-trained YOLOv8 model (transfer learning from COCO weights)
- Detection on image, video file or webcam
- Green box for helmet, red box for violation, with live violation counter
- Evaluated with mAP, precision, recall and confusion matrix

## Tech Stack
Python, YOLOv8 (Ultralytics), PyTorch, OpenCV, NumPy, Matplotlib, Google Colab (GPU training), Roboflow (dataset)

## Dataset
- Source: <DATASET NAME + ROBOFLOW LINK>
- Classes: <e.g. With Helmet, Without Helmet>
- Split: <train / valid / test image counts>
- License: <dataset license>

## Results
| Metric | Value |
|---|---|
| mAP@50 | <fill> |
| mAP@50-95 | <fill> |
| Precision | <fill> |
| Recall | <fill> |

Training curves and confusion matrix are in the `results/` folder.

## Project Structure
```
safetyvision-helmet-detection/
├── notebooks/train_yolov8.ipynb   # Colab training notebook
├── weights/best.pt                # trained model
├── results/                       # graphs and sample outputs
├── detect.py                      # inference script
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone https://github.com/XCIXCMXIX/safetyvision-helmet-detection
cd safetyvision-helmet-detection
pip install -r requirements.txt
```

## Usage
```bash
python detect.py --source image.jpg        # image
python detect.py --source video.mp4        # video
python detect.py --source 0                # webcam (press q to quit)
```
Optional: `--conf 0.5` (confidence), `--weights weights/best.pt`, `--out results`.
Output is saved in the `results/` folder.

## Training
Open `notebooks/train_yolov8.ipynb` in Google Colab with a T4 GPU, add your Roboflow API key, and run all cells.

## Limitations and Future Work
- Accuracy drops in low light, heavy occlusion and very small/distant riders
- Next: number plate recognition for violators, tracking across video frames, deployment as a web app

## Author
Himanshu: [LinkedIn](https://linkedin.com/in/h-i-m-a-n-s-h-u-37b137437) | [GitHub](https://github.com/XCIXCMXIX)
