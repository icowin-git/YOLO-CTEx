python tools/yolo2coco.py --yolo_dir data/exp3_20251029/train --output data/exp3_20251029/annotations/train_emphysema.json
python tools/yolo2coco.py --yolo_dir data/exp3_20251029/val --output data/exp3_20251029/annotations/val_emphysema.json
python tools/yolo2coco.py --yolo_dir data/exp3_20251029/test --output data/exp3_20251029/annotations/test_emphysema.json

wait