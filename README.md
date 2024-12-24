# Detecting, tracking vehicle in intersection
### Setting
```bash
git clone <https-url>
```
```bash
pip install -r requirements
```

### Path config
#### Model path
```models
...
├──models
|  ├── configs
|  |   ├──yolov5.yaml
|  |   └──yolov8.yaml
|  ├── yolov5.pt
|  ├── yolov8.pt
|  ├── ssd.pt
|  └── fpn-fasterrcnn.pt

```
#### Dataset path
```dataset
├──COCO-dataset # For FasterRCNN and SSD
|  ├── train
|  |    ├── ...
|  |    └── _annotations.coco.json
|  ├── valid
|  |    ├── ...
|  |    └──_annotations.coco.json
|  └── test
|       ├── ...
|       └── _annotations.coco.json  
```

```dataset
├──YOLO_Dataset # For YOLO
|  ├── train
|  |   ├── images
|  |   └── labels
|  ├── valid
|  |   ├── images
|  |   └── labels
|  ├── test
|  |   ├── images
|  |   └── labels
|  └── data.yaml
```

### Training
```bash
python --model_name <name> -- --epochs <num_epochs>
```