import argparse
import time
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision.datasets import VOCDetection, CocoDetection
from torchvision.models.detection import (
    FasterRCNN, retinanet_resnet50_fpn, ssd300_vgg16,
    fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, RetinaNet
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.ssd import SSD300_VGG16_Weights, SSDClassificationHead
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
from functools import partial
import pandas as pd

def collate_fn(batch):
    images, targets = zip(*batch)
    formatted_targets = []
    
    for target in targets:
        valid_boxes = []
        valid_labels = []
        
        for obj in target:
            x, y, w, h = obj['bbox']
            if w > 0 and h > 0:  # Only include boxes with positive width and height
                valid_boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]

                valid_labels.append(obj['category_id'])
        
        # Convert boxes and labels into tensors
        boxes = torch.tensor(valid_boxes, dtype=torch.float32)
        labels = torch.tensor(valid_labels, dtype=torch.int64)

        # Check if boxes are empty and handle it
        if boxes.shape[0] == 0:
            # If there are no valid boxes, add a dummy entry
            boxes = torch.empty((0, 4), dtype=torch.float32)  # Shape [0, 4]
            labels = torch.empty((0,), dtype=torch.int64)     # Shape [0]
        
        formatted_targets.append({"boxes": boxes, "labels": labels})
    
    return list(images), formatted_targets


# Training function
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss=0
    for images, targets in tqdm(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
            
    return total_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation for faster evaluation
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Temporarily set model to training mode to get the loss
            model.train()
            loss_dict = model(images, targets)
            model.eval()  # Switch back to eval mode

            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            
    return total_loss / len(data_loader)

def evaluate_with_map(model, data_loader, device):
    model.eval()
    coco_predictions = []
    coco_ground_truths = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # Assign a pseudo-ID based on batch index and position
                pseudo_id = batch_idx * len(images) + i

                # Format ground truth boxes and labels
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()

                for box, label in zip(gt_boxes, gt_labels):
                    coco_ground_truths.append({
                            "image_id": pseudo_id,
                            "category_id": int(label),
                            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # COCO format
                            "iscrowd": 0
                        })

                    # Format predictions
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    coco_predictions.append({
                            "image_id": pseudo_id,
                            "category_id": int(label),
                            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # COCO format
                            "score": float(score)
                        })

        # Convert lists to COCO format dictionaries
    coco_gt = val_loader.dataset.coco
    coco_gt.dataset['annotations'] = coco_ground_truths
    coco_dt = coco_gt.loadRes(coco_predictions)

        # Evaluate using COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map_score_1 = coco_eval.stats[0]
    map_score_2 = coco_eval.stats[1]
    return [map_score_1,map_score_2]

def train_and_evaluate(models, train_loader, val_loader, num_epochs=10, lr=0.0001):
    results = {}
    mAP_results= {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        
        train_losses, val_losses, mAP_0_5_0_95, mAP_0_5 = [], [], [], []
        
        if model_name == 'SSD':
            num_epochs = num_epochs * 3
        
        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            train_loss = train_one_epoch(model, optimizer, train_loader, device)
            val_loss = evaluate(model, val_loader, device)
            map_scores = evaluate_with_map(model, val_loader, device)
            model.eval()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            mAP_0_5_0_95.append(map_scores[0])
            mAP_0_5.append(map_scores[1])
            print(f"Train Loss after epoch [{epoch}]: {train_loss}")
            print(f"Validation Loss after epoch [{epoch}]: {val_loss}")
            print(f"Epoch time: {time.time() - start_time:.2f} seconds")
        
        results[model_name] = {
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
        mAP_results[model_name] = {
            "mAP@0.5:0.95": mAP_0_5_0_95,
            "mAP@0.5": mAP_0_5,
        }

    return results, mAP_results

# Helper function for setting up models
def get_model(model_name, num_classes):
    if model_name == "FPN-FasterRCNN":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    elif model_name == "YOLOv5":
        model = YOLO("yolov5.yaml")

    elif model_name == "YOLOv8":
        from ultralytics import YOLO
        model = YOLO('yolov8.yaml')  
    
    elif model_name == "SSD":
        model = ssd300_vgg16(weights_backbone = 'DEFAULT', num_classes=num_classes)
        
    else:
        raise ValueError(f"Model {model_name} is not recognized.")
    
    return model

# Visualization
def plot_results(results):
    for model_name, metrics in results.items():
        plt.plot(metrics["train_loss"], label=f"{model_name} - Train Loss")
        plt.plot(metrics["val_loss"], label=f"{model_name} - Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss across Models")
    plt.show()
    
def plot_results_map(mAP_results):
    for model_name, metrics in mAP_results.items():
        plt.plot(metrics["mAP_0.5_0.95"], label=f"{model_name} - IoU=0.5:0.95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP scores")
    plt.legend()
    plt.title("mAP 0.5:0.95 across Models")
    plt.show()
    
def save_model(models, results, mAP_results, path):
    os.makedirs(path, exist_ok=True)
    
    # Save models
    for model_name, model in models.items():
        model_path = os.path.join(path, f"{model_name}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model '{model_name}' saved at {model_path}")

    # Save training and validation results
    results_dict = {
        "Model": [],
        "Epoch": [],
        "Train Loss": [],
        "Validation Loss": [],
    }
    for model_name, metrics in results.items():
        for epoch, (train_loss, val_loss) in enumerate(zip(metrics["train_loss"], metrics["val_loss"]), start=1):
            results_dict["Model"].append(model_name)
            results_dict["Epoch"].append(epoch)
            results_dict["Train Loss"].append(train_loss)
            results_dict["Validation Loss"].append(val_loss)
    
    results_df = pd.DataFrame(results_dict)
    results_csv_path = os.path.join(path, "results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Training and validation results saved at {results_csv_path}")

    # Save mAP scores
    mAP_dict = {
        "Model": [],
        "Epoch": [],
        "mAP@0.5:0.95": [],
        "mAP@0.5": [],
    }
    for model_name, scores in mAP_results.items():
        for epoch, (map_0_5_0_95, map_0_5) in enumerate(zip(scores["mAP@0.5:0.95"], scores["mAP@0.5"]), start=1):
            mAP_dict["Model"].append(model_name)
            mAP_dict["Epoch"].append(epoch)
            mAP_dict["mAP@0.5:0.95"].append(map_0_5_0_95)
            mAP_dict["mAP@0.5"].append(map_0_5)

    mAP_results_df = pd.DataFrame(mAP_dict)
    mAP_csv_path = os.path.join(path, "mAP_scores.csv")
    mAP_results_df.to_csv(mAP_csv_path, index=False)
    print(f"mAP scores saved at {mAP_csv_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate object detection models on COCO dataset")
    
    parser.add_argument('--model_name', type=str, nargs='+', 
                        required=True,
                        help="List of model names to train [FPN-FasterRCNN, SSD,YOLOv8,YOLOv5]")
    
    parser.add_argument('--num_classes', type=int, required=True,
                        help="Number of classes in the dataset (including background)")
    
    parser.add_argument('--save_path', type=str, default="models/{model_name}",
                        help="Directory path to save model and results")
    
    parser.add_argument('--coco_root', type=str, required=True,
                        help="Root directory of the COCO dataset (should contain 'train' and 'valid' folders)")
    
    return parser.parse_args()

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Parse command-line arguments
    args = parse_args()

    # Use parsed arguments to set variables
    model_name = args.model_name
    num_classes = args.num_classes
    save_path_template = args.save_path
    coco_root = args.coco_root

    coco_train = f"{coco_root}/train"
    coco_val = f"{coco_root}/valid"
    coco_train_ann = f"{coco_train}/_annotations.coco.json"
    coco_val_ann = f"{coco_val}/_annotations.coco.json"

    # Define transformation for COCO images
    transform = T.Compose([
        T.ToTensor(),
    ])

    # Load the COCO dataset for training and validation
    train_data = CocoDetection(root=coco_train, annFile=coco_train_ann, transform=transform)
    val_data = CocoDetection(root=coco_val, annFile=coco_val_ann, transform=transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

    models = {name: get_model(name, num_classes) for name in model_name}

    for model_name in model_name:
        save_path = save_path_template.format(model_name=model_name)

    results,mAP_results = train_and_evaluate(models.to(device), train_loader, val_loader, num_epochs=10)

    save_model(models,results, mAP_results, save_path)
