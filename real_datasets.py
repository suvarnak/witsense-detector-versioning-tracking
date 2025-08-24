# Real YOLO Datasets for Model Versioning and Drift Detection

from pathlib import Path
import yaml
from ultralytics import YOLO

def download_coco_dataset():
    """Download COCO dataset using ultralytics"""
    # COCO8 - small subset for testing (8 images per class)
    config = {
        'path': 'coco8',
        'train': 'images/train',
        'val': 'images/val', 
        'nc': 8,
        'names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']  # truncated
    }
    
    # Save config
    with open('coco8.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Download will happen automatically when training
    print("COCO8 dataset config ready!")
    return 'coco8.yaml'

def download_voc_dataset():
    """Download Pascal VOC dataset"""
    config = {
        'path': 'VOC',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 20,
        'names': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    }
    
    with open('VOC.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("VOC dataset config ready!")
    return 'VOC.yaml'

def get_dataset_choice():
    """Get user choice for dataset"""
    print("\nAvailable datasets:")
    print("1. COCO8 (small COCO subset - recommended)")
    print("2. Pascal VOC")
    print("3. Synthetic (original toy dataset)")
    
    try:
        choice = input("Choose dataset (1-3): ").strip()
        
        if choice == '1':
            return download_coco_dataset()
        elif choice == '2':
            return download_voc_dataset()
        else:
            print("Using synthetic datasets...")
            return None  # Use synthetic
    except KeyboardInterrupt:
        print("\nUsing synthetic datasets...")
        return None
    except Exception as e:
        print(f"Error: {e}. Using synthetic datasets...")
        return None

