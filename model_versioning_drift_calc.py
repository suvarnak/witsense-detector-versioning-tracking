# YOLOv8 Model Versioning and Drift Detection Project
# Complete MLOps pipeline with model tracking and drift detection

import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# MLOps and Monitoring imports
import mlflow
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift

# YOLOv8 imports
from ultralytics import YOLO
import yaml

# Set up directories
PROJECT_ROOT = Path("yolov8_drift_project")
DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

class DatasetGenerator:
    """Generate synthetic datasets with controlled variations to simulate drift"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.image_size = (640, 640)
        self.colors = {
            'v1': [(255, 0, 0), (0, 255, 0), (0, 0, 255)],  # RGB colors
            'v2': [(255, 128, 0), (128, 255, 0), (0, 128, 255)],  # Shifted colors
            'v3': [(128, 0, 128), (255, 255, 0), (0, 255, 255)]   # Different colors
        }
        
    def create_synthetic_image(self, version: str, idx: int, split: str) -> Tuple[np.ndarray, List[Dict]]:
        """Create a synthetic image with bounding boxes"""
        img = np.ones((*self.image_size, 3), dtype=np.uint8) * 50  # Dark background
        
        # Add noise based on version (simulating different camera conditions)
        noise_levels = {'v1': 10, 'v2': 25, 'v3': 40}
        noise = np.random.normal(0, noise_levels[version], img.shape).astype(np.uint8)
        img = np.clip(img + noise, 0, 255)
        
        annotations = []
        colors = self.colors[version]
        
        # Generate 1-3 objects per image
        num_objects = np.random.randint(1, 4)
        
        for i in range(num_objects):
            # Random position and size
            x_center = np.random.randint(100, self.image_size[1] - 100)
            y_center = np.random.randint(100, self.image_size[0] - 100)
            width = np.random.randint(50, 150)
            height = np.random.randint(50, 150)
            
            # Draw rectangle
            color = colors[i % len(colors)]
            cv2.rectangle(img, 
                         (x_center - width//2, y_center - height//2),
                         (x_center + width//2, y_center + height//2),
                         color, -1)
            
            # YOLO format: class_id x_center y_center width height (normalized)
            class_id = i % 3  # 3 classes: 0, 1, 2
            x_norm = x_center / self.image_size[1]
            y_norm = y_center / self.image_size[0]
            w_norm = width / self.image_size[1]
            h_norm = height / self.image_size[0]
            
            annotations.append({
                'class_id': class_id,
                'x_center': x_norm,
                'y_center': y_norm,
                'width': w_norm,
                'height': h_norm
            })
        
        return img, annotations
    
    def generate_dataset(self, version: str, num_train: int = 100, num_val: int = 20):
        """Generate a complete dataset with train/val splits"""
        version_path = self.base_path / version
        
        # Create directory structure
        for split in ['train', 'val']:
            (version_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (version_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Generate images and labels
        splits_config = {'train': num_train, 'val': num_val}
        
        for split, count in splits_config.items():
            for i in range(count):
                # Generate image and annotations
                img, annotations = self.create_synthetic_image(version, i, split)
                
                # Save image
                img_path = version_path / 'images' / split / f'{i:04d}.jpg'
                cv2.imwrite(str(img_path), img)
                
                # Save labels in YOLO format
                label_path = version_path / 'labels' / split / f'{i:04d}.txt'
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann['class_id']} {ann['x_center']:.6f} "
                               f"{ann['y_center']:.6f} {ann['width']:.6f} "
                               f"{ann['height']:.6f}\n")
        
        # Create dataset config file
        dataset_config = {
            'path': str(version_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 3,
            'names': ['red_object', 'green_object', 'blue_object']
        }
        
        config_path = version_path / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f)
        
        print(f"Generated dataset {version} with {num_train} train and {num_val} val images")
        return config_path

class YOLOv8Trainer:
    """Train and manage YOLOv8 models with MLflow tracking"""
    
    def __init__(self, experiment_name: str = "yolov8_drift_experiment"):
        # Set up MLflow with proper tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Creating new experiment: {experiment_name}")
            mlflow.create_experiment(experiment_name)
        self.model_configs = {
            'epochs': 10,
            'imgsz': 640,
            'batch': 8,
            'device': 'cpu',  # Change to 'cuda' if GPU available
            'patience': 5,
            'save': True,
            'cache': False
        }
    
    def train_model(self, dataset_config_path: Path, version: str, model_size: str = 'n') -> Dict:
        """Train YOLOv8 model and log with MLflow"""
        
        try:
            with mlflow.start_run(run_name=f"yolov8_{version}") as run:
                run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_params(self.model_configs)
            mlflow.log_param("dataset_version", version)
            mlflow.log_param("model_size", model_size)
            
            # Initialize model
            model = YOLO(f'yolov8{model_size}.pt')
            
            # Train model
            print(f"Training YOLOv8 on dataset {version}...")
            results = model.train(
                data=str(dataset_config_path),
                **self.model_configs
            )
            
            # Get training results
            metrics = results.results_dict if hasattr(results, 'results_dict') else {}
            
            # Log metrics
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
            
            # Save and log model
            model_path = MODELS_DIR / f"yolov8_{version}.pt"
            model.save(str(model_path))
            mlflow.log_artifact(str(model_path), "model")
            
            # Log dataset config
            mlflow.log_artifact(str(dataset_config_path), "dataset_config")
            
            # Get model info for return
            model_info = {
                'run_id': run_id,
                'version': version,
                'model_path': model_path,
                'metrics': metrics,
                'config': self.model_configs
            }
            
            print(f"Model {version} training completed. Run ID: {run_id}")
            return model_info
        except Exception as e:
            print(f"MLflow error: {e}")
            # Return basic model info without MLflow tracking
            model_path = MODELS_DIR / f"yolov8_{version}.pt"
            return {
                'run_id': 'local_run',
                'version': version,
                'model_path': model_path,
                'metrics': {},
                'config': self.model_configs
            }

class DriftAnalyzer:
    """Analyze drift between dataset versions and model performance"""
    
    def __init__(self):
        self.feature_extractors = {}
        
    def extract_image_features(self, image_dir: Path, max_images: int = 50) -> pd.DataFrame:
        """Extract basic image features for drift analysis"""
        features = []
        
        image_files = list(image_dir.glob("*.jpg"))[:max_images]
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Basic image statistics
            feature_dict = {
                'filename': img_path.name,
                'mean_brightness': np.mean(img),
                'std_brightness': np.std(img),
                'mean_r': np.mean(img[:, :, 2]),
                'mean_g': np.mean(img[:, :, 1]),
                'mean_b': np.mean(img[:, :, 0]),
                'contrast': np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                'sharpness': cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def analyze_dataset_drift(self, base_version: str, compare_version: str) -> Report:
        """Analyze drift between two dataset versions"""
        
        base_dir = DATA_DIR / base_version / "images" / "train"
        compare_dir = DATA_DIR / compare_version / "images" / "train"
        
        # Extract features
        base_features = self.extract_image_features(base_dir)
        compare_features = self.extract_image_features(compare_dir)
        
        # Prepare data for Evidently
        base_features['dataset'] = base_version
        compare_features['dataset'] = compare_version
        
        # Create drift report
        drift_report = Report(metrics=[
            DriftedColumnsCount(),
            ValueDrift(column='mean_brightness'),
            ValueDrift(column='contrast'),
        ])
        
        # Run analysis
        drift_report.run(reference_data=base_features.drop('dataset', axis=1),
                        current_data=compare_features.drop('dataset', axis=1))
        
        # Save report as JSON
        report_data = {
            'metadata': drift_report.__dict__.get('metadata', {}),
            'timestamp': drift_report.__dict__.get('_timestamp', ''),
            'metrics': []
        }
        
        for metric in drift_report.metrics:
            metric_data = metric.__dict__.copy()
            report_data['metrics'].append(metric_data)
        
        report_path = REPORTS_DIR / f"drift_report_{base_version}_vs_{compare_version}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"Drift analysis completed: {base_version} vs {compare_version}")
        print(f"Report saved to: {report_path}")
        
        return drift_report
    
    def compare_model_performance(self, model_infos: List[Dict]) -> pd.DataFrame:
        """Compare performance across model versions"""
        comparison_data = []
        
        for model_info in model_infos:
            metrics = model_info.get('metrics', {})
            comparison_data.append({
                'version': model_info['version'],
                'run_id': model_info['run_id'],
                **metrics
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_path = REPORTS_DIR / "model_comparison.csv"
        df.to_csv(comparison_path, index=False)
        
        return df

def setup_project_structure():
    """Create project directory structure"""
    directories = [
        PROJECT_ROOT,
        DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("Project structure created successfully!")

def visualize_dataset_samples(version: str, num_samples: int = 4):
    """Visualize sample images from a dataset version"""
    images_dir = DATA_DIR / version / "images" / "train"
    labels_dir = DATA_DIR / version / "labels" / "train"
    
    image_files = list(images_dir.glob("*.jpg"))[:num_samples]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, img_path in enumerate(image_files):
        if i >= num_samples:
            break
            
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Read labels
        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        # Convert to pixel coordinates
                        img_h, img_w = img.shape[:2]
                        x1 = int((x_center - width/2) * img_w)
                        y1 = int((y_center - height/2) * img_h)
                        x2 = int((x_center + width/2) * img_w)
                        y2 = int((y_center + height/2) * img_h)
                        
                        boxes.append((x1, y1, x2, y2, int(class_id)))
        
        # Draw bounding boxes
        for x1, y1, x2, y2, class_id in boxes:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            color = colors[class_id % 3]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        axes[i].imshow(img)
        axes[i].set_title(f"{version} - {img_path.name}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"dataset_samples_{version}.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main pipeline execution"""
    print("=== YOLOv8 Model Versioning and Drift Detection Project ===\n")
    
    # Step 1: Setup project structure
    print("1. Setting up project structure...")
    setup_project_structure()
    
    # Step 2: Choose dataset type
    from real_datasets import get_dataset_choice
    
    print("\n2. Choose dataset type:")
    real_dataset_config = get_dataset_choice()
    
    if real_dataset_config:
        # Use real dataset
        print(f"Using real dataset: {real_dataset_config}")
        dataset_configs = {'real': real_dataset_config}
        versions = ['real']
    else:
        # Use synthetic datasets
        print("\n2. Generating synthetic datasets...")
        dataset_generator = DatasetGenerator(DATA_DIR)
        
        dataset_configs = {}
        versions = ['v1', 'v2', 'v3']
        for version in versions:
            config_path = dataset_generator.generate_dataset(version, num_train=80, num_val=20)
            dataset_configs[version] = config_path
            
            # Visualize samples
            print(f"Visualizing samples from {version}...")
            visualize_dataset_samples(version)
    
    # Step 3: Train models
    print("\n3. Training YOLOv8 models...")
    trainer = YOLOv8Trainer()
    
    model_infos = []
    for version in versions:
        config_path = dataset_configs[version]
        model_info = trainer.train_model(config_path, version)
        model_infos.append(model_info)
    
    # Step 4: Analyze drift (only for synthetic datasets)
    if not real_dataset_config:
        print("\n4. Analyzing dataset drift...")
        drift_analyzer = DriftAnalyzer()
        
        # Compare v1 vs v2
        drift_analyzer.analyze_dataset_drift('v1', 'v2')
        
        # Compare v1 vs v3
        drift_analyzer.analyze_dataset_drift('v1', 'v3')
        
        # Compare v2 vs v3
        drift_analyzer.analyze_dataset_drift('v2', 'v3')
    else:
        print("\n4. Skipping drift analysis for real dataset (single version)")
    
    # Step 5: Compare model performance
    print("\n5. Comparing model performance...")
    if real_dataset_config or 'drift_analyzer' not in locals():
        drift_analyzer = DriftAnalyzer()
    
    comparison_df = drift_analyzer.compare_model_performance(model_infos)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Step 6: Generate summary report
    print("\n6. Generating summary report...")
    summary = {
        'project_info': {
            'total_datasets': len(dataset_configs),
            'total_models': len(model_infos),
            'mlflow_experiment': "yolov8_drift_experiment"
        },
        'datasets': {version: str(path) for version, path in dataset_configs.items()},
        'dataset_type': 'real' if real_dataset_config else 'synthetic',
        'models': [{
            'version': info['version'],
            'run_id': info['run_id'],
            'model_path': str(info['model_path'])
        } for info in model_infos],
        'reports_generated': [
            'drift_report_v1_vs_v2.json' if not real_dataset_config else 'No drift reports (single dataset)',
            'drift_report_v1_vs_v3.json' if not real_dataset_config else '',
            'drift_report_v2_vs_v3.json' if not real_dataset_config else '',
            'model_comparison.csv'
        ] if not real_dataset_config else ['model_comparison.csv']
    }
    
    summary_path = PROJECT_ROOT / "project_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nProject completed successfully!")
    print(f"Project directory: {PROJECT_ROOT.absolute()}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nTo view MLflow UI, run: mlflow ui --backend-store-uri ./mlruns")
    print(f"Drift reports saved in: {REPORTS_DIR}")

if __name__ == "__main__":
    main()