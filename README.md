# YOLOv8 Model Versioning and Drift Detection

A complete MLOps pipeline for YOLOv8 model versioning, training, and data drift detection using synthetic datasets.

## Key Features

- **Synthetic Dataset Generation**: Creates controlled datasets with varying noise levels and colors to simulate data drift
- **YOLOv8 Model Training**: Automated training pipeline with MLflow experiment tracking
- **Data Drift Detection**: Uses Evidently library to detect statistical drift between dataset versions
- **Model Performance Comparison**: Tracks and compares model metrics across different dataset versions
- **MLflow Integration**: Complete experiment tracking with parameters, metrics, and model artifacts
- **Visualization**: Sample dataset visualization with bounding boxes

## Installation

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup with uv

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd witsense-detector-versioning-tracking
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

## Usage

### Run the Complete Pipeline

```bash
uv run python model_versioning_drift_calc.py
```

This will:
1. Create project directory structure
2. Generate 3 synthetic datasets (v1, v2, v3) with different characteristics
3. Train YOLOv8 models on each dataset
4. Analyze drift between dataset pairs
5. Generate comparison reports

### Using Real Datasets

For real YOLO datasets, use the helper script:

```bash
uv run python real_datasets.py
```

Choose from:
- COCO8 (recommended for testing)
- Pascal VOC
- Custom datasets

## Project Structure

```
witsense-detector-versioning-tracking/
├── model_versioning_drift_calc.py    # Main pipeline script
├── real_datasets.py                  # Real dataset utilities
├── pyproject.toml                    # Project configuration
├── requirements.txt                  # Dependencies
└── yolov8_drift_project/             # Generated during execution
    ├── dataset/                      # Synthetic datasets
    ├── models/                       # Trained models
    ├── reports/                      # Drift analysis reports
    └── configs/                      # Configuration files
```

## Key Components

### 1. DatasetGenerator
- Creates synthetic images with colored rectangles
- Simulates different camera conditions with varying noise
- Generates YOLO format annotations
- Creates dataset configuration files

### 2. YOLOv8Trainer
- Trains YOLOv8 models with MLflow tracking
- Logs parameters, metrics, and artifacts
- Handles training errors gracefully
- Supports different model sizes (n, s, m, l, x)

### 3. DriftAnalyzer
- Extracts image features (brightness, contrast, color channels)
- Uses Evidently library for statistical drift detection
- Generates JSON reports with drift metrics
- Compares model performance across versions

## MLflow Tracking

View experiment results:
```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

Access at: http://localhost:5000

## Configuration

### Model Training Parameters
```python
model_configs = {
    'epochs': 10,
    'imgsz': 640,
    'batch': 8,
    'device': 'cpu',  # Change to 'cuda' for GPU
    'patience': 5,
    'save': True,
    'cache': False
}
```

### Dataset Parameters
- Training images: 80 per version
- Validation images: 20 per version
- Image size: 640x640
- Classes: 3 (red_object, green_object, blue_object)

## Output Files

- **Models**: `yolov8_drift_project/models/yolov8_v*.pt`
- **Drift Reports**: `yolov8_drift_project/reports/drift_report_*.json`
- **Model Comparison**: `yolov8_drift_project/reports/model_comparison.csv`
- **Visualizations**: `yolov8_drift_project/reports/dataset_samples_*.png`

## Requirements

Core dependencies:
- ultralytics (YOLOv8)
- torch, torchvision (PyTorch)
- opencv-python (Computer Vision)
- mlflow (Experiment Tracking)
- evidently (Drift Detection)
- pandas, numpy (Data Processing)
- matplotlib, seaborn (Visualization)

## CUDA Support

For GPU acceleration, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then set `device: 'cuda'` in model configuration.

## License

MIT License