# Training System Implementation Guide

## Overview
This is a complete training system for the FindMyDog AdminAI application with deep learning model training capability.

## Files Created/Updated

### 1. **training/forms.py** ✅
- `TrainingForm`: Django form for training configuration
- Fields:
  - `training_name`: Name for the training session
  - `model_name`: Select model type (Key-value, Classification, Extraction, OCR, ResNet18)
  - `dataset_count`: Number of datasets to use
  - `epochs`: Number of training epochs

### 2. **scraips/train_resnet18.py** ✅
Deep learning training module with:
- `DogImageDataset`: Custom PyTorch dataset for dog images
- `ResNet18Trainer`: Main trainer class with:
  - Model loading and fine-tuning
  - Train/validation loops
  - Model saving and loading
  - Single image prediction capability
- `train_dog_model()`: Main function to run training

**Dependencies:**
```bash
pip install torch torchvision pillow numpy
```

### 3. **models.py** ✅
Added `TrainingSession` model to track training:
- `training_name`: Name of the training
- `model_name`: Type of model used
- `dataset_count`: Number of datasets
- `epochs`: Number of epochs
- `status`: Current status (pending, training, completed, failed)
- `accuracy`: Final accuracy
- `loss`: Final loss value
- `created_at`, `started_at`, `completed_at`: Timestamps
- `error_message`: Error details if failed
- `model_path`: Path to saved model

### 4. **PMai_views.py** ✅
Training API endpoints:
- `page_training()`: Main training dashboard
- `start_training()`: POST endpoint to start training
  - Accepts: name, model, dataset_count, epochs
  - Returns: JSON response with training_id
- `run_training_in_background()`: Background thread task
- `training_status()`: Get training session status
- `training_list()`: Get all training sessions

### 5. **urls.py** ✅
New URL routes:
```
/train-model/              → start_training (POST)
/training_status/<id>/     → training_status (GET)
/training_list/            → training_list (GET)
/page_training/            → page_training (GET)
```

### 6. **templates/admin/Training/TrainingAlerts.html** ✅
Updated with:
- Popup form for creating new training
- Fields: Training Name, Model Selection, Dataset Count, Epochs
- Validation and error handling
- Real-time status updates

### 7. **templates/admin/Training/TrainingStarted.html** ✅
New page showing:
- Success message
- Training details card
- Real-time progress bar
- Training logs section
- Results section (after completion)
- Auto-refresh status every 5 seconds

## Setup Instructions

### 1. Install Dependencies
```bash
pip install torch torchvision pillow
```

### 2. Apply Migrations
```bash
python manage.py migrate
```

### 3. Start Training
1. Go to Training Dashboard (`/page_training/`)
2. Click "Train new model" button
3. Fill in the form:
   - Training Name (e.g., "Dog Detection v1")
   - Select Model (e.g., "ResNet18 (Dog Detection)")
   - Dataset Count (default: 1)
   - Epochs (default: 10)
4. Click "Start Training"
5. See the training progress on the "Training Started" page

## API Endpoints

### Start Training
```
POST /train-model/
Content-Type: application/json

{
    "name": "Training Name",
    "model": "resnet18",
    "dataset_count": 1,
    "epochs": 10
}

Response:
{
    "status": "success",
    "message": "Training started",
    "training_id": 1
}
```

### Get Training Status
```
GET /training_status/<training_id>/

Response:
{
    "id": 1,
    "training_name": "Dog Detection v1",
    "status": "training",
    "accuracy": null,
    "created_at": "2025-12-01T10:00:00",
    "started_at": "2025-12-01T10:00:05",
    "completed_at": null
}
```

### List All Trainings
```
GET /training_list/

Response:
{
    "trainings": [
        {
            "id": 1,
            "training_name": "Dog Detection v1",
            "model_name": "resnet18",
            "status": "completed",
            "accuracy": 92.5,
            "created_at": "2025-12-01T10:00:00",
            "completed_at": "2025-12-01T10:30:00"
        }
    ]
}
```

## Architecture

```
User Interface (TrainingAlerts.html)
           ↓
    Popup Form + Frontend JS
           ↓
    /train-model/ API (POST)
           ↓
    start_training() View
           ↓
    Background Thread
           ↓
    run_training_in_background()
           ↓
    train_dog_model() (train_resnet18.py)
           ↓
    ResNet18Trainer
           ↓
    Save Model + Update DB (TrainingSession)
           ↓
    TrainingStarted.html shows progress
           ↓
    Polling /training_status/ API
```

## Data Flow

1. **User Input**: Training form in popup
2. **API Call**: POST to /train-model/ with training parameters
3. **DB Record**: Create TrainingSession with pending status
4. **Background Job**: Spawn thread to run training
5. **Training**: ResNet18 trains on dog images
6. **Results**: Save model, update accuracy, mark complete
7. **Frontend**: Poll /training_status/ to get updates

## Training Process

1. Load pretrained ResNet18
2. Modify final layer for classification
3. Split data: 80% train, 20% validation
4. Apply augmentations (rotation, flip)
5. Train for N epochs
6. Calculate accuracy and loss
7. Save best model based on validation accuracy
8. Update database with results

## Future Enhancements

- [ ] Add batch training support
- [ ] Implement model comparison
- [ ] Add training scheduler (automated training at specific times)
- [ ] Real-time WebSocket updates instead of polling
- [ ] Model evaluation metrics (precision, recall, F1)
- [ ] Training history graphs
- [ ] Model deployment endpoint
- [ ] Multi-GPU support
- [ ] Different model architectures (ResNet50, EfficientNet, etc.)

## Troubleshooting

### Error: "CUDA out of memory"
Solution: Reduce batch size or use CPU training
```python
device = 'cpu'  # Force CPU in train_resnet18.py
```

### Error: "No images found"
Solution: Ensure dog images exist in database
```bash
python manage.py shell
from myapp.models import DogImage
print(DogImage.objects.count())
```

### Training not starting
Solution: Check Django logs
```bash
python manage.py runserver --verbosity 3
```

## Performance Tips

1. **GPU Training**: Install CUDA support for faster training
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Batch Size**: Adjust in train_resnet18.py (line 115, 135)
   ```python
   train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduce if OOM
   ```

3. **Epochs**: Start with fewer epochs (5-10) for testing
   ```python
   epochs=5  # Start low
   ```

## Notes

- All training happens in background threads
- Frontend polls every 5 seconds for updates
- Models saved in `models/` directory by default
- Training sessions persist in database
- Failed trainings show error messages

---
Last Updated: December 1, 2025
