"""
Deep Learning Training Module for Dog Detection
Using ResNet18 architecture
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DogImageDataset(Dataset):
    """Custom Dataset for dog images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            raise


class ResNet18Trainer:
    """Training class for ResNet18 model"""
    
    def __init__(self, num_classes=2, learning_rate=0.001, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        
        # Modify final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def get_transforms(self):
        """Return train and val transforms"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return train_transform, val_transform
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(val_loader)
        epoch_accuracy = 100 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(self, train_loader, val_loader, epochs=10, save_path='model.pth'):
        """Complete training loop"""
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        best_accuracy = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            logger.info(
                f'Epoch [{epoch+1}/{epochs}] '
                f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | '
                f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%'
            )
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                self.save_model(save_path)
                logger.info(f'Best model saved at epoch {epoch+1}')
        
        logger.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
        return self.history, best_accuracy
    
    def save_model(self, path):
        """Save model to disk"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
    
    def predict(self, image_path):
        """Predict on a single image"""
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence


def train_dog_model(training_data, save_dir='models'):
    """
    Main function to train the model
    
    Args:
        training_data: dict with 'image_paths', 'labels', 'epochs', 'dataset_count'
        save_dir: directory to save the model
    
    Returns:
        dict with training results
    """
    try:
        # Create save directory if not exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Get data
        image_paths = training_data.get('image_paths', [])
        labels = training_data.get('labels', [])
        epochs = training_data.get('epochs', 10)
        
        if not image_paths or not labels:
            raise ValueError("No image paths or labels provided")
        
        # Create datasets
        train_transform, val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ]), transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Split data into train/val
        split_idx = int(0.8 * len(image_paths))
        train_paths = image_paths[:split_idx]
        train_labels = labels[:split_idx]
        val_paths = image_paths[split_idx:]
        val_labels = labels[split_idx:]
        
        train_dataset = DogImageDataset(train_paths, train_labels, train_transform)
        val_dataset = DogImageDataset(val_paths, val_labels, val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Create trainer
        trainer = ResNet18Trainer(num_classes=2, device=device)
        
        # Train model
        model_path = os.path.join(save_dir, f'resnet18_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        history, best_accuracy = trainer.train(
            train_loader,
            val_loader,
            epochs=epochs,
            save_path=model_path
        )
        
        return {
            'status': 'success',
            'accuracy': best_accuracy,
            'model_path': model_path,
            'history': history,
            'message': f'Training completed with {best_accuracy:.2f}% accuracy'
        }
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'message': f'Training failed: {str(e)}'
        }


if __name__ == '__main__':
    # Example usage
    logger.info("ResNet18 Training Module loaded successfully")
