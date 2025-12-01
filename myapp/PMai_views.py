
import threading
import json
import logging
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

from .forms import DogForm, DogImageFormSet, OrgAdminDogForm, VACCINE_CHOICES, NotificationForm
from .training.forms import TrainingForm
from django.shortcuts import render, redirect, get_object_or_404
from django.http import Http404
from django.contrib.auth import authenticate, login as auth_login
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth.decorators import login_required
from .models import Dog, DogImage, User, Notification, AdoptionParent, TrainingSession
from django.db.models import Q
from django.db import models

# Import training script
from scraips.train_resnet18 import train_dog_model

logger = logging.getLogger(__name__)


def admin_page(request):
    return render(request, 'admin/dashdoardAI/dashdoard.html')


def set_auto_training(request):
    return render(request, 'admin/Training/SetautoTraining.html')

def page_training(request):
    """View training page with all training sessions"""
    training_sessions = TrainingSession.objects.all()
    context = {
        'training_sessions': training_sessions
    }
    return render(request, 'admin/Training/TrainingAlerts.html', context)


@require_http_methods(["POST"])
@csrf_exempt
def start_training(request):
    """
    API endpoint to start training
    Receives: training_name, model, dataset_count
    Returns: JSON response with training_id and redirect option
    """
    try:
        data = json.loads(request.body)
        
        training_name = data.get('name')
        model_name = data.get('model')
        dataset_count = int(data.get('dataset_count', 1))
        epochs = int(data.get('epochs', 10))
        
        # Validate required fields
        if not training_name or not model_name:
            return JsonResponse({
                'status': 'error',
                'message': 'Training name and model are required'
            }, status=400)
        
        # Create training session record
        training_session = TrainingSession.objects.create(
            training_name=training_name,
            model_name=model_name,
            dataset_count=dataset_count,
            epochs=epochs,
            status='pending'
        )
        
        # Start training in background thread
        thread = threading.Thread(
            target=run_training_in_background,
            args=(training_session.id, model_name, epochs)
        )
        thread.daemon = True
        thread.start()
        
        return JsonResponse({
            'status': 'success',
            'message': f'Training "{training_name}" started',
            'training_id': training_session.id,
            'redirect_url': f'/training-details/{training_session.id}/'
        })
    
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON'
        }, status=400)
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }, status=500)


def run_training_in_background(training_session_id, model_name, epochs):
    """
    Background task to run training
    """
    try:
        training_session = TrainingSession.objects.get(id=training_session_id)
        training_session.status = 'training'
        training_session.started_at = timezone.now()
        training_session.save()
        
        # Prepare training data
        # In real scenario, fetch from DogImage and prepare dataset
        dog_images = DogImage.objects.all()[:100]  # Example: get first 100 images
        
        image_paths = [img.image.path for img in dog_images if img.image]
        labels = [0] * len(image_paths)  # Example labels
        
        training_data = {
            'image_paths': image_paths,
            'labels': labels,
            'epochs': epochs,
            'dataset_count': training_session.dataset_count
        }
        
        # Run training
        if model_name == 'resnet18':
            result = train_dog_model(training_data)
            
            if result['status'] == 'success':
                training_session.status = 'completed'
                training_session.accuracy = result['accuracy']
                training_session.model_path = result['model_path']
            else:
                training_session.status = 'failed'
                training_session.error_message = result.get('error', 'Unknown error')
        else:
            # For other models, mark as completed
            training_session.status = 'completed'
            training_session.accuracy = 85.5  # Placeholder
        
        training_session.completed_at = timezone.now()
        training_session.save()
        
        logger.info(f"Training {training_session_id} completed with status: {training_session.status}")
    
    except TrainingSession.DoesNotExist:
        logger.error(f"Training session {training_session_id} not found")
    except Exception as e:
        logger.error(f"Error in background training: {str(e)}")
        try:
            training_session = TrainingSession.objects.get(id=training_session_id)
            training_session.status = 'failed'
            training_session.error_message = str(e)
            training_session.completed_at = timezone.now()
            training_session.save()
        except:
            pass


@login_required
def training_details(request, training_id):
    """Show training details page with actual data from database"""
    training_session = get_object_or_404(TrainingSession, id=training_id)
    
    context = {
        'training': training_session,
        'training_id': training_session.id,
        'training_name': training_session.training_name,
        'model_name': training_session.model_name,
        'dataset_count': training_session.dataset_count,
        'epochs': training_session.epochs,
        'status': training_session.status,
    }
    
    return render(request, 'admin/Training/TrainingStarted.html', context)


@login_required
def training_status(request, training_id):
    """Get training session status"""
    training_session = get_object_or_404(TrainingSession, id=training_id)
    
    status_data = {
        'id': training_session.id,
        'training_name': training_session.training_name,
        'model_name': training_session.model_name,
        'dataset_count': training_session.dataset_count,
        'epochs': training_session.epochs,
        'status': training_session.status,
        'accuracy': training_session.accuracy,
        'loss': training_session.loss,
        'error_message': training_session.error_message,
        'model_path': training_session.model_path,
        'created_at': training_session.created_at.isoformat() if training_session.created_at else None,
        'started_at': training_session.started_at.isoformat() if training_session.started_at else None,
        'completed_at': training_session.completed_at.isoformat() if training_session.completed_at else None,
    }
    
    return JsonResponse(status_data)


@login_required
def training_list(request):
    """Get list of all training sessions"""
    training_sessions = TrainingSession.objects.all().values(
        'id', 'training_name', 'model_name', 'status', 'accuracy', 
        'created_at', 'completed_at'
    )
    
    return JsonResponse({
        'trainings': list(training_sessions)
    })