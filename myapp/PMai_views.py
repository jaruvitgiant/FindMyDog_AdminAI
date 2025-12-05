from django.http import JsonResponse
from django.shortcuts import render, redirect ,get_object_or_404
from django.utils import timezone
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from myapp.models import Dog, DogImage
import logging
from scripts.train_model import train_resnet18, train_resnet50
from .models import TrainingSession

logger = logging.getLogger(__name__)

# def admin_page(request):
#     return render(request, 'admin/base.html')

def page_training(request):
    # ดึงข้อมูล TrainingSession ทั้งหมด (เรียงล่าสุดก่อน ตาม ordering ใน model)
    sessions = TrainingSession.objects.all()
    img_nun_em = DogImage.objects.filter(embedding_binary__isnull=True).count()
    context = {
        "sessions": sessions,
        "img_nun_em": img_nun_em
    }

    return render(request, "admin/Training/Training.html", context)

def set_auto_training(request):
    pass


# def start_training(request):
#     if request.method == "POST":
#         model_type = request.POST.get("model_type")
#         training_name = request.POST.get("training_name")

#         # เลือกโมเดลตาม dropdown
#         if model_type == "resnet18":
#             train_resnet18()
#             return render(request, 'admin/Training/Training.html')

#         elif model_type == "resnet50":
#             train_resnet50()

#         else:
#             return JsonResponse({"error": "Invalid model type"})

#         return JsonResponse({"status": "Training started!"})

#     return JsonResponse({"error": "Invalid method"}, status=405)



def start_training(request):
    if request.method == "POST":
        training_name = request.POST.get("training_name")
        model_type = request.POST.get("model_type")

        session = TrainingSession.objects.create(
            training_name=training_name,
            model_name=model_type,
            status="training",
            data_added=timezone.now(), 
            img_files = DogImage.objects.all().count(),
            model_version="v1.0",
            
        )

        try:
            if model_type == "resnet18":
                train_resnet18()

            elif model_type == "resnet50":
                train_resnet50()

            session.status = "completed"
            session.completed_at = timezone.now()
            session.save()

        except Exception as e:
            session.status = "failed"
            session.error_message = str(e)
            session.completed_at = timezone.now()
            session.save()

        # return JsonResponse({"status": "Training started!"})
        return redirect('page_training')

    return JsonResponse({"error": "Invalid method"}, status=405)