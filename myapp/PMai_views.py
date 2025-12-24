from django.http import JsonResponse
from django.shortcuts import render, redirect ,get_object_or_404
from django.utils import timezone
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from myapp.models import Dog, DogImage,EvaluationResult
import logging
from scripts.train_model import train_resnet18, train_resnet50
from .models import TrainingSession


logger = logging.getLogger(__name__)

# def admin_page(request):
#     return render(request, 'admin/base.html')

def page_training(request):
    sessions = TrainingSession.objects.all()

    images_no_embedding = DogImage.objects.filter(
        embedding_binary__isnull=True
    )

    dataset_count = images_no_embedding.count()

    context = {
        "sessions": sessions,
        "dataset_count": dataset_count,
    }

    return render(request, "admin/Training/Training.html", context)


def set_auto_training(request):
    return render(request, 'admin/Training/SetautoTraining.html')


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
            img_files=DogImage.objects.filter(embedding_binary__isnull=True).count(),
            model_version="v1.0",
            
        )

        try:
            if model_type == "resnet18":
                train_resnet18(session)

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

from django.contrib import messages
def delete_training_session(request, session_id):
    session = get_object_or_404(TrainingSession, id=session_id)
    DogImage.objects.filter(
        training_session=session
    ).update(
        embedding_binary=None,
        training_session=None   # แนะนำให้ reset ด้วย
    )
    session.delete()
    messages.success(request, "Training session deleted successfully.")
    return redirect('page_training')

# def test_model_performance(request):
#     return render(request, 'admin/Training/test_model.html')

def test_model_performance(request, model_id):
    model_data = get_object_or_404(TrainingSession, id=model_id)
    all_images = model_data.images.all()
    all_images_count = all_images.count()
    print("IMAGE COUNT =", all_images_count)

    context = {
        "training_name": model_data.training_name,
        "model_name": model_data.model_name,
        "created_at": model_data.data_added.strftime("%Y-%m-%d %H:%M"),
        "model_id": model_data.id,
        "all_images_count": all_images_count,
    }

    return render(request, "admin/Training/test_model.html", context)

# from django.shortcuts import get_object_or_404
# from django.http import JsonResponse
from django.conf import settings
# from django.core.files import File

# from .models import TrainingSession, DogImage, EvaluationResult
from scripts.Test_model.utils import load_embeddings_from_images
from scripts.Test_model.Test_KNN import run_knn
# from scripts.Test_model.Test_TSNE import run_tsne

from sklearn.model_selection import train_test_split
from django.http import HttpResponse

# import osponse

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")

from django.shortcuts import get_object_or_404
from django.http import HttpResponse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from scripts.Test_model.Test_KNN import fit_model_knn, save_plot_to_evaluation

def knn_test(request, model_id):
    session = get_object_or_404(TrainingSession, id=model_id)
    all_images = session.images.all()
    print(f"DEBUG: Found {all_images.count()} images total in session {model_id}")
    EMBED_DIM = 512
    all_features = []
    all_labels = []


    for img in all_images:
        if not img.embedding_binary:
            continue

        try:
            emb = np.frombuffer(
                img.embedding_binary,
                dtype=np.float32
            )

            if emb.size != EMBED_DIM:
                print(f"Image {img.id}: invalid dim {emb.size}")
                continue

            all_features.append(emb)
            #all_labels.append(img.id)
            all_labels.append(img.dog.id)

        except Exception as e:
            print(f"Error decoding embedding for image {img.id}: {e}")
    # 3. เตรียมข้อมูล KNN (Scikit-learn)
    if not all_features:
        # ตรวจสอบว่าตกลงมีรูปไหม หรือมีแต่ไม่มีข้อมูล vector
        if all_images.count() == 0:
            return HttpResponse(f"Error: No images linked to session {model_id}")
        else:
            return HttpResponse(f"Error: Found {all_images.count()} images, but all have NULL embedding_binary")

    if all_features:
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"Data ready: {len(X)} samples")

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y,
            test_size=0.5,       
            random_state=42,
            stratify=y    
        )
        y_pred = fit_model_knn(X_train, y_train, X_test)
        # print("Train size:", X_train.shape)
        # print("Test size:", X_test.shape)

        # accuracy
        accuracy = (y_pred == y_test).mean()

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # ====== CREATE FIGURE ======
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix (KNN)")

        # ====== CALL HELPER ======
        evaluation = save_plot_to_evaluation(
            training_session=session,
            fig=fig,
            eval_type="knn",
            score=accuracy
        )

                ##return HttpResponse(f"Loaded {len(X)} images with labels. acc{accuracy:.4f}")
    else:
        return HttpResponse("No images with embeddings found in this session.")


def show_img_test_knn(request):
        results = EvaluationResult.objects.filter(
            training_session=session,
            eval_type='knn'
        ).exclude(image='')

        context = {
            "results": results,
        }

        return render(request, "admin/Training/test_model.html", context)





def visualize_knn_results(image_ids, y_true, y_pred, num_samples=10):
    """
    ฟังก์ชันสำหรับดึงรูปจาก DB มาแสดงผลเปรียบเทียบ
    """
    # สุ่มเลือก index มาแสดง
    indices = np.random.choice(len(image_ids), min(num_samples, len(image_ids)), replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 7)) # ปรับ size ตามจำนวนรูป
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img_obj = TrainingImage.objects.get(id=image_ids[idx]) # ดึง Object รูปภาพ
        
        # โหลดรูปภาพ
        img = Image.open(img_obj.image_file.path)
        axes[i].imshow(img)
        
        # ตรวจสอบว่าทายถูกไหม
        is_correct = y_true[idx] == y_pred[idx]
        color = 'green' if is_correct else 'red'
        
        # แสดง Label (ควรดึงชื่อสุนัขมาแสดงแทน ID เพื่อความเข้าใจง่าย)
        # true_name = Dog.objects.get(id=y_true[idx]).name
        # pred_name = Dog.objects.get(id=y_pred[idx]).name
        
        axes[i].set_title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}", color=color, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    return fig
    
def TSNE_test(request,model_id):
    print("ENTER TSNE_test")


