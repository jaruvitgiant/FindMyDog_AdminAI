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

    # Query evaluation result images for this training session
    knn_res = EvaluationResult.objects.filter(training_session=model_data, eval_type='knn').last()
    tsne_res = EvaluationResult.objects.filter(training_session=model_data, eval_type='tsne').last()
    
    # รวมเข้า list เพื่อส่งไป loop ใน template
    results = filter(None, [knn_res, tsne_res])
    
    context = {
        "training_name": model_data.training_name,
        "model_name": model_data.model_name,
        "created_at": model_data.data_added.strftime("%Y-%m-%d %H:%M"),
        "model_id": model_data.id,
        "all_images_count": all_images_count,
        "results": results,
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
from scripts.Test_model.Test_TSNE import run_tsne
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
        plt.close(fig)
        
        # ====== TSNE ======
        fig_tsne = run_tsne(X, y)
        save_plot_to_evaluation(session, fig_tsne, "tsne", score=accuracy)
        plt.close(fig_tsne)

        return redirect("test_model_performance", model_id=model_id)

        # return HttpResponse(f"Loaded {len(X)} images with labels. acc{accuracy:.4f}")
    else:
        return HttpResponse("No images with embeddings found in this session.")

# def tsne_test(request, model_id):
#     session = get_object_or_404(TrainingSession, id=model_id)
#     all_images = session.images.all()

#     EMBED_DIM = 512
#     all_features = []
#     all_labels = []

#     for img in all_images:
#         if not img.embedding_binary:
#             continue

#         try:
#             emb = np.frombuffer(
#                 img.embedding_binary,
#                 dtype=np.float32
#             )

#             if emb.size != EMBED_DIM:
#                 print(f"Image {img.id}: invalid dim {emb.size}")
#                 continue

#             all_features.append(emb)
#             all_labels.append(img.dog.id)

#         except Exception as e:
#             print(f"Error decoding embedding for image {img.id}: {e}")

#     if all_features:
#         X = np.array(all_features)
#         y = np.array(all_labels)

#         # ====== CALL TSNE ======
#         from scripts.Test_model.Test_TSNE import run_tsne

#         fig = run_tsne(X, y)

#         # ====== SAVE RESULT ======
#         evaluation = save_plot_to_evaluation(
#             training_session=session,
#             fig=fig,
#             eval_type="tsne"
#         )

#         return redirect("test_model_performance", model_id=model_id)

#     else:
#         return HttpResponse("No images with embeddings found in this session.")