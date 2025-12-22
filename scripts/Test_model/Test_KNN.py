import torch
from scripts.resnet18 import ResNet18
from django.db import models
from scripts.resnet18 import ResNet18
from myapp.models import Dog, DogImage, User,Notification, AdoptionParent
from django.db.models import Q
from django.http import JsonResponse
import io
import numpy as np
from torchvision import transforms
from myapp.models import DogImage
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from myapp.models import EvaluationResult
from django.core.files.base import ContentFile

def fit_model_knn(X_train, y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train) 
    y_pred = knn.predict(X_test)        
    return y_pred

def save_plot_to_evaluation(training_session, fig, eval_type, score=None):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)

    ev = EvaluationResult.objects.create(
        training_session=training_session,
        eval_type=eval_type,
        score=score
    )

    ev.image.save(
        f"{eval_type}_session_{training_session.id}.png",
        ContentFile(buffer.read()),
        save=True
    )
    

def run_knn(X_train, y_train, X_test, y_test, save_path):
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine"
    )
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    plt.figure()
    plt.bar(["Accuracy"], [acc])
    plt.ylim(0, 1)
    plt.title("KNN Accuracy")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    return acc
