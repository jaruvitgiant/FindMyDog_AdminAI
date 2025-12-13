
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
from myapp.models import TrainingSession

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
embedding_size = 512

class FaceNetModel(nn.Module):
    def __init__(self,embedding_size=embedding_size):
        super().__init__()
        self.backbone = ResNet18()
        self.embedding = nn.Linear(512, embedding_size) 

    def forward(self, x):
        features_512 = self.backbone.extract_features(x)
        embeddings = self.embedding(features_512)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings 
    
def train_resnet50(request):
    pass

def model_EMResnet18(embedding_size=embedding_size): ##weight_path='scripts/Traindog_face_lmargin_70.pt'
    model = FaceNetModel(embedding_size)
    # state_dict = torch.load('scripts/Traindog_face_lmargin_70.pt')
    # model.load_state_dict(state_dict)
    weight_path = os.path.join('scripts', 'Traindog_face_lmargin_70.pt')
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    # model.eval()

    # model = ResNet18(embedding_size=embedding_size)
    # state = torch.load(weight_path, map_location=device)
    # model.load_state_dict(state)
    # model = model.to(device)
    return model

# def save_embedding_to_db(img_obj, embedding):
#     em_np = embedding.squeeze().cpu().numpy().astype(np.float32)
#     img_obj.embedding_binary = em_np.tobytes()
#     img_obj.save()

def save_embedding_to_db(img_obj, embedding, session):

    embedding_np = embedding.squeeze().cpu().numpy().astype("float32")

    img_obj.embedding_binary = embedding_np.tobytes()
    img_obj.training_session = session
    img_obj.save(update_fields=[
        "embedding_binary",
        "training_session"
    ])

# def train_resnet18():
#     model = model_EMResnet18()
#     model.eval()

#     #all_images = DogImage.objects.all()
#     all_images = DogImage.objects.filter(
#         embedding_binary__isnull=True
#     )
#     with torch.no_grad():
#         for img_obj in all_images:

#             if img_obj.embedding_binary:
#                 # print(f"Skip: image {img_obj.id} already has embedding.")
#                 continue

#             img_path = img_obj.image.path

#             img = Image.open(img_path).convert("RGB")
#             img_tensor = transform(img).unsqueeze(0)

#             embedding = model(img_tensor)
#             save_embedding_to_db(img_obj, embedding)

#     return True

def train_resnet18(session):
    model = model_EMResnet18()
    model.eval()

    # train เฉพาะรูปที่ยังไม่มี embedding
    all_images = DogImage.objects.filter(
        embedding_binary__isnull=True
    )

    with torch.no_grad():
        for img_obj in all_images:

            img_path = img_obj.image.path
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)

            embedding = model(img_tensor)

            # 👇 สำคัญมาก
            save_embedding_to_db(
                img_obj,
                embedding,
                session
            )

    return True


