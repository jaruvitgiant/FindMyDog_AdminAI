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

def Kmean():
    all_images = DogImage.objects.all()