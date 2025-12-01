from django import forms

MODEL_CHOICES = [
    ('key-value', 'Key-value & Table'),
    ('document-classification', 'Document Classification'),
    ('entity-extraction', 'Entity Extraction'),
    ('ocr', 'OCR Model'),
    ('resnet18', 'ResNet18 (Dog Detection)'),
]

class TrainingForm(forms.Form):
    training_name = forms.CharField(
        label="Training Name",
        max_length=255,
        required=True,
        widget=forms.TextInput(attrs={
            'placeholder': 'Enter training name',
            'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500'
        })
    )
    
    model_name = forms.ChoiceField(
        label="Select Model",
        choices=MODEL_CHOICES,
        required=True,
        widget=forms.Select(attrs={
            'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500'
        })
    )
    
    dataset_count = forms.IntegerField(
        label="Number of Datasets",
        min_value=1,
        initial=1,
        required=True,
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500',
            'min': '1'
        })
    )
    
    epochs = forms.IntegerField(
        label="Epochs",
        min_value=1,
        initial=10,
        required=True,
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500',
            'min': '1'
        })
    )
