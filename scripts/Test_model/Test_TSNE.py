import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def plot_tsne_embeddings(model, test_loader, device, model_path=None, num_classes_to_show=30):

    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    all_features, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    
    unique_classes = torch.unique(all_labels)
    selected_classes = unique_classes[:num_classes_to_show]

    mask = torch.isin(all_labels, selected_classes)
    subset_features = all_features[mask].numpy()
    subset_labels = all_labels[mask].numpy()

    perplexity_value = min(30, (len(subset_features) - 1) // 3)
    print(f"Using perplexity = {perplexity_value}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
    features_2d = tsne.fit_transform(subset_features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=subset_labels, cmap='jet', s=10, alpha=0.8)
    plt.colorbar(scatter)
    plt.title("t-SNE Embeddings")
    # plt.show()