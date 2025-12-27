import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from image_dataset import CaseMultiViewDataset
from visualize import visualize_samples_multiview, visualize_tsne_multiview
from multiview_model import vae_loss, ConvVAE
from image_dataset import CaseImageDataset

DATASET_FOLDER = "C:\\ITU"
FOLDER_TO_SAVE_FIGURES = ".\\SINGLEVIEW_VAE_FIGURES"

transform = transforms.ToTensor()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
dataset = CaseImageDataset(root_dir=DATASET_FOLDER, transform=transform)

def visualize_outliers_singleview(embeddings, model, folder, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    embeddings_np = embeddings.numpy()
    rec_errors = []
    with torch.no_grad():
        for i, embedding_np in enumerate(embeddings_np):
            reconstructed = model.decode(torch.tensor(embedding_np).float().to(device)).cpu().squeeze()
            rec_errors.append(torch.mean((reconstructed - dataset.__getitem__(i)[0]) ** 2).item())

    outliers_10 = np.array(rec_errors).argsort()[-10:][::-1]
    
    save_image(
        dataset.__getitem__(outliers_10[0])[0], os.path.join(folder, "singleview_outlier.png"),
        padding=0,
        format="PNG"
    )