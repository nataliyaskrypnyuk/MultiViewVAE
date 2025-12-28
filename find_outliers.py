import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
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
    os.makedirs(folder, exist_ok=True)
    embeddings_np = embeddings.numpy()
    rec_errors = []
    with torch.no_grad():
        for i, embedding_np in enumerate(embeddings_np):
            reconstructed = model.decode(torch.tensor(embedding_np).float().to(device)).cpu().squeeze()
            rec_errors.append(torch.mean((reconstructed - dataset.__getitem__(i)[0]) ** 2).item())

    outliers_10 = np.array(rec_errors).argsort()[-10:][::-1]
    
    outlier_pictures = []
    for outlier in outliers_10:
        picture = dataset.__getitem__(outlier)[0]
        outlier_pictures.append(picture)
    batch = torch.stack(outlier_pictures, dim=0)
    # 2 pictures per row
    save_image(
        batch, os.path.join(folder, "singleview_outliers.png"),
        nrow=2,
        padding=10,
        normalize=False,
        value_range=(0, 1),
        pad_value = 1
    )