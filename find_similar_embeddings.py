import mlflow
import numpy as np
import os
import faiss
from sklearn.decomposition import PCA
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

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

cases_with_brackets = ['20250724182936647-342e25b8-677a-46e9-82ce-8503251c3085/lower/lower_occlusal_256.png',
                       '20250724182936647-342e25b8-677a-46e9-82ce-8503251c3085/upper/upper_occlusal_256.png',
                       '20250724175749275-91645211-bb5e-481e-845a-8c2cadf3963f/lower/lower_occlusal_256.png',
                       '20250724175749275-91645211-bb5e-481e-845a-8c2cadf3963f/upper/upper_occlusal_256.png',
                       '20250724173642746-000fd235-9fea-4b83-b944-591015d5070b/lower/lower_occlusal_256.png',
                       '20250724173642746-000fd235-9fea-4b83-b944-591015d5070b/upper/upper_occlusal_256.png',
                       '20250724171159628-b8bfd8f7-d075-4883-8487-0f693ca5c0f6/lower/lower_occlusal_256.png',
                       '20250724171159628-b8bfd8f7-d075-4883-8487-0f693ca5c0f6/upper/upper_occlusal_256.png',
                       '20250724151839487-33993608-bd18-4276-a4c5-1417545ae7b2/lower/lower_occlusal_256.png',
                       '20250724151839487-33993608-bd18-4276-a4c5-1417545ae7b2/upper/upper_occlusal_256.png',
                      ]

def visualize_similar_cases_singleview(embeddings, caseids, folder, cases_given = cases_with_brackets):
    os.makedirs(folder, exist_ok=True)
    embeddings_np = embeddings.numpy()
    caseids_given = []
    embeddings_given_average = np.zeros(embeddings_np.shape[1])
    for case in cases_given:
        caseids_given.append(caseids.index(case))
        embeddings_given_average += embeddings_np[caseids.index(case)]
    embeddings_given_average = embeddings_given_average / len(cases_given)
    embeddings_np = np.append(embeddings_np, [embeddings_given_average], axis=0)

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    D, I = index.search(embeddings_np, k=11)
    # remove the last element which does not exist in the dataset as it is just the average embedding
    similar_cases = I[-1][I[-1] != embeddings_np.shape[0] - 1]

    similar_pictures = []
    for caseid in similar_cases:
        picture = dataset.__getitem__(caseid)[0]
        similar_pictures.append(picture)
    batch = torch.stack(similar_pictures, dim=0)
    # 2 pictures per row
    save_image(
        batch, os.path.join(folder, "singleview_similar.png"),
        nrow=2,
        padding=10,
        normalize=False,
        value_range=(0, 1),
        pad_value = 1
    )

