import mlflow
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from image_dataset import CaseMultiViewDataset
from visualize import visualize_samples_multiview, visualize_tsne_multiview
from multiview_model import vae_loss, ConvVAE

DATASET_FOLDER = "C:\\ITU"
MLFLOW_EXPERIMENT_NAME = "VAE_experiment_multiview_test"
MLFLOW_RUN_NAME = "vae_test_run"
FOLDER_TO_SAVE_FIGURES = ".\\MULTIVIEW_VAE_FIGURES"
EPOCHS_NUMBER = 10 # change the number of epochs to run here

transform = transforms.ToTensor()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

caseids_brackets = []
embeddings_brackets_average = np.zeros(embeddings_np.shape[1])
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
for case in cases_with_brackets:
    caseids_brackets.append(caseids.index(case))
    embeddings_brackets_average += embeddings_np[caseids.index(case)]
embeddings_brackets_average = embeddings_brackets_average / len(cases_with_brackets)
embeddings_np = np.append(embeddings_np, [embeddings_brackets_average], axis=0)

import faiss
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)
D, I = index.search(embeddings_np, k=10)

