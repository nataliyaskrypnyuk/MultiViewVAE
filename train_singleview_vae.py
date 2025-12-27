import mlflow
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from image_dataset import CaseMultiViewDataset
from visualize import visualize_samples_multiview, visualize_tsne_multiview, visualize_pca_multiview

DATASET_FOLDER = "C:\\ITU"
MLFLOW_EXPERIMENT_NAME = "VAE_experiment_multiview_test"
MLFLOW_RUN_NAME = "vae_test_run"
FOLDER_TO_SAVE_FIGURES = "./MULTIVIEW_VAE_FIGURES"

transform = transforms.ToTensor()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = CaseMultiViewDataset(root_dir=DATASET_FOLDER, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvVAE().to(device)
optimizer = optim.AdamW(model.parameters())

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
# Training loop
with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
    mlflow.log_param("latent_dim", model.latent_dim)
    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_rec = 0
        train_kld = 0
        beta = epoch + 1
        for batch_idx, (data, _) in enumerate(dataloader):
            data = [image.to(device) for image in data]
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            (rec, kld) = vae_loss(recon_batch, data, mu, logvar)
            loss = rec + kld * beta # first experiment - beta = (epoch / 2 + 1)
            loss.backward()
            train_loss += loss.item()
            train_rec += rec.item()
            train_kld += kld.item()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {train_loss / len(dataloader.dataset):.4f}')
        print(f'Epoch {epoch+1}, KL divergence: {train_kld / len(dataloader.dataset):.4f}')
        print(f'Epoch {epoch+1}, Reconstruction Loss: {train_rec / len(dataloader.dataset):.4f}')
        mlflow.log_metric("recon_loss_train", train_rec / len(dataloader.dataset), step=epoch)
        mlflow.log_metric("kld_loss_train", train_kld / len(dataloader.dataset), step=epoch)
        mlflow.log_metric("loss_train", train_loss / len(dataloader.dataset), step=epoch)
        mlflow.log_metric("kld_beta", beta, step=epoch)

model.eval()
embeddings = []
logvars = []
caseids = []
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
with torch.no_grad():
    for data, target in dataloader:
        data = [image.to(device) for image in data]
        mu, logvar = model.encode(data)
        embeddings.append(mu.cpu())
        logvars.append(logvar.cpu())
        caseids.extend(list(target))
embeddings = torch.cat(embeddings)

visualize_samples_multiview(model, FOLDER_TO_SAVE_FIGURES)

visualize_tsne_multiview(embeddings, FOLDER_TO_SAVE_FIGURES)

visualize_pca_multiview(embeddings, FOLDER_TO_SAVE_FIGURES)
