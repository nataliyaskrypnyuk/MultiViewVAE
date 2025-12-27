import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
# visualize + find closest (fassn) + T-SNE
# ----------------------------
# Visualization
# ----------------------------
def visualize_samples_singleview(model, folder, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_samples=4):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z)
        samples = [sample.cpu() for sample in samples]

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        img = samples[i][0].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        img = samples[i][1].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        img = samples[i][2].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def visualize_samples_multiview(model, folder, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_samples=4):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z)
        samples = [sample.cpu() for sample in samples]
    batch = torch.cat(samples, dim=0)
    batch = batch.detach().cpu()
    # 4 pictures per row
    save_image(
        batch, os.path.join(folder, "multiview_vae_samples.png"),
        nrow=4,
        padding=10,
        normalize=False,
        value_range=(0, 1),
        pad_value = 1
    )

def visualize_tsne_multiview(embeddings, caseids, folder):
    os.makedirs(folder, exist_ok=True)
    embeddings_np = embeddings.numpy()
    lowervsupper = [case.split("/")[1] for case in caseids]
    unique_labels = list(set(lowervsupper))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    colors = [label_to_int[l] for l in lowervsupper]

    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(embeddings_np[:3540])

    # Plot
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors, cmap='viridis')
    plt.colorbar()
    plt.title("VAE Latent Space Visualization with t-SNE, yellow is " + unique_labels[1])
    plt.savefig(os.path.join(folder, "multiview_vae_tsne.png"))
