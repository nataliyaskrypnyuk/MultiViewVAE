import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256, input_size=(256, 256)):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        
        # Compute encoder output shape dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            h = self._encode_conv(dummy)
            self.enc_shape = h.shape[1:]  # (C, H, W)
            self.enc_out_dim = h.numel()  # Flatten size

        # Latent layers
        self.enc_fc = nn.Linear(self.enc_out_dim, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder with 3 heads
        self.dec_fc1 = nn.Linear(latent_dim, 512)
        self.dec_fc2 = nn.Linear(512, self.enc_out_dim)
        self.dec_deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_deconv_head1 = nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1)
        self.dec_deconv_head2 = nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1)
        self.dec_deconv_head3 = nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1)

    def _encode_conv(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))  
        return x

    def encode(self, views):
        embeddings = []
        for view in views:
            x = self._encode_conv(view)
            x = x.view(x.size(0), -1)
            x = F.relu(self.enc_fc(x))
            embeddings.append(x)
        aggregated = torch.mean(torch.stack(embeddings), dim = 0) # mean pooling
        return self.fc_mu(aggregated), self.fc_logvar(aggregated)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        # Reshape dynamically using stored encoder shape
        h = h.view(-1, *self.enc_shape)
        h = F.relu(self.dec_deconv1(h))
        h = F.relu(self.dec_deconv2(h))
        return torch.sigmoid(self.dec_deconv_head1(h)), torch.sigmoid(self.dec_deconv_head2(h)), torch.sigmoid(self.dec_deconv_head3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    REC1 = F.mse_loss(recon_x[0], x[0], reduction='sum') # Compare to first view
    REC2 = F.mse_loss(recon_x[1], x[1], reduction='sum') # Compare to second view
    REC3 = F.mse_loss(recon_x[2], x[2], reduction='sum') # Compare to third view
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return ((REC1 + REC2 + REC3) / 3, KLD)
