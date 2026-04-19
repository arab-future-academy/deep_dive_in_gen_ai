
import torch
import torch.nn as nn

from afa_datasets import CelebADataSet
from afa_save_load import resume_checkpoint, save_checkpoint

from pathlib import Path
import torchvision


class VAE(nn.Module):
    def __init__(self, input_channels=3, encoder_feature_size=128,
                 decoder_feature_size=128, latent_dim=500):
        super().__init__()
        self.latent_dim = latent_dim

        # construct the features dimentions for example  [3, 128, 256, 512, 1024, 1024]
        n = [2**i * encoder_feature_size for i in range(4)]
        n = [input_channels] + n + [n[-1]]

        # construct encoder layer
        layers = [ 
            VAE.conv_block(n[i], n[i+1], 4, 2, 1) for i in range(0, len(n)-1) 
        ] + [ nn.Flatten() ]
        self.encoder_layers = nn.Sequential(*layers)

        self.encoder_mu = nn.Linear( in_features=encoder_feature_size*8*4*4, out_features=latent_dim )
        self.encoder_logvar = nn.Linear( in_features=encoder_feature_size*8*4*4, out_features=latent_dim )

        # Decoder layers 
        n = n[::-1] # reverse n
        layers = ( 
            [ 
                nn.Linear( in_features=latent_dim, out_features=decoder_feature_size*8*4*4 ),
                nn.Unflatten(dim=1, unflattened_size=(decoder_feature_size*8, 4, 4)),
                nn.LeakyReLU(0.2, inplace=True),
            ] + 
            [ VAE.conv_transpose_block(n[i], n[i+1], 4, 2, 1) for i in range(len(n)-2) ] +  # we only need 4 layers
            [
                nn.ConvTranspose2d( in_channels=decoder_feature_size, out_channels=input_channels, kernel_size=4, stride=2, padding=1 ),
                nn.Sigmoid(),
            ]
        )
        self.decoder_layers = nn.Sequential(*layers)

    @staticmethod
    def conv_block( in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1 ) -> nn.Sequential:
        return nn.Sequential( 
            nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )


    @staticmethod
    def conv_transpose_block( in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1 ) -> nn.Sequential:
        return nn.Sequential( 
            nn.ConvTranspose2d( in_channels, out_channels, kernel_size, stride, padding ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    # Encoder forward
    def encode(self, x):
        h = self.encoder_layers(x)
        return self.encoder_mu(h), self.encoder_logvar(h)

    # Decoder forward
    def decode(self, z):
        return self.decoder_layers(z)


    # Reparameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # device-aware
        return mu + eps * std

    # Full forward
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar



# Use BCE with sum reduction instead of size_average
reconstruction_criterion: nn.BCELoss = nn.BCELoss(reduction='sum')


def vae_loss_function(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> torch.Tensor:
    bce_loss = reconstruction_criterion(reconstructed, original)

    kld_element = 1 + logvar - mu.pow(2) - logvar.exp()
    kld_loss = -0.5 * torch.sum(kld_element)

    total_loss = beta * bce_loss + kld_loss
    return total_loss


def train(epoch: int, model: nn.Module, train_loader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, 
          loss_function: torch.Tensor, device: torch.device) -> float:
    model.train()
    train_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)


        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        print(
            f"Train Epoch: {epoch} "
            f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
            f"({100. * batch_idx / len(train_loader):.0f}%)] "
            f"Loss: {loss.item() / len(data):.6f}"
        )

    avg_loss = train_loss / len(train_loader.dataset)

    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")

    return avg_loss



def test(
    epoch: int,
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_function: torch.Tensor,  # Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
    device: torch.device,
    save_dir: str = "imgs"
) -> float:
    model.eval()
    test_loss: float = 0.0

    # Paths for saving images
    path_test = Path(save_dir) / f"GT_Epoch_{epoch}.jpg"
    path_reco = Path(save_dir) / f"RECON_Epoch_{epoch}.jpg"
    path_test.parent.mkdir(parents=True, exist_ok=True)
    path_reco.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)

            # Forward pass
            recon_batch, mu, logvar = model(data)

            # Compute loss
            batch_loss = loss_function(recon_batch, data, mu, logvar)
            test_loss += batch_loss.item()

            # Save images (only for the first batch, optional)
            if batch_idx == 0:
                torchvision.utils.save_image(data, path_test, nrow=8, padding=2)
                torchvision.utils.save_image(recon_batch, path_reco, nrow=8, padding=2)

            break

    # Normalize loss by total number of images
    avg_test_loss: float = test_loss / len(test_loader.dataset)
    print(f"====> Test set loss: {avg_test_loss:.4f}")

    return avg_test_loss


if __name__ == "__main__":

    expermint = 4
    n_epochs = 40
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # converts to [0,1] tensor
    ])
    dataset = CelebADataSet(
        r"D:\a\learn\AI\Variational-Autoencoder-PyTorch\data-celeb-128\CelebA-img", 
        transform
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,      # use multiple CPU workers
        pin_memory=True     # faster GPU transfer
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,      # use multiple CPU workers
        pin_memory=True     # faster GPU transfer
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_channels=3, encoder_feature_size=128, decoder_feature_size=128, latent_dim=500).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    
    last_epoch, _ = resume_checkpoint( "models" + str(expermint), model, optimizer, device )

    for epoch in range(last_epoch+1, last_epoch+n_epochs+1):
        print(f"\n===== Epoch {epoch} =====")

        avg_train_loss: float = train( epoch, model, data_loader, optimizer, vae_loss_function, device)

        test_loss: float = test( epoch, model, test_loader, vae_loss_function, device, "imgs" + str(expermint) )

        save_checkpoint( model, optimizer, epoch, avg_train_loss, "models" + str(expermint) )
