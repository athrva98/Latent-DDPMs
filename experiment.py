from typing import List
import torch
import torch.utils.data
from base import DenoiseDiffusion
from unet import UNet
from datasets import CarsDataset
from LatentDiffusionModel import LatentEncoderDecoder, load_model
from torchinfo import summary

class Configs:
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iter_counter : int = 0

    eps_model: UNet
    diffusion: DenoiseDiffusion
    image_channels: int = 1
    image_size: int = 32
    n_channels: int = 64
    channel_multipliers: List[int] = [1, 2, 2, 4]
    is_attention: List[int] = [False, False, False, True]

    n_steps: int = 1_000
    batch_size: int = 10
    n_samples: int = 16
    learning_rate: float = 2e-5
    epochs: int = 1_000
    dataset_path: str = '/home/athrva/Desktop/DiffusionModels/Autoencoder/data/train/' 
    dataset: torch.utils.data.Dataset = CarsDataset(dataset_path)
    data_loader: torch.utils.data.DataLoader
    optimizer: torch.optim.Adam

    def init(self, debug=True):
        base_encoder_decoder = load_model()
        self.latent_model = LatentEncoderDecoder(model=base_encoder_decoder) # this maps images to the Latent space
        if debug:
            summary(base_encoder_decoder, input_size=(1,3,224,224))
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        self.diffusion = DenoiseDiffusion(
            latent_model=self.latent_model,
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)
            for t_ in range(self.n_steps):
                t = self.n_steps - t_ - 1
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))


    def train(self):
        for _, data in enumerate(self.data_loader):
            self.iter_counter += 1
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.diffusion.loss(data)
            loss.backward()
            self.optimizer.step()
            if self.iter_counter % 1_00 == 0: # saves the model every 100 updates
                self.diffusion.serialize_eps_model(self.iter_counter, self.optimizer, loss)

    def run(self):
        """
        ### Training loop
        """
        for _ in range(self.epochs):
            self.train()
            self.sample()


def main():
    configs = Configs()
    configs.init()
    configs.run()


if __name__ == '__main__':
    main()
