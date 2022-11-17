from typing import List

import torch
import torch.utils.data
import torchvision
from PIL import Image

from base import DenoiseDiffusion
from model import UNet


class Configs:
    device: torch.device = device

    eps_model: UNet
    diffusion: DenoiseDiffusion
    image_channels: int = 3
    image_size: int = 32
    n_channels: int = 64
    channel_multipliers: List[int] = [1, 2, 2, 4]
    is_attention: List[int] = [False, False, False, True]

    n_steps: int = 1_000
    batch_size: int = 64
    n_samples: int = 16
    learning_rate: float = 2e-5
    epochs: int = 1_000

    dataset: torch.utils.data.Dataset
    data_loader: torch.utils.data.DataLoader
    optimizer: torch.optim.Adam

    def init(self):
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        self.diffusion = DenoiseDiffusion(
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
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.diffusion.loss(data)
            loss.backward()
            self.optimizer.step()
            self.save_loss('loss', loss)

    def run(self):
        """
        ### Training loop
        """
        for _ in range(self.epochs):
            self.train()
            self.sample()
            self.save_checkpoint()


class CelebADataset(torch.utils.data.Dataset):
    """
    ### CelebA HQ dataset
    """

    def __init__(self, image_size: int):
        super().__init__()

        # CelebA images folder
        folder = lab.get_data_path() / 'celebA'
        # List of files
        self._files = [p for p in folder.glob(f'**/*.jpg')]

        # Transformations to resize the image and convert to tensor
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        """
        Size of the dataset
        """
        return len(self._files)

    def __getitem__(self, index: int):
        """
        Get an image
        """
        img = Image.open(self._files[index])
        return self._transform(img)


def celeb_dataset(c: Configs):
    """
    Create CelebA dataset
    """
    return CelebADataset(c.image_size)


class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


@option(Configs.dataset, 'MNIST')
def mnist_dataset(c: Configs):
    """
    Create MNIST dataset
    """
    return MNISTDataset(c.image_size)


def main():
    configs = Configs()
    configs.init()
    configs.run()


if __name__ == '__main__':
    main()
