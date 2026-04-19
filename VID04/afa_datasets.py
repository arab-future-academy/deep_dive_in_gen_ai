from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CelebADataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            msg = f"image forlder path ({root_dir}) does not exists!"
            raise Exception(msg)
        self.transform = transform
        self.image_pathes = sorted(self.root_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.image_pathes)
    
    def __getitem__(self, index):
        path = self.image_pathes[index]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image