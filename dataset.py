from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class PairedGrayDataset(Dataset):
    def __init__(self, root, phase='train', loadSize=1024):
        self.a_dir = os.path.join(root, phase, 'A')
        self.b_dir = os.path.join(root, phase, 'B')
        self.files = sorted(os.listdir(self.a_dir))
        self.transformA = T.Compose([
            T.Resize((loadSize, loadSize)),
            T.RandomHorizontalFlip() if phase=='train' else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize([0.5],[0.5])
        ])
        self.transformB = self.transformA
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        a = Image.open(os.path.join(self.a_dir, self.files[idx])).convert('L')
        b = Image.open(os.path.join(self.b_dir, self.files[idx])).convert('L')
        return self.transformA(a), self.transformB(b)
