import torch
from torch.utils.data import Dataset
import numpy as np
import json

class ASRDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        feature_path = sample["feature_filepath"]
        text = sample["text"]

        # 加载 .npy 作为特征
        features = torch.tensor(np.load(feature_path), dtype=torch.float32)
        # 这里假设 text 需要转成 token id
        label = torch.tensor([ord(c) for c in text], dtype=torch.long)

        return features, label