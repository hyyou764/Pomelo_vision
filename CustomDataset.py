import torch
from torch.utils.data import Dataset
import numpy as np
import os


class ActionDataset(Dataset):
    def __init__(self, data_path, actions_list):
        self.data_path = data_path
        self.actions_list = actions_list
        self.file_list = []
        self.labels = []


        self.label_map = {action: i for i, action in enumerate(self.actions_list)}
        for action in self.actions_list:
            action_dir = os.path.join(self.data_path, action)
            if not os.path.exists(action_dir):
                print(f"警告: 找不到文件夹 {action_dir}")
                continue

            files = [f for f in os.listdir(action_dir) if f.endswith('.npy')]
            for file in files:
                self.file_list.append(os.path.join(action_dir, file))
                self.labels.append(self.label_map[action])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx]).astype(np.float32)
        label = self.labels[idx]

        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)