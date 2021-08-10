import torch
from torch.utils.data import Dataset
import os

class Dataset(Dataset):
    
    def __init__(self, dir, ):
        # super().__init__()는 쓰지 않아도 괜찮다.
        # torch.utils.data.Dataset에는 __init__이 정의되어 있지 않아서 의미있는 결과가 안나옴
        # 반면 torch.nn.Module에서는 필수임
        self.data_dir = dir