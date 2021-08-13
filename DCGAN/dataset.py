import torch
import os
import torchvision.transforms as transforms

from skimage import io
from torch.utils.data import Dataset

class Dataset(Dataset):
    
    def __init__(self, dir_path, img_size = 64, ):
        # super().__init__()는 쓰지 않아도 괜찮다.
        # torch.utils.data.Dataset에는 __init__이 정의되어 있지 않아서 의미있는 결과가 안나옴
        # 반면 torch.nn.Module에서는 필수임
        self.data_dir = dir_path
        self.transform = transforms.Compose([transforms.Resize(img_size),
                                             transforms.CenterCrop(img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        
        file_list = os.listdir(self.data_dir)
        file_list.sort()

        self.file_list = file_list                                

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.data_dir, self.file_list[idx])
        img = io.imread(img_name)        
        data = {'image': img}
        data = self.transform(data)
                    
        return data