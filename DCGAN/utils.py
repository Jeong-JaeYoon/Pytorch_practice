import os
import shutil
import torch

def init_weights(m):
    
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weigth.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def save_checkpoints(state, checkpoint, Gen):
    
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    else :
        pass

    if Gen :
        filepath = os.path.join(checkpoint, 'last_Gen.pth')
    else :
        filepath = os.path.join(checkpoint, 'last_Dis.pth')

    torch.save(state, filepath)