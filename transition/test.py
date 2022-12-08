import torch
from model import transition_model_residual
from torch.utils.data import Dataset
import numpy as np
class TransitionDataset(Dataset):

    def __init__(self, data):
        self.state = torch.from_numpy(data['state']).to(torch.float32)
        self.q = torch.from_numpy(data['q']).to(torch.float32)
        self.res_v = torch.from_numpy(data['res_v']).to(torch.float32)

    def __getitem__(self, index):
        data = torch.cat((self.state[index],self.q[index]),dim=0)
        return data, self.res_v[index]
    def __len__(self):
        return self.state.shape[0]

if __name__ == "__main__":
    model = transition_model_residual(1402, 322, 512)
    model.load_state_dict(torch.load("/data1/wangmr/CMDP4PDN/transition/bus322/resnet_1_best_model"))
    model.eval()
    # model = model.cuda()
    _data = np.load("/data1/wangmr/CMDP4PDN/transition/bus322/200000.npy",allow_pickle=True).item()
    dataset = TransitionDataset(_data)
    x, y = dataset[1523]
    x = x.unsqueeze(0)
    pred = model(x).squeeze(0)
    print(pred - y)

    x = torch.load("/data1/wangmr/CMDP4PDN/transition/bus322/test_data").cpu()
    print(x)
    pred = model(x).squeeze(0)
    print(pred)