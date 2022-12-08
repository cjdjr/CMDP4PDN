
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.utils.data import random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import argparse
import wandb
from model import transition_model, transition_model_linear, transition_model_residual, VoltageNet

NUM_EPOCHS = 2000
LR = 0.001
SAVE_INTERVAL = 10
BATCH_SIZE = 1024
HIDDEN_DIM = 512

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


def get_args():
    parser = argparse.ArgumentParser(description="Train rl agent.")
    parser.add_argument("--name", type=str, nargs="?", help="Please input the valid name of an environment scenario.")
    parser.add_argument("--dataset", type=str, nargs="?")
    parser.add_argument("--save-path", type=str, nargs="?", default="./", help="Please enter the directory of saving model.")
    parser.add_argument("--wandb",  action='store_true')
    args = parser.parse_args()
    args.num_epochs = NUM_EPOCHS
    args.lr = LR
    args.save_interval = SAVE_INTERVAL
    args.batch_size = BATCH_SIZE
    args.hidden_dim = HIDDEN_DIM
    return args

if __name__=="__main__":
    args = get_args()
    if args.wandb:
        wandb.init(
            project='mapdn_cmdp_transition_model',
            entity="chelly",
            name=args.name,
            group='_'.join(args.name.split('_')[:-1]),
            save_code=True
        )
        wandb.config.update(args)
        wandb.run.log_code('.')

    _data = np.load(args.dataset,allow_pickle=True).item()
    dataset = TransitionDataset(_data)
    len_train = int(len(dataset) * 0.8)
    len_val = len(dataset) - len_train
    train_dataset, valid_dataset = random_split(
        dataset=dataset,
        lengths=[len_train, len_val],
        generator=torch.Generator().manual_seed(0)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # model=transition_model().cuda()
    e_data, e_label = dataset[0]
    input_dim = e_data.shape[-1]
    output_dim = e_label.shape[-1]
    print("input_dim = {}   output_dim = {}".format(input_dim, output_dim))
    model=transition_model_residual(input_dim, output_dim, args.hidden_dim).cuda()
    # model = VoltageNet(input_dim, output_dim, args.hidden_dim).cuda()

    loss_func=nn.MSELoss()
    optm=torch.optim.Adam(model.parameters(),args.lr)
    scheduler = lr_scheduler.StepLR(optm, step_size=100, gamma=0.8)
    train_epochs_loss = []
    valid_epochs_loss = []
    # acc=acc_func()
    for epoch in range(args.num_epochs):
        train_loss = []
        scheduler.step()
        for idx,(data_x, data_y) in enumerate(train_dataloader,0):
            data_x = data_x.cuda()
            data_y = data_y.cuda()
            outputs = model(data_x)
            optm.zero_grad()
            loss = loss_func(data_y,outputs)
            loss.backward()
            optm.step()
            train_loss.append(loss.item())
            # train_loss.append(loss.item())
            if idx%(len(train_dataloader)//2)==0:
                print("epoch={}/{},{}/{}of train, loss={}".format(
                    epoch, NUM_EPOCHS, idx, len(train_dataloader),loss.item()))
        if args.wandb:
            wandb.log({"train_l2_loss": np.average(train_loss)},epoch)
        train_epochs_loss.append(np.average(train_loss))

        #=====================valid============================

        valid_loss = []
        with torch.no_grad():
            for idx,(data_x,data_y) in enumerate(valid_dataloader,0):
                data_x = data_x.cuda()
                data_y = data_y.cuda()
                outputs = model(data_x)
                loss = torch.mean(torch.abs(outputs - data_y))
                valid_loss.append(loss.item())
        print("val epoch = {}   :    {}".format(epoch,np.average(valid_loss)))
        if args.wandb:
            wandb.log({"val_l1_loss": np.average(valid_loss)},epoch)
        if epoch == 0 or np.average(valid_loss) < np.min(valid_epochs_loss):
            path = args.save_path + "/"+args.name+"_best_model"
            print("SAVE best model to {}".format(path))
            torch.save(model.state_dict(), path)
        valid_epochs_loss.append(np.average(valid_loss))

        # if epoch % args.save_interval ==0 :
        #     # path = args.scenario + '.res_model{}'.format(epoch)
        #     path = args.save_path + "/res_model_h={}_{}".format(args.hidden_dim,epoch)
        #     print("SAVE model to {}".format(path))
        #     torch.save(model.state_dict(), path)

        # fig = plt.figure()
        # plt.plot(np.arange(len(train_epochs_loss)), train_epochs_loss)
        # plt.xlabel("epoch")
        # plt.title("train_loss")
        # fig.savefig(args.save_path+"/"+args.name+"_train_loss.png".format(args.hidden_dim))
        # plt.close()

        # fig = plt.figure()
        # plt.plot(np.arange(len(valid_epochs_loss)), valid_epochs_loss)
        # plt.xlabel("epoch")
        # plt.title("valid_loss")
        # fig.savefig(args.save_path+"/"+args.name+"_valid_loss.png".format(args.hidden_dim))
        # plt.close()



    print("Best model : {}".format(np.min(valid_epochs_loss)))