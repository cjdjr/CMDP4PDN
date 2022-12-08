import torch.nn as nn
import torch
import numpy as np

class transition_model(nn.Module):
    def __init__(self, input_dim = 1364+38, output_dim=322):
        super(transition_model, self).__init__()
        lys=[]
        self.stem=nn.Linear(input_dim , 512)
        lys.append(nn.ReLU())
        for x in range(2):
            lys.append(nn.Linear(512,512))
            # lys.append(nn.Dropout(0.5))
            lys.append(nn.ReLU())
        self.ly=nn.Sequential(*lys)
        self.out=nn.Sequential(nn.Linear(512,output_dim))

    def forward(self, x):
        x=self.stem(x)
        return self.out(self.ly(x))

class transition_model_linear(nn.Module):
    def __init__(self, input_dim = 1364+38, action_dim=38, output_dim=322):
        super(transition_model_linear, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        lys=[]
        self.stem=nn.Linear(input_dim , 1024)
        lys.append(nn.ReLU())
        for x in range(1):
            lys.append(nn.Linear(1024,1024))
            # lys.append(nn.Dropout())
            lys.append(nn.ReLU())
        lys.append(nn.Linear(1024,512))
        lys.append(nn.ReLU())
        for x in range(1):
            lys.append(nn.Linear(512,512))
            # lys.append(nn.Dropout())
            lys.append(nn.ReLU())
        self.ly=nn.Sequential(*lys)
        self.out=nn.Sequential(nn.Linear(512,output_dim*action_dim + output_dim))

    def forward(self, input):
        x=self.stem(input)
        out = self.out(self.ly(x))
        B, _ = input.shape
        q = input[:,-self.action_dim:,None]
        weight, bias = out[:,:self.output_dim * self.action_dim].view(B,self.output_dim,self.action_dim), out[:,self.output_dim * self.action_dim:]
        return torch.bmm(weight, q).squeeze(-1) + bias

    def get_coff(self, state, q):
        x = torch.cat((state,q),dim=1)
        x=self.stem(x)
        out = self.out(self.ly(x))
        B, _ = x.shape
        weight, bias = out[:,:self.output_dim * self.action_dim].view(B,self.output_dim,self.action_dim), out[:,self.output_dim * self.action_dim:]
        return weight.squeeze(dim=0).detach().cpu().numpy(), bias.squeeze(dim=0).detach().cpu().numpy()


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(x + self.feed_forward(x))
        return x

class transition_model_residual(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 256):
        super(transition_model_residual, self).__init__()
        lys=[]
        self.stem=nn.Linear(input_dim , hidden_dim)
        lys.append(nn.ReLU())
        for x in range(2):
            lys.append(ResidualBlock(hidden_dim))
        self.ly=nn.Sequential(*lys)
        self.out=nn.Sequential(nn.Linear(hidden_dim,output_dim))

    def forward(self, x):
        x=self.stem(x)
        return self.out(self.ly(x))



class VoltageNet(nn.Module):
    """Residual network for voltage prediction."""

    def __init__(self, input_shape, output_shape, hid_size = 256, hid_depth = 2):
        """initialize."""
        super(VoltageNet, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hid_size = hid_size
        self.hid_depth = hid_depth
        self.activation = nn.ReLU()
        self.input = nn.Sequential(
            nn.Linear(self.input_shape, self.hid_size),
            nn.BatchNorm1d(self.hid_size),
            # nn.LayerNorm(self.hid_size),
            nn.ReLU()
        )
        self.ResBlocks = nn.ModuleList([
            *[self._make_ResBlock(self.hid_size, self.hid_size, self.hid_size) for _ in range(0, self.hid_depth)]
        ])
        self.output = nn.Linear(self.hid_size, self.output_shape)

    def _make_ResBlock(self, in_features, hid_features, out_features):
        """Build the residual block."""
        return nn.Sequential(
            nn.Linear(in_features, hid_features),
            nn.BatchNorm1d(hid_features),
            self.activation,
            nn.Linear(hid_features, out_features),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x):
        """Predict the voltages of all buses."""
        x = self.input(x)
        for ResBlock in self.ResBlocks:
            now_x = self.activation(ResBlock(x) + x)
            x = now_x
        return self.output(x)

    # def correct(self, act, state, v_lower, v_upper, bus_vol):
    #     """Map actions to safe ranges."""
    #     bus_num = bus_vol.shape[0]

    #     P = matrix(np.eye(act.shape[0]))
    #     q = matrix(np.zeros(act.shape[0], dtype=np.float64))
    #     # q = matrix(np.float64(-2 * act))

    #     s = torch.cat([torch.Tensor(state), torch.Tensor(act)], dim=-1).unsqueeze(0).to(next(self.parameters()).device)
    #     self.eval()

    #     V = self(s)
    #     # if np.sum(V.detach().numpy() > v_upper) + np.sum(V.detach().numpy() < v_lower) <= 0 and np.sum(bus_vol > v_upper) + np.sum(bus_vol < v_lower) <= 0:
    #     #     return act

    #     v = V.detach().squeeze().numpy()
    #     if np.sum(v > v_upper) + np.sum(v < v_lower) <= 0:
    #         return act

    #     G = jacobian(func=lambda x: self(x), inputs=s, vectorize=True).squeeze()[:, -act.shape[0]:]
    #     G = matrix(np.float64(np.vstack((G, -G))))

    #     H = np.zeros((bus_num - 1) * 2)
    #     # H[:bus_num-1] = v_upper - bus_vol[1:]
    #     # H[bus_num-1:] = -v_lower + bus_vol[1:]
    #     H[:bus_num-1] = v_upper - v
    #     H[bus_num-1:] = -v_lower + v
    #     H = matrix(np.float64(H))
    #     try:
    #         sv = cvxopt.solvers.qp(P=P, q=q, G=G, h=H)
    #     except Exception as e:
    #         print(e)
    #         return act

    #     # return np.squeeze(np.array(sv['x']))
    #     return np.squeeze(np.array(sv['x'])) + act