import torch.nn as nn
import torch

class PreAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(PreAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hid_size)
        if args.layernorm:
            self.layernorm = nn.LayerNorm(args.hid_size)
        self.rnn = nn.GRUCell(args.hid_size, args.hid_size)
        self.fc2 = nn.Linear(args.hid_size, args.action_dim)

        if self.args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif self.args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.agent_num, self.args.hid_size).zero_()

    def forward(self, inputs, hidden_state):
        x = self.fc1(inputs)
        if self.args.layernorm:
            x = self.layernorm(x)
        x = self.hid_activation(x)
        h_in = hidden_state.reshape(-1, self.args.hid_size)
        h = self.rnn(x, h_in)
        a = self.fc2(h)
        return a, None, h, x

class MLPAgent(nn.Module):
    def __init__(self, input_shape, args, id_dim=16):
        super(MLPAgent, self).__init__()
        self.args = args

        # Easiest to reuse hid_size variable
        self.obs_enc = nn.Linear(input_shape, args.hid_size)
        self.id_emb = nn.Embedding(args.agent_num, id_dim)
        self.act_enc = nn.Linear(args.action_dim, args.hid_size - id_dim)
        self.fc1 = nn.Linear(args.hid_size * 3, args.hid_size)
        self.fc2 = nn.Linear(args.hid_size, args.action_dim)
        if args.layernorm:
            self.layernorm = nn.LayerNorm(args.hid_size)

        if self.args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif self.args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.obs_enc.weight.new(1, self.args.hid_size).zero_()

    def forward(self, inputs, actions, ids, hidden_state):
        x = self.obs_enc(inputs)
        if self.args.layernorm:
            x = self.layernorm(x)
        x = self.hid_activation(x)

        act_rep = torch.cat((self.act_enc(actions), self.id_emb(ids)),dim=-1)
        if self.args.layernorm:
            act_rep = self.layernorm(act_rep)
        act_rep = self.hid_activation(act_rep)

        x = self.hid_activation(self.fc1(torch.cat((x, act_rep, hidden_state),dim=-1)))
        a = self.fc2(x)
        return a


class MLPAgent1(nn.Module):
    def __init__(self, input_shape, args, id_dim=16):
        super(MLPAgent1, self).__init__()
        self.args = args

        # Easiest to reuse hid_size variable
        self.obs_enc = nn.Linear(input_shape, args.hid_size)
        self.id_emb = nn.Embedding(args.agent_num, id_dim)
        self.act_enc = nn.Linear(args.action_dim, args.hid_size - id_dim)
        self.fc1 = nn.Linear(args.hid_size * 2, args.hid_size)
        self.fc2 = nn.Linear(args.hid_size, args.action_dim)
        if args.layernorm:
            self.layernorm = nn.LayerNorm(args.hid_size)

        if self.args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif self.args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.obs_enc.weight.new(1, self.args.hid_size).zero_()

    def forward(self, inputs, actions, ids, hidden_state):
        x = self.obs_enc(inputs)
        if self.args.layernorm:
            x = self.layernorm(x)
        x = self.hid_activation(x)

        act_rep = torch.cat((self.act_enc(actions), self.id_emb(ids)),dim=-1)
        act_rep = act_rep + hidden_state
        if self.args.layernorm:
            act_rep = self.layernorm(act_rep)
        act_rep = self.hid_activation(act_rep)

        x = self.hid_activation(self.fc1(torch.cat((x, act_rep),dim=-1)))
        a = self.fc2(x)
        return a