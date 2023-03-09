import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import select_action
from models.model import Model
from critics.mlp_critic import MLPCritic, MLP2HCritic



class ACSAFEMADDPG(Model):
    def __init__(self, args, target_net=None):
        super(ACSAFEMADDPG, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        # load parameters
        if self.args.safety_filter == "droop" or self.args.safety_filter == "droop_ind":
            self.safety_filter.pred_network.load_state_dict(th.load(self.args.pred_model_path))
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)
        self.beta = getattr(args, "safe_loss_beta", 1.0)
        self.multiplier = th.nn.Parameter(th.tensor(args.init_lambda,device=self.device))

    def construct_value_net(self):
        if self.args.agent_id:
            input_shape = (self.obs_dim + self.act_dim) * self.n_ + self.n_
        else:
            input_shape = (self.obs_dim + self.act_dim) * self.n_
        output_shape = 1
        if self.args.shared_params:
            self.value_dicts = nn.ModuleList( [ MLP2HCritic(input_shape, output_shape, self.args) ] )
        else:
            self.value_dicts = nn.ModuleList( [ MLP2HCritic(input_shape, output_shape, self.args) for _ in range(self.n_) ] )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def construct_policy_net(self):
        if self.args.agent_id:
            input_shape = self.obs_dim + self.n_
        else:
            input_shape = self.obs_dim

        if self.args.agent_type == 'mlp':
            if self.args.gaussian_policy:
                from agents.mlp_agent_gaussian import MLPAgent
            else:
                from agents.mlp_agent import MLPAgent
            Agent = MLPAgent
        elif self.args.agent_type == 'rnn':
            if self.args.gaussian_policy:
                from agents.rnn_agent_gaussian import RNNAgent
            else:
                from agents.rnn_agent import RNNAgent
            Agent = RNNAgent
        elif self.args.agent_type == 'diffusion':
            if self.args.gaussian_policy:
                NotImplementedError()
            else:
                from agents.diffusion_agent import Diffusion
            Agent = Diffusion
        else:
            NotImplementedError()

        if self.args.shared_params:
            self.policy_dicts = nn.ModuleList([ Agent(input_shape, self.args) ])
            if self.args.algorithm == "acsafemaddpg":
                from agents.correction_agent import PreAgent, MLPAgent
                self.policy_dicts = nn.ModuleList([ PreAgent(input_shape, self.args) ])
                self.correction_dicts = nn.ModuleList([ MLPAgent(input_shape, self.args) ])
        else:
            self.policy_dicts = nn.ModuleList([ Agent(input_shape, self.args) for _ in range(self.n_) ])

    def value(self, obs, act):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)

        obs_repeat = obs.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, o)
        obs_reshape = obs_repeat.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, n*o)

        # add agent id
        agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # shape = (b, n, n)
        if self.args.agent_id:
            obs_reshape = th.cat( (obs_reshape, agent_ids), dim=-1 ) # shape = (b, n, n*o+n)

        # make up inputs
        act_repeat = act.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, a)
        act_mask_others = agent_ids.unsqueeze(-1) # shape = (b, n, n, 1)
        act_mask_i = 1. - act_mask_others
        act_i = act_repeat * act_mask_others
        act_others = act_repeat * act_mask_i

        # detach other agents' actions
        act_repeat = act_others.detach() + act_i # shape = (b, n, n, a)

        if self.args.shared_params:
            obs_reshape = obs_reshape.contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*o+n)
            act_reshape = act_repeat.contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*a)
        else:
            obs_reshape = obs_reshape.contiguous().view( batch_size, self.n_, -1 ) # shape = (b, n, n*o+n)
            act_reshape = act_repeat.contiguous().view( batch_size, self.n_, -1 ) # shape = (b, n, n*a)

        inputs = th.cat( (obs_reshape, act_reshape), dim=-1 )

        if self.args.shared_params:
            agent_value = self.value_dicts[0]
            values, costs, _ = agent_value(inputs, None)
            values = values.contiguous().view(batch_size, self.n_, 1)
            costs = costs.contiguous().view(batch_size, self.n_, 1)
        else:
            values = []
            costs = []
            for i, agent_value in enumerate(self.value_dicts):
                value, cost, _ = agent_value(inputs[:, i, :], None)
                values.append(value)
                costs.append(cost)
            values = th.stack(values, dim=1)
            costs = th.stack(costs, dim=1)

        return values, costs

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        # obs_shape = (b, n, o)
        batch_size = obs.size(0)

        # add agent id
        if self.args.agent_id:
            agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # shape = (b, n, n)
            obs = th.cat( (obs, agent_ids), dim=-1 ) # shape = (b, n, n+o)
            ids = th.arange(self.n_).repeat(batch_size, 1).to(self.device)

        if self.args.shared_params:
            # print (f"This is the shape of last_hids: {last_hid.size()}")
            obs = obs.contiguous().view(batch_size*self.n_, -1) # shape = (b*n, n+o/o)
            ids = ids.contiguous().view(batch_size*self.n_)
            agent_policy = self.policy_dicts[0]
            pre_means, log_stds, hiddens, obs_enc = agent_policy(obs, last_hid)
            agent_correction = self.correction_dicts[0]
            corr_means = agent_correction(obs, pre_means, ids, hiddens)
            means = pre_means + corr_means
            # hiddens = th.stack(hiddens, dim=1)
            means = means.contiguous().view(batch_size, self.n_, -1)
            pre_means = pre_means.contiguous().view(batch_size, self.n_, -1)
            hiddens = hiddens.contiguous().view(batch_size, self.n_, -1)
            if self.args.gaussian_policy:
                log_stds = log_stds.contiguous().view(batch_size, self.n_, -1)
            else:
                stds = th.ones_like(means).to(self.device) * self.args.fixed_policy_std
                log_stds = th.log(stds)
        else:
            raise NotImplementedError

        return means, log_stds, hiddens, pre_means

    def get_actions(self, state, status, exploration, actions_avail, target=False, last_hid=None):
        target_policy = self.target_net.policy if self.args.target else self.policy
        if self.args.continuous:
            means, log_stds, hiddens, pre_means = self.policy(state, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
            if means.size(-1) > 1:
                means_ = means.sum(dim=1, keepdim=True)
                log_stds_ = log_stds.sum(dim=1, keepdim=True)
            else:
                means_ = means
                log_stds_ = log_stds
            actions, log_prob_a = select_action(self.args, means_, status=status, exploration=exploration, info={'log_std': log_stds_})
            restore_mask = 1. - (actions_avail == 0).to(self.device).float()
            restore_actions = restore_mask * actions
            action_out = (means, log_stds, pre_means)
        else:
            logits, _, hiddens = self.policy(state, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out, hiddens

    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, safe_actions, global_state, old_log_prob_a, old_values, old_next_values, rewards, costs, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=last_hids)
        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)
        compose = self.value(state, actions_pol)
        values_pol, costs_pol = compose[0].contiguous().view(-1, self.n_), compose[1].contiguous().view(-1, self.n_)
        compose = self.value(state, actions)
        values, costs = compose[0].contiguous().view(-1, self.n_), compose[1].contiguous().view(-1, self.n_)
        compose = self.target_net.value(next_state, next_actions.detach())
        next_values, next_costs = compose[0].contiguous().view(-1, self.n_), compose[1].contiguous().view(-1, self.n_)
        returns = th.zeros((batch_size, self.n_), dtype=th.float).to(self.device)
        cost_returns = th.zeros((batch_size, self.n_), dtype=th.float).to(self.device)
        assert values_pol.size() == next_values.size()
        assert returns.size() == values.size()
        done = done.to(self.device)
        returns = rewards + self.args.gamma * (1 - done) * next_values.detach()
        cost_returns = costs + self.args.cost_gamma * (1-done) * next_costs.detach()
        deltas = returns - values
        cost_deltas = cost_returns - costs
        advantages = values_pol
        cost_advantages = costs_pol
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        policy_loss = - advantages
        policy_loss = policy_loss.mean()
        value_loss = deltas.pow(2).mean() + cost_deltas.pow(2).mean()
        pre_actions_pol = action_out[-1]
        values_pre_pol = self.value(state, pre_actions_pol)[0].contiguous().view(-1, self.n_)
        correction_loss = th.maximum(th.zeros_like(values_pol), values_pre_pol - values_pol).mean()
        correction_loss += self.beta * self.cal_safe_loss(global_state, actions_pol, self.safety_filter)
        correction_loss += self.multiplier * cost_advantages.mean()
        lambda_loss = (cost_returns.detach().mean() - self.args.cost_limit) * self.multiplier
        return policy_loss, value_loss, correction_loss, lambda_loss, action_out

    def cal_safe_loss(self, state, actions, safety_filter):
        with th.no_grad():
            safe_actions = safety_filter.batch_correct(state, actions)
        return nn.MSELoss()(actions, safe_actions)

    def reset_multiplier(self):
        if self.multiplier < 0:
            with th.no_grad():
                self.multiplier = th.nn.Parameter(th.tensor(0.,device=self.device))
