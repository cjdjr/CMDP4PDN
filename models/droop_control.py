import torch as th
import numpy as np
from models.model import Model
from utilities.util import prep_obs, translate_action
from .safety_filter import Droop_control

class DroopControlAgent(Model):

    def __init__(self, args, target_net=None):
        super(DroopControlAgent, self).__init__(args)
        self.args = args
        assert(isinstance(self.safety_filter, Droop_control))
        self.policy_dicts = th.nn.ModuleList([th.nn.Linear(1,1)])
        self.value_dicts = th.nn.ModuleList([th.nn.Linear(1,1)])

    def train_process(self, stat, trainer):
        trainer.episodes += 1

    def translate_action_env2nn(self, q):
        '''
        [-action_scale, action_scale]  ->   [-1,1]
        '''
        low = self.args.action_bias - self.args.action_scale
        high = self.args.action_bias + self.args.action_scale
        return th.from_numpy((q - low) / (high-low) * 2 - 1)

    def evaluation(self, stat, trainer):
        num_eval_episodes = self.args.num_eval_episodes
        stat_test = {}
        with th.no_grad():
            for _ in range(num_eval_episodes):
                stat_test_epi = {'mean_test_reward': 0}
                state, global_state = trainer.env.reset()
                last_q = self.translate_action_env2nn(trainer.env.last_q)
                for t in range(self.args.max_steps):
                    state_ = prep_obs(state).to(self.device).contiguous().view(1, self.n_, self.obs_dim)
                    action, _, _, _ = self.safety_filter.correct(trainer.env.get_state(), last_q)
                    _, actual = translate_action(self.args, action, trainer.env)
                    reward, done, info = trainer.env.step(actual)
                    done_ = done or t==self.args.max_steps-1
                    next_state = trainer.env.get_obs()
                    if isinstance(done, list): done = np.sum(done)
                    for k, v in info.items():
                        if 'mean_test_' + k not in stat_test_epi.keys():
                            stat_test_epi['mean_test_' + k] = v
                        else:
                            stat_test_epi['mean_test_' + k] += v
                    stat_test_epi['mean_test_reward'] += reward
                    if done_:
                        break
                    # set the next state
                    state = next_state
                    # set the next last_q
                    last_q = self.translate_action_env2nn(trainer.env.last_q)
                for k, v in stat_test_epi.items():
                    stat_test_epi[k] = v / float(t+1)
                for k, v in stat_test_epi.items():
                    if k not in stat_test.keys():
                        stat_test[k] = v
                    else:
                        stat_test[k] += v
        for k, v in stat_test.items():
            stat_test[k] = v / float(num_eval_episodes)
        stat.update(stat_test)
