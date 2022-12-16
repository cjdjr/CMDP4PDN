import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        '''
            pred, targ : tensor [ batch_size x action_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        return weighted_loss

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    r"""
    Mish activation function is proposed in "Mish: A Self
    Regularized Non-Monotonic Neural Activation Function"
    paper, https://arxiv.org/abs/1908.08681.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim,
                 t_dim=16):

        super(MLP, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)

class Diffusion(nn.Module):
    def __init__(self, state_dim, args):
        super(Diffusion, self).__init__()
        self.args = args
        self.state_dim = state_dim
        self.action_dim = args.action_dim
        self.max_action = getattr(args, "max_action", 1.0)
        self.model = MLP(self.state_dim, self.action_dim, args.hid_size)
        self.model_frozen = copy.deepcopy(self.model)
        beta_schedule = getattr(args, "beta_schedule", "linear")
        self.n_timesteps = int(getattr(args, "n_timesteps", 100))
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(self.n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(self.n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.clip_denoised = getattr(args, "clip_denoised", True)
        self.predict_epsilon = getattr(args, "predict_epsilon", True)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        loss_type = getattr(args, "loss_type", "l2")
        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s, grad=True):
        if grad:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model_frozen(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s, grad=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, grad=grad)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)


            if return_diffusion: diffusion.append(x)


        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)

    def guided_sample(self, state, q_fun, start=0.2, verbose=False, return_diffusion=False):
        device = self.betas.device
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        x = torch.randn(shape, device=device)
        i_start = self.n_timesteps * start

        if return_diffusion: diffusion = [x]

        def guided_p_sample(x, t, s, fun):
            b, *_, device = *x.shape, x.device
            with torch.no_grad():
                model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
            noise = torch.randn_like(x)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

            # Involve Function Guidance
            a = model_mean.clone().requires_grad_(True)
            q_value = fun(s, a)
            # q_value = q_value / q_value.abs().mean().detach()  # normalize q
            grads = torch.autograd.grad(outputs=q_value, inputs=a, create_graph=True, only_inputs=True)[0].detach()
            return (model_mean + model_log_variance * grads) + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if i <= i_start:
                x = guided_p_sample(x, timesteps, state, q_fun)
            else:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state)


            if return_diffusion: diffusion.append(x)


        x = x.clamp_(-self.max_action, self.max_action)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1).clamp_(-self.max_action, self.max_action)
        else:
            return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.model.final_layer.weight.new(1, self.args.agent_num, self.args.hid_size).zero_()

    def forward(self, state, hidden_state, *args, **kwargs):
        mean = self.sample(state, *args, **kwargs)
        hidden = torch.zeros((mean.shape[0], self.args.hid_size), dtype=mean.dtype, device=mean.device).reshape(-1, self.args.hid_size)
        # log_stds = torch.zeros((mean.shape[0], self.args.hid_size), dtype=mean.dtype, device=mean.device)
        log_stds = None
        return mean, log_stds, hidden

    def step_frozen(self):
        for param, target_param in zip(self.model.parameters(), self.model_frozen.parameters()):
            target_param.data.copy_(param.data)

    def sample_t_middle(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        t = np.random.randint(0, int(self.n_timesteps*0.2))
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state, grad=(i == t))
        action = x
        return action.clamp_(-self.max_action, self.max_action)

    def sample_t_last(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device

        x = torch.randn(shape, device=device)
        cur_T = np.random.randint(int(self.n_timesteps * 0.8), self.n_timesteps)
        for i in reversed(range(0, cur_T)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if i != 0:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state)
            else:
                x = self.p_sample(x, timesteps, state)

        action = x
        return action.clamp_(-self.max_action, self.max_action)

    def sample_last_few(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device

        x = torch.randn(shape, device=device)
        nest_limit = 5
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if i >= nest_limit:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state)
            else:
                x = self.p_sample(x, timesteps, state)

        action = x
        return action.clamp_(-self.max_action, self.max_action)

