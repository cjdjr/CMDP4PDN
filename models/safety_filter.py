import torch
import numpy as np
import copy
class Safety_filter(object):
    def __init__(self, args):
        super(Safety_filter, self).__init__()
        self.args = args

    def correct(self, **kwargs):
        raise NotImplementedError()

    def translate_action(self, q):
        q = np.clip(q, -1.0, 1.0)
        low = self.args.action_bias - self.args.action_scale
        high = self.args.action_bias + self.args.action_scale
        return 0.5 * (q + 1.0) * (high - low) + low

class None_filter(Safety_filter):
    def __init__(self, args):
        super(None_filter, self).__init__(args)
    def correct(self, env, q):
        return q, 0, 1, 0

class Droop_control(Safety_filter):
    def __init__(self, args, pred_network = None):
        super(Droop_control, self).__init__(args)
        self.v_ref = 1.0
        self.va = 0.95
        self.vd = 1.05
        self.vb = 1.0
        self.vc = 1.0
        self.q_max = 1.0
        self.max_iter = getattr(args, "max_iter", 10)
        self.gain = getattr(args, "gain", 0.1)
        self.penalty_coff = getattr(args, "penalty_coff", 0.01)
        self.pred_network = pred_network
        self.pv_index = args.pv_index

    def one_step_droop_control(self, v):
        if v <= self.va:
            q = self.q_max
        elif v > self.vd:
            q = -self.q_max
        elif v >= self.vb and v <= self.vc:
            q = 0
        elif v < self.vb:
            droop_k = (self.q_max - 0) / (self.va - self.vb)
            q = droop_k * (v - self.vb)
        elif v > self.vc:
            droop_k = (0 - self.q_max) / (self.vc - self.vd)
            q = droop_k * (self.vc - v)
        return q

    def imagine_predict(self, state, actions):
        device = next(self.pred_network.parameters()).device
        input = np.concatenate((state, actions))
        input = torch.from_numpy(input).to(torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            self.pred_network.eval()
            pred_voltage = self.pred_network(input).squeeze(0)
        pred_voltage = pred_voltage.detach().cpu().numpy()
        pred_voltage_pv_bus = copy.deepcopy(pred_voltage[self.pv_index])
        return pred_voltage, pred_voltage_pv_bus, True

    def correct(self, state, q):
        q_last_last = q.detach().squeeze().cpu().numpy()
        q_last = q.detach().squeeze().cpu().numpy()
        last_voltage_pv_bus = None
        count = 0
        safe = 0
        for i in range(self.max_iter):
            if self.pred_network is None:
                raise NotImplementedError()
                voltage, voltage_pv_bus, solvable = env.predict(self.translate_action(q_last))
            else:
                voltage, voltage_pv_bus, solvable = self.imagine_predict(state, self.translate_action(q_last))
            # crash
            if not solvable:
                q_last = q_last_last
                break
            # safe action
            if np.all(voltage >= self.va) and np.all(voltage <= self.vd):
                safe = 1
                break
            # convergence
            if last_voltage_pv_bus is not None and np.sqrt(((last_voltage_pv_bus - voltage_pv_bus)**2).mean())<=1e-3:
                break

            last_voltage_pv_bus = voltage_pv_bus
            q_new = np.array([self.one_step_droop_control(v) for v in voltage_pv_bus])
            q_last_last = q_last
            q_last = (1-self.gain) * q_last + self.gain * q_new
            count += 1

        filter_penalty = self.penalty_coff * count
        # if count > 0:
        #     filter_penalty += 2 * self.penalty_coff

        return torch.tensor(q_last, dtype=q.dtype, device=q.device).view(q.shape), count, safe, filter_penalty

    def batch_correct(self, state, q):
        batch_size = state.shape[0]
        res = []
        for i in range(batch_size):
            safe_action, _, _, _, = self.correct(state[i].detach().cpu().numpy(), q[i])
            res.append(safe_action)
        return torch.stack(res, dim=0)

class Droop_control_ind(Droop_control):
    def __init__(self, args, pred_network = None):
        super(Droop_control_ind, self).__init__(args, pred_network)

    def correct(self, env, q):
        q_last_last = q.detach().squeeze().cpu().numpy()
        q_last = q.detach().squeeze().cpu().numpy()
        last_voltage_pv_bus = None
        count = 0
        safe = 0
        for i in range(self.max_iter):
            if self.pred_network is None:
                voltage, voltage_pv_bus, solvable = env.predict(self.translate_action(q_last))
            else:
                voltage, voltage_pv_bus, solvable = self.imagine_predict(env, self.translate_action(q_last))
            # crash
            if not solvable:
                q_last = q_last_last
                break
            # safe action
            if np.all(voltage >= self.va) and np.all(voltage <= self.vd):
                safe = 1
                break
            # convergence
            if last_voltage_pv_bus is not None and np.sqrt(((last_voltage_pv_bus - voltage_pv_bus)**2).mean())<=1e-3:
                break

            last_voltage_pv_bus = voltage_pv_bus
            q_new = np.array([self.one_step_droop_control(v) for v in voltage_pv_bus])
            q_last_last = q_last
            q_last = (1-self.gain) * q_last + self.gain * q_new
            count += 1

        filter_penalty = (q_last - q.detach().squeeze().cpu().numpy())**2
        filter_penalty += self.penalty_coff * count
        # if count > 0:
        #     filter_penalty += 2 * self.penalty_coff

        return torch.tensor(q_last, dtype=q.dtype, device=q.device).view(q.shape), count, safe, filter_penalty