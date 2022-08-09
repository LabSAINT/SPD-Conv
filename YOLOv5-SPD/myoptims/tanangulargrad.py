import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import torch.nn as nn


class tanangulargrad(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(tanangulargrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(tanangulargrad, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'tanangulargrad does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Previous gradient
                    state['previous_grad'] = torch.zeros_like(p.data)
                    # temporary minimum value for comparison
                    state['min'] = torch.zeros_like(p.data)
                    # temporary difference between gradients for comparison
                    state['diff'] = torch.zeros_like(p.data)
                    # final tan value to be used
                    state['final_tan_theta'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, previous_grad, min, diff, final_tan_theta = state['exp_avg'], state['exp_avg_sq'], \
                                                                                 state['previous_grad'], state['min'], \
                                                                                 state['diff'], state['final_tan_theta']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                tan_theta = abs((previous_grad - grad) / (1 + previous_grad * grad))

                angle = torch.atan(tan_theta) * (180 / 3.141592653589793238)
                ans = torch.gt(angle, min)
                ans1, count = torch.unique(ans, return_counts=True)

                try:
                    if (count[1] < count[0]):
                        min = angle
                        diff = abs(previous_grad - grad)
                        final_tan_theta = tan_theta.clone()
                except:
                    if (ans1[0].item() == False):
                        min = angle
                        diff = abs(previous_grad - grad)
                        final_tan_theta = tan_theta.clone()

                angular_coeff = torch.tanh(abs(final_tan_theta)) * 0.5 +0.5    # Calculating Angular coefficient

                state['previous_grad'] = grad.clone()
                state['min'] = min.clone()
                state['diff'] = diff.clone()
                state['final_tan_theta'] = final_tan_theta.clone()

                # update momentum with angular_coeff
                exp_avg1 = exp_avg * angular_coeff

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg1, denom)

        return loss
