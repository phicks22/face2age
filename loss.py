import torch
import torch.nn as nn
import numpy as np


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class HuboshLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        log_cosh = LogCoshLoss()
        log_cosh_loss = log_cosh(y_true, y_pred)
        
        huber = nn.SmoothL1Loss(beta=0.1)
        huber_loss = huber(y_pred, y_true)
        
        total_loss = torch.add(huber_loss, log_cosh_loss)
 
        return torch.mean(total_loss)


class RMSLE():
    def __init__(self):
        super().__init()

    def forward(self, y_pred, y_true):
        log_loss = np.log(1 + y_true) - np.log(1 + y_pred)

        return np.sqrt(np.mean(np.square(log_loss)))

