from collections import defaultdict
from itertools import chain
from paddle.optimizer import Optimizer
import paddle
import warnings


class Lookahead:
    def __init__(self, model, k=5, alpha=0.5):
        self.model = model
        self.k = k
        self.alpha = alpha
        self.counter = 0

        self.slow_params = {n: p.clone() for n, p in self.model.named_parameters()}

    def step(self):
        self.counter += 1
        if self.counter < self.k: return

        self.counter = 0
        with paddle.no_grad():
            for n, p in self.model.named_parameters():
                delta = self.alpha * (p - self.slow_params[n])

                p.stop_gradient = True
                p.subtract_(delta)
                p.stop_gradient = False

            self.slow_params = {n: p.clone() for n, p in self.model.named_parameters()}

