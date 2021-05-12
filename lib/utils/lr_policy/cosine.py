import math

from ..loggers import STDLogger as logger
from .lr_policy import LRPolicy
from ...core.config import Config as cfg


class Cosine(LRPolicy):
    """Cosine learning rate decay policy
    """

    @staticmethod
    def require_args():
        cfg.add_argument('--eta_min', default=0.0, type=float,
                         help='minimum learning rate')

    def __init__(self, base_lr, eta_min=None, t_max=None):
        self.base_lr = base_lr
        self.eta_min = eta_min if eta_min is not None else cfg.eta_min
        self.t_max = t_max if t_max is not None else cfg.max_epochs

        logger.debug('Going to use [cosine] learning policy for optimization with '
                     'base learning rate %.5f, eta_min %f and t_max %f' %
                     (self.base_lr, self.eta_min, self.t_max))

    def _update_(self, steps):
        """decay learning rate according to current step

        Arguments:
            steps {int} -- current steps

        Returns:
            float -- updated learning rate
        """

        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * steps / self.t_max))
