from torch import distributions

import autoroot # for imports from src
import src.nsf.utils as utils


class TweakedUniform(distributions.Uniform):
    def log_prob(self, value, context):
        return utils.sum_except_batch(super().log_prob(value))

    def sample(self, num_samples, context):
        return super().sample((num_samples, ))
