import copy
import numpy as np

class SearchSpace:

    def __init__(self, min, max, type):
        self.min = min
        self.max = max
        self.type = type

    def sample(self):
        if self.type == int:
            return np.random.randint(self.min, self.max)
        elif self.type == float:
            return round(np.random.uniform(self.min, self.max), 2)


def sample_hyperparameters(hyperparameter_ranges):
    hyperparameters = copy.deepcopy(hyperparameter_ranges)
    for hparam in hyperparameter_ranges:
        if isinstance(hyperparameter_ranges[hparam], SearchSpace):
            search_space = hyperparameter_ranges[hparam]
            hyperparameters[hparam] = search_space.sample()
        elif isinstance(hyperparameter_ranges[hparam], list):
            for i, block in enumerate(hyperparameter_ranges[hparam]):
                for block_hparam in block:
                    if isinstance(block[block_hparam], SearchSpace):
                        search_space = block[block_hparam]
                        hyperparameters[hparam][i][block_hparam] = search_space.sample()
    return hyperparameters
