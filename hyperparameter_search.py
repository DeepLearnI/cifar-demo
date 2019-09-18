import os
os.environ['FOUNDATIONS_COMMAND_LINE'] = 'True'
import foundations
import numpy as np
import copy


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


hyperparameter_ranges = {'num_epochs': 1,
                         'batch_size': 64,
                         'learning_rate': 0.0001,
                         'depthwise_separable_blocks': [{'depthwise_conv_stride': 2, 'pointwise_conv_output_filters': 6},
                                                  {'depthwise_conv_stride': 2, 'pointwise_conv_output_filters': 12}],
                         'dense_blocks': [{'size': SearchSpace(64, 256, int),
                                           'dropout_rate': SearchSpace(0.1, 0.5, float)}],
                         'decay': 1e-6}

num_jobs = 5
for _ in range(num_jobs):
    hyperparameters = sample_hyperparameters(hyperparameter_ranges)
    foundations.submit(scheduler_config='scheduler', job_dir='.', command='driver.py', params=hyperparameters, stream_job_logs=True)

