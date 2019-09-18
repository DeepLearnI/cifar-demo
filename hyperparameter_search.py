from utils import SearchSpace, sample_hyperparameters
import os
os.environ['FOUNDATIONS_COMMAND_LINE'] = 'True'
import foundations


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

