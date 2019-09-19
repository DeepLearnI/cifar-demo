from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model import DepthwiseSeparableConvNet
import numpy as np

# load saved data
x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')

# train test split
x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5)

# preprocessing
y_validation = to_categorical(y_validation, num_classes=10)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
x_validation = x_validation.astype('float32')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_validation /= 255
x_train /= 255
x_test /= 255

# define training hyperparameters
hyperparameters = {'num_epochs': 1,
                   'batch_size': 64,
                   'learning_rate': 0.0001,
                   'depthwise_separable_blocks': [{'depthwise_conv_stride': (2,2), 'pointwise_conv_output_filters': 6},
                                                  {'depthwise_conv_stride': (2,2), 'pointwise_conv_output_filters': 12}],
                   'dense_blocks': [{'size': 128, 'dropout_rate': 0.1}],
                   'decay': 1e-6}

# train
input_shape = x_train.shape[1:]
model = DepthwiseSeparableConvNet(input_shape, hyperparameters, num_classes=10)
model.model.summary()
model.train(x_train, y_train, x_validation, y_validation)

# evaluate
model.evaluate(x_test, y_test, verbose=1)

