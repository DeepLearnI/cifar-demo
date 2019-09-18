import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from model import DepthwiseSeparableConvNet
from datagen import datagen
import os

# load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5)
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
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
model.train(x_train, y_train, x_test, y_test)

# evaluate
scores = model.model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
