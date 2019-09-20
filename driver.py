import foundations
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model import DepthwiseSeparableConvNet
import numpy as np

foundations.set_tensorboard_logdir('train_logs')

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

# define hyperparameters
hyperparameters = foundations.load_parameters()

# train
input_shape = x_train.shape[1:]
model = DepthwiseSeparableConvNet(input_shape, hyperparameters, num_classes=10)
model.model.summary()
model.train(x_train, y_train, x_validation, y_validation)

# evaluate model
model.evaluate(x_test, y_test, verbose=1)