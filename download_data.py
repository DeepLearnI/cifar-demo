import tensorflow
import numpy as np

# download CIFAR10 to 'data' directory
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.cifar10.load_data()
np.save('data/x_train', x_train)
np.save('data/y_train', y_train)
np.save('data/x_test', x_test)
np.save('data/y_test', y_test)