import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, DepthwiseConv2D, BatchNormalization, ReLU, Conv2D, MaxPooling2D


def depthwise_separable_block(inpt, depthwise_conv_stride, pointwise_conv_output_filters):
    x = DepthwiseConv2D(kernel_size=(3,3), strides=(depthwise_conv_stride, depthwise_conv_stride), padding="same")(inpt)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(pointwise_conv_output_filters, kernel_size=(1,1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def dense_block(inpt, size, dropout_rate):
    x = BatchNormalization()(inpt)
    x = Dropout(dropout_rate)(x)
    x = Dense(size)(x)
    x = ReLU()(x)
    return x


class DepthwiseSeparableConvNet:

    def __init__(self, input_shape, hyperparameters, num_classes):
        self.input_shape = input_shape
        self.hyperparameters = hyperparameters
        input = Input(input_shape)
        x = input
        for block_params in self.hyperparameters['depthwise_separable_blocks']:
            x = depthwise_separable_block(x, depthwise_conv_stride=block_params['depthwise_conv_stride'],
                                          pointwise_conv_output_filters=block_params['pointwise_conv_output_filters'])
        x = Flatten()(x)
        for block_params in self.hyperparameters['dense_blocks']:
            x = dense_block(x, block_params['size'], block_params['dropout_rate'])
        output = Dense(num_classes, activation='softmax')(x)
        self.model = Model(inputs=input, outputs=output)

    def train(self, x_train, y_train, x_test, y_test):
        opt = RMSprop(lr=self.hyperparameters['learning_rate'], decay=self.hyperparameters['decay'])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])
        self.model.fit(x_train, y_train,
                       batch_size=self.hyperparameters['batch_size'],
                       epochs=self.hyperparameters['num_epochs'],
                       validation_data=(x_test, y_test),
                       shuffle=True)

    def evaluate(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
