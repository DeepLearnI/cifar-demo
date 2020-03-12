import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, DepthwiseConv2D, BatchNormalization, ReLU, Conv2D, MaxPooling2D

def depthwise_separable_block(input, depthwise_conv_stride, pointwise_conv_output_filters):
    x = DepthwiseConv2D(kernel_size=(3,3), strides=depthwise_conv_stride, padding="same")(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(pointwise_conv_output_filters, kernel_size=(1,1))(x)
    x = BatchNormalization()(x)
    output = ReLU()(x)
    return output


def dense_block(inpt, size, dropout_rate):
    x = BatchNormalization()(inpt)
    x = Dropout(dropout_rate)(x)
    x = Dense(size)(x)
    output = ReLU()(x)
    return output


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

        log_dir = 'train_logs'
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(x_train, y_train,
                       batch_size=self.hyperparameters['batch_size'],
                       epochs=self.hyperparameters['num_epochs'],
                       validation_data=(x_test, y_test),
                       shuffle=True,
                       callbacks=[tensorboard_callback])

    def evaluate(self, x_test, y_test, verbose):
        # get predictions and scores
        predictions = self.model.predict(x_test)
        scores = self.model.evaluate(x_test, y_test, verbose=verbose)

        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        # get the most and least confident images from the test set
        least_confident_prediction_index = np.argmin(np.min(predictions, axis=1))
        most_confident_prediction_index = np.argmax(np.max(predictions, axis=1))

        # Save images of the least and most confident images
        most_confident_img = Image.fromarray((x_test[most_confident_prediction_index]*255).astype('uint8'), 'RGB')
        least_confident_img = Image.fromarray((x_test[least_confident_prediction_index]*255).astype('uint8'), 'RGB')
        most_confident_img = most_confident_img.resize((256, 256))
        least_confident_img = least_confident_img.resize((256, 256))
        most_confident_img.save('data/most_confident_image.png')
        least_confident_img.save('data/least_confident_image.png')
