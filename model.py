import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers

L2_WEIGHT_DECAY = 0.0001
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 0.00001


def build_simple_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def DenseLayer(x, nb_filter, bn_size=4, alpha=0.3, drop_rate=0.2):
    # Bottleneck layers
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                            kernel_initializer='he_normal')(x)

    # Composite function
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same', use_bias=False,
                            kernel_initializer='he_normal')(x)

    if drop_rate: x = keras.layers.Dropout(drop_rate)(x)

    return x


def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = keras.layers.concatenate([x, conv], axis=3)

    return x


def TransitionLayer(x, compression=0.5, alpha=0.3, is_max=0):
    nb_filter = int(x.shape.as_list()[-1] * compression)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                            kernel_initializer='he_normal')(x)
    if is_max != 0:
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    else:
        x = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    return x


def DenseModel(num_classes):
    input_data = keras.layers.Input(shape=(32, 32, 3))

    x = input_data
    bn_axis = 3

    x = keras.layers.Conv2D(24, 3, strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = keras.layers.ReLU()(x)

    # x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = DenseBlock(x, 12, 12, 0)
    x = TransitionLayer(x, 1, 0.0, 1)
    x = DenseBlock(x, 12, 12, 0)
    x = TransitionLayer(x, 1, 0.0, 1)
    x = DenseBlock(x, 12, 12, 0)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.1e-5)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create model.
    return keras.models.Model([input_data], x, name='denseModel')


def identity_block(input_tensor, kernel_size, filters):
    filters1, filters2, filters3 = filters

    x = keras.layers.Conv2D(filters1, (1, 1), use_bias=False,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(input_tensor)

    x = keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(x)

    x = keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)

    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(x)

    x = keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)

    x = keras.layers.add([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, strides=(1, 1)):
    filters1, filters2, filters3 = filters

    x = keras.layers.Conv2D(filters1, (1, 1), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(input_tensor)
    x = keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, strides=strides, padding='same', use_bias=False,
                            kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(x)
    x = keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(x)
    x = keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)

    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(input_tensor)
    shortcut = keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def conv_block_AP(input_tensor, kernel_size, filters, strides=(1, 1, 1)):
    filters1, filters2, filters3 = filters

    x = keras.layers.Conv2D(filters1, (1, 1), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(input_tensor)
    x = keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, strides=strides, padding='same', use_bias=False,
                            kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(
        x)
    x = keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(x)
    x = keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)

    shortcut = keras.layers.AveragePooling2D((2, 2), strides=strides, padding='same')(input_tensor)
    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal',
                                   kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(shortcut)
    shortcut = keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def ResModel(num_classes):
    input_data = keras.layers.Input(shape=(32, 32, 3))

    x = input_data
    bn_axis = 3

    x = keras.layers.Conv2D(32, 5,
                            strides=(2, 2),
                            padding='valid', use_bias=False,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Dropout(0.2)(x)
    x = conv_block_AP(x, 3, [32, 32, 32], (1, 1))
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = conv_block_AP(x, 3, [64, 64, 64], (1, 1))
    x = conv_block_AP(x, 3, [64, 64, 64], (1, 1))
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = conv_block_AP(x, 3, [128, 128, 128], (1, 1))
    x = conv_block_AP(x, 3, [128, 128, 128], (1, 1))

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(
        num_classes, activation='softmax',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        bias_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY))(x)

    # Create model.
    return keras.models.Model([input_data], x, name='ResModel')