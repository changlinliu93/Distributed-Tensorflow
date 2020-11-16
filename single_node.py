import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from model import build_simple_model, DenseModel, ResModel
from utils import config_gpu, preprocess_input
# tf.debugging.set_log_device_placement(True)

if __name__ == '__main__':
    config_gpu(incremental=True,
               virtualize=True,
               mem_limit=[4096])

    [train_images, train_labels], [test_images, test_labels] = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    train_images = preprocess_input(train_images)
    test_images = preprocess_input(test_images)

    generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                             width_shift_range=5. / 32,
                                                             height_shift_range=5. / 32,
                                                             horizontal_flip=True)

    generator.fit(train_images, seed=0)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = ResModel(10)
        model.compile(optimizer= keras.optimizers.Adam(1e-3),
                      loss= "categorical_crossentropy",
                      metrics=['accuracy'])

    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath='record/dense3.{epoch:02d}-{val_loss:.2f}.h5',
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True)

    my_callbacks = [
        # model_checkpoint_callback,
        # tf.keras.callbacks.CSVLogger('record/dense3_sn2g.csv', separator=",", append=True)
    ]

    batch_size = 64
    nb_epoch = 150

    history = model.fit(generator.flow(train_images, train_labels, batch_size=batch_size),
                        steps_per_epoch=len(train_images) // batch_size, epochs=nb_epoch,
                        callbacks=my_callbacks,
                        validation_data=(test_images, test_labels), verbose=1)

