import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from model import build_simple_model
from utils import config_gpu
# tf.debugging.set_log_device_placement(True)

if __name__ == '__main__':
    config_gpu(incremental=True,
               virtualize=True,
               mem_limit=[4096])

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_simple_model()
        history = model.fit(train_images, train_labels,
                            validation_data=(test_images, test_labels),
                            batch_size=256,
                            epochs=50
                            )
