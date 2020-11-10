import os
import json
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from model import build_simple_model
from utils import config_gpu

def set_tf_config(resolver, environment=None):
    """Set the TF_CONFIG env variable from the given cluster resolver"""
    cfg = {'cluster': resolver.cluster_spec().as_dict(),
           'task': {'type': resolver.get_task_info()[0], 'index': resolver.get_task_info()[1]},
           'rpc_layer': resolver.rpc_layer,
           }
    if environment:
        cfg['environment'] = environment
    os.environ['TF_CONFIG'] = json.dumps(cfg)


if __name__ == '__main__':
    resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
    set_tf_config(resolver)
    config_gpu(incremental=True,
               virtualize=True,
               mem_limit=[4096])
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    with strategy.scope():
        model = build_simple_model()
        history = model.fit(train_images, train_labels,
                            validation_data=(test_images, test_labels),
                            batch_size=256,
                            epochs=50
                            )
