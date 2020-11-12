import os
import json
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from model import build_simple_model
from utils import config_gpu


host_ip_map = {
    'classt1' : '192.168.212.201',
    'classt2' : '192.168.212.202',
    'classt3' : '192.168.212.203',
    'classt4' : '192.168.212.204',
    'classt5' : '192.168.212.205',
    'classt6' : '192.168.212.206'
}


def set_tf_config(resolver, environment=None):
    """Set the TF_CONFIG env variable from the given cluster resolver"""
    cluster = resolver.cluster_spec().as_dict()
    mapped = [':'.join((host_ip_map[x.split(':')[0]], x.split(':')[1])) for x in cluster['worker']]
    cluster['worker'] = mapped
    cfg = {'cluster': cluster,
           'task': {'type': resolver.get_task_info()[0], 'index': resolver.get_task_info()[1]},
           'rpc_layer': resolver.rpc_layer,
           }
    if environment:
        cfg['environment'] = environment
    os.environ['TF_CONFIG'] = json.dumps(cfg)


if __name__ == '__main__':
    resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=8080)
    set_tf_config(resolver)
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    config_gpu(incremental=True,
               virtualize=True,
               mem_limit=[4096])

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    with strategy.scope():
        model = build_simple_model()
        history = model.fit(train_images, train_labels,
                            validation_data=(test_images, test_labels),
                            batch_size=256,
                            epochs=50
                            )
