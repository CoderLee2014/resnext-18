__author__ = 'xli'

import mxnet as mx
import logging
import gzip, struct
import numpy as np
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def get_iterators(batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec         = './image_train.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        preprocess_threads  = 16,
        rand_crop           = True,
        rand_mirror         = True)
    val = mx.io.ImageRecordIter(
        path_imgrec         = './image_val.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False)
    return (train, val)


def get_fine_tune_resnet(symbol, arg_params, num_classes, layer_name='flatten0'):
    all_layers = symbol.get_internals()
    net = all_layers['flatten0_output']
    net = mx.sym.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.sym.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k : arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    #devs = [mx.gpu(i) for i in range(4, 8)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    model_prefix = './model/ft-resnext-18'
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    mod.fit(train, val,
            num_epoch=1000,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            epoch_end_callback=checkpoint,
            batch_end_callback=mx.callback.Speedometer(batch_size, 10),
            kvstore='device',
            optimizer='sgd',
            optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)

num_classes = 2
batch_per_gpu = 64
num_gpus = 4
sym, arg_params, aux_params = mx.model.load_checkpoint('./model/ft-resnext-18', 20)
(new_sym, new_args) = get_fine_tune_resnet(sym, arg_params, num_classes)
batch_size = batch_per_gpu * num_gpus
train, val = get_iterators(batch_size)
mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)

