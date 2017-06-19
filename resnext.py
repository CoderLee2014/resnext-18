import mxnet as mx

def resnext(units, num_stages, filter_list, num_classes, num_group, image_shape, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    '''
    return resnext symbol of parameters
    -------------------------------
    units: list
        Number of units in each stage.
    num_stages: int 
        Number of stage
    filter_list: list
        Channel size of each stage
    num_classes: int 
        Output size of the symbol
    num_groupes: int 
        Number of conv groups
    dataset: str
        Dataset type, only cifar10 and imagenet supports
    workspace: int
        Workspace used in convolutional operator

    '''
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=True, name="conv0", workspace=workspace)
    else:
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7,7), stride=(2,2), pad=(3,3), no_bias=True, name='conv0',workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.sym.Pooling(data=body, kernel=(3,3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(stages):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False, name='stage%d_unit%d'%(i+1, 1), bottle_neck=bottle_neck, num_group=num_group, bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d'%(i+1, j+2), bottle_neck=bottle_neck, num_group=num_group, bn_mom=bn_mom, workspace=workspace, memonger=memonger)

    pool1 = mx.symbol.Pooling(data=body, global_pool=True, kernel=(7,7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes,  name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')


