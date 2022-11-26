import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import keras
import collections
import tensorflow as tf
from keras.layers import BatchNormalization, Conv2D, ReLU, Add, Dense,\
     MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Input, ZeroPadding2D, Activation
from keras.utils import layer_utils, get_file
import keras.backend as K

# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

def get_bn_params(**params):
    axis = 3 # if K.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params

def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params

# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------

def ResNet_conv_block(x, filters, kernel_size, name, num, strides=1):
    if kernel_size==3:
        x = ZeroPadding2D(padding=(1,1))(x)
    elif kernel_size==7:
        x = ZeroPadding2D(padding=(3,3))(x)
    
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, name=name+f"_conv{num}", **get_conv_params())(x)
    
    if name=="residual_block_0":
        bn_params = get_bn_params(scale=False)
    else:
        bn_params = get_bn_params()

    x = BatchNormalization(name=name+f"_bn{num}", **bn_params)(x)
    x = ReLU(name=name+f"_relu{num}")(x)
    return x


def ResNet_Bottleneck_Blocks(input, type, filters, strides, name):
    
    def ResNet_bottleneck_IdentityBlock(input, filters, name):

        x = ResNet_conv_block(input, filters=filters,
                            kernel_size=1, strides=1, name=name, num=1)
        x = ResNet_conv_block(x, filters=filters, kernel_size=3,
                            strides=1, name=name, num=2)
        
        x = Conv2D(filters=4*filters, kernel_size=1,
                strides=1, name=name+"_conv3")(x)
        x = BatchNormalization(name=name+"_bn3", **get_bn_params())(x)

        x = Add(name=name+"_Add")([input, x])
        x = ReLU(name=name+"_relu3")(x)
        return x

    def ResNet_bottleneck_ProjectionBlock(input, filters, strides, name):

        # left stream
        x = ResNet_conv_block(input, filters=filters,
                            kernel_size=1, strides=strides, name=name, num=1)
        x = ResNet_conv_block(x, filters=filters, kernel_size=3,
                            strides=1, name=name, num=2)
        
        x = Conv2D(filters=4*filters, kernel_size=1,
                strides=strides, name=name+"_conv3")(x)
        x = BatchNormalization(name=name+"_bn3", **get_bn_params())(x)

        # right stream
        proj_short = Conv2D(filters=4*filters, kernel_size=1,
                strides=strides, name=name+"_conv_right")(input)
        proj_short = BatchNormalization(name=name+"_bn_right", **get_bn_params())(proj_short)

        x = Add(name=name+"_Add")([proj_short, x])
        x = ReLU(name=name+"_relu3")(x)
        return x

    if type=="identity":
        return ResNet_bottleneck_IdentityBlock(input, filters, name)
    elif type=="projection":
        return ResNet_bottleneck_ProjectionBlock(input, filters, strides, name)
    else:
        raise ValueError('type not in ["identity", "projection"]')


def ResNet_Blocks(input, type, filters, strides, name):
    
    def ResNet_IdentityBlock(input, filters, name):

        x = ResNet_conv_block(input, filters=filters,
                            kernel_size=3, strides=1, name=name, num=1)
        x = ResNet_conv_block(x, filters=filters, kernel_size=3,
                            strides=1, name=name, num=2)

        x = Add(name=name+"_Add")([input, x])
        x = ReLU(name=name+"_relu3")(x)
        return x

    def ResNet_ProjectionBlock(input, filters, strides, name):

        # left stream
        x = ResNet_conv_block(input, filters=filters,
                            kernel_size=1, strides=strides, name=name, num=1)
        x = ZeroPadding2D(padding=(1,1))(x)
        x = Conv2D(filters=filters, kernel_size=3,
                    strides=1, padding="valid", name=name+f"_conv2")(x)
        x = BatchNormalization(name=name+f"_bn2", **get_bn_params())(x)

        # right stream
        proj_short = Conv2D(filters=filters, kernel_size=1,
                strides=strides, name=name+"_conv_right")(input)
        proj_short = BatchNormalization(name=name+"_bn_right", **get_bn_params())(proj_short)

        x = Add(name=name+"_Add")([proj_short, x])
        x = ReLU(name=name+"_relu3")(x)
        return x

    if type=="identity":
        return ResNet_IdentityBlock(input, filters, name)
    elif type=="projection":
        return ResNet_ProjectionBlock(input, filters, strides, name)
    else:
        raise ValueError('type not in ["identity", "projection"]')



def load_resnet_block(resnet_block, input, num_blocks, filters, stage):

    # first block of first stage without strides because we have maxpooling before
    if stage==0:
        strides = 1
    else:
        strides = 2

    name = f"residual_block_{stage+1}"
    x = resnet_block(input=input, type="projection", filters=filters, strides=strides, name=name+"_1")
    for i in range(num_blocks-1):
        x = resnet_block(input=x, type="identity", filters=filters, strides=1, name=name+f"_{i+2}")
    return x
    


def ResNet(model_params, input_shape=None, input_tensor=None, include_top=True,
           classes=1000, weights=None, **kwargs):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Args:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='Input_Layer')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # choose residual block type
    ResidualBlock = model_params.residual_block

    # get parameters for model layers
    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64
    # resnet bottom
    # x = BatchNormalization(name='bn_data', **get_bn_params(scale=False))(img_input)
    # x = ZeroPadding2D(padding=(3, 3))(img_input)
    # x = Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    # x = BatchNormalization(name='bn0', **get_bn_params())(x)
    # x = Activation('relu', name='relu0')(x)
    # x = ZeroPadding2D(padding=(1, 1))(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnet bottom
    x = ResNet_conv_block(x=img_input, filters=init_filters, kernel_size=7, name="residual_block_0", num="", strides=2)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='max_pooling0')(x)

    # resnet body
    for stage, num_blocks in enumerate(model_params.block_design):

        filters = init_filters * (2 ** stage)
        print(stage, filters)
        x = load_resnet_block(resnet_block=ResidualBlock, input=x, num_blocks=num_blocks, filters=filters, stage=stage)

    # x = BatchNormalization(name='bn1', **bn_params)(x)
    # x = Activation('relu', name='relu1')(x)

    # resnet top
    if include_top:
        x = GlobalAveragePooling2D(name='glob_avg_pool1')(x)
        x = Dense(units=classes, activation="softmax", name='fc1')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = tf.keras.models.Model(inputs, x)

    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, model_params.model_name,
                               weights, classes, include_top, **kwargs)

    return model


ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'block_design', 'residual_block']
)

MODELS_PARAMS = {
    'resnet18': ModelParams('resnet18', (2, 2, 2, 2), ResNet_Blocks),
    'resnet34': ModelParams('resnet34', (3, 4, 6, 3), ResNet_Blocks),
    'resnet50': ModelParams('resnet50', (3, 4, 6, 3), ResNet_Bottleneck_Blocks),
    'resnet101': ModelParams('resnet101', (3, 4, 23, 3), ResNet_Bottleneck_Blocks),
    'resnet152': ModelParams('resnet152', (3, 8, 36, 3), ResNet_Bottleneck_Blocks),
}

def ResNet18(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNet(
        MODELS_PARAMS['resnet18'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def ResNet34(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNet(
        MODELS_PARAMS['resnet34'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def ResNet50(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNet(
        MODELS_PARAMS['resnet50'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def ResNet101(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNet(
        MODELS_PARAMS['resnet101'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def ResNet152(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNet(
        MODELS_PARAMS['resnet152'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def _find_weights(model_name, dataset, include_top):
    w = list(filter(lambda x: x['model'] == model_name, WEIGHTS_COLLECTION))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))
    return w


def load_model_weights(model, model_name, dataset, classes, include_top, **kwargs):

    weights = _find_weights(model_name, dataset, include_top)

    if weights:
        weights = weights[0]

        if include_top and weights['classes'] != classes:
            raise ValueError('If using `weights` and `include_top`'
                             ' as true, `classes` should be {}'.format(weights['classes']))

        weights_path = get_file(
            weights['name'],
            weights['url'],
            cache_subdir='models',
            md5_hash=weights['md5']
        )

        model.load_weights(weights_path)

    else:
        raise ValueError('There is no weights for such configuration: ' +
                         'model = {}, dataset = {}, '.format(model.name, dataset) +
                         'classes = {}, include_top = {}.'.format(classes, include_top))


inp_shape = (224,224,3)
model = ResNet18(input_shape=inp_shape, input_tensor=None, classes=2, include_top=False)
model.summary()

# WEIGHTS_COLLECTION = [

#     # ResNet18
#     {
#         'model': 'resnet18',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': True,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000.h5',
#         'name': 'resnet18_imagenet_1000.h5',
#         'md5': '64da73012bb70e16c901316c201d9803',
#     },

#     {
#         'model': 'resnet18',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': False,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5',
#         'name': 'resnet18_imagenet_1000_no_top.h5',
#         'md5': '318e3ac0cd98d51e917526c9f62f0b50',
#     },

#     # ResNet34
#     {
#         'model': 'resnet34',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': True,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5',
#         'name': 'resnet34_imagenet_1000.h5',
#         'md5': '2ac8277412f65e5d047f255bcbd10383',
#     },

#     {
#         'model': 'resnet34',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': False,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
#         'name': 'resnet34_imagenet_1000_no_top.h5',
#         'md5': '8caaa0ad39d927cb8ba5385bf945d582',
#     },

#     # ResNet50
#     {
#         'model': 'resnet50',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': True,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_1000.h5',
#         'name': 'resnet50_imagenet_1000.h5',
#         'md5': 'd0feba4fc650e68ac8c19166ee1ba87f',
#     },

#     {
#         'model': 'resnet50',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': False,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_1000_no_top.h5',
#         'name': 'resnet50_imagenet_1000_no_top.h5',
#         'md5': 'db3b217156506944570ac220086f09b6',
#     },

#     {
#         'model': 'resnet50',
#         'dataset': 'imagenet11k-places365ch',
#         'classes': 11586,
#         'include_top': True,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet11k-places365ch_11586.h5',
#         'name': 'resnet50_imagenet11k-places365ch_11586.h5',
#         'md5': 'bb8963db145bc9906452b3d9c9917275',
#     },

#     {
#         'model': 'resnet50',
#         'dataset': 'imagenet11k-places365ch',
#         'classes': 11586,
#         'include_top': False,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet11k-places365ch_11586_no_top.h5',
#         'name': 'resnet50_imagenet11k-places365ch_11586_no_top.h5',
#         'md5': 'd8bf4e7ea082d9d43e37644da217324a',
#     },

#     # ResNet101
#     {
#         'model': 'resnet101',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': True,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet101_imagenet_1000.h5',
#         'name': 'resnet101_imagenet_1000.h5',
#         'md5': '9489ed2d5d0037538134c880167622ad',
#     },

#     {
#         'model': 'resnet101',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': False,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet101_imagenet_1000_no_top.h5',
#         'name': 'resnet101_imagenet_1000_no_top.h5',
#         'md5': '1016e7663980d5597a4e224d915c342d',
#     },

#     # ResNet152
#     {
#         'model': 'resnet152',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': True,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet_1000.h5',
#         'name': 'resnet152_imagenet_1000.h5',
#         'md5': '1efffbcc0708fb0d46a9d096ae14f905',
#     },

#     {
#         'model': 'resnet152',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': False,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet_1000_no_top.h5',
#         'name': 'resnet152_imagenet_1000_no_top.h5',
#         'md5': '5867b94098df4640918941115db93734',
#     },

#     {
#         'model': 'resnet152',
#         'dataset': 'imagenet11k',
#         'classes': 11221,
#         'include_top': True,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet11k_11221.h5',
#         'name': 'resnet152_imagenet11k_11221.h5',
#         'md5': '24791790f6ef32f274430ce4a2ffee5d',
#     },

#     {
#         'model': 'resnet152',
#         'dataset': 'imagenet11k',
#         'classes': 11221,
#         'include_top': False,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet11k_11221_no_top.h5',
#         'name': 'resnet152_imagenet11k_11221_no_top.h5',
#         'md5': '25ab66dec217cb774a27d0f3659cafb3',
#     }
# ]
