##                                         YoloV4 implemented via TF 2.x

### What is YoloV4 ?

 It's an one stage object detection algorithm and implemented based on YoloV3. It improved MAP to 44 while still keeps FPS same as YoloV3. Below image depicting the performance of Yolov4 compared with other algorithms.

 ![yolov4_performance](https://github.com/Qucy/yolo4-tf-27/blob/master/yolov4_performance.JPG)

To summary the change in YoloV4 which I thought is quite important and effective is as below

- Backbone network upgrade from DarkNet53 => CSPDarkNet53

- Using SPP and PAN in the neck network for generate enhanced feature

- Training tricks: Mosaic data argumentation, Label smoothing, CIOU and cosine annealing learning rate
- Activation function change to Mish

 So let's take a look these items one by one to see details of these changes



### 1. Backbone network update

There are 2 main updates in the backbone network, DarkNet53 => CSPDarkNet53 and activation function changed to Mish

#### 1.1 DarkNet53 => CSPDarkNet53

If you familiar with YoloV3, the backbone network is consist of multiple residual blocks with down sampling layers. In YoloV4 the residual block is replaced by CSPNet. CSPNet can be divided into 2 parts, the left part is still original input and the right part is consist of multiple residual blocks. Left part and right part will concat at final layer. Below image depicting the different networks.

![CSPDarknet](https://github.com/Qucy/yolo4-tf-27/blob/master/CSPDarknet.jpg)

#### 1.2 Activation function from LeakyReLU to Mish

The formula for Mish function and plot is as below, it is implemented in Tensorflow Addons, so we need to run command ```pip install tensorflow-addons ``` to install this package first.
$$
Mish = x \times tanh(ln(1 + e^x))
$$
![mish_activation_function](https://github.com/Qucy/yolo4-tf-27/blob/master/mish_activation_function.jpg)

The source for CSPDarkNet53 is as below

```python
from functools import wraps

import tensorflow_addons as tfa
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D, Concatenate, ZeroPadding2D, Layer)
from tensorflow.keras.regularizers import l2
from utils.utils import compose


class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return tfa.activations.mish(inputs)

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """
    Single DarkNetConv2D Block
    :param args:
    :param kwargs:
    :return: DarknetConv2D with specified parameters
    """
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Mish(*args, **kwargs):
    """
    DarknetConv2D + BatchNormalization + Mish
    :param args:
    :param kwargs:
    :return: A DarkNetConv2D block with DarknetConv2D + BatchNormalization + Mish
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    """
    Residual body construct function
    :param x: input features
    :param num_filters: number of filters
    :param num_blocks:  number of residual blocks
    :param all_narrow: all narrow or not
    :return:
    """
    # Use ZeroPadding2D and a Conv with strides of 2 to reduce image height and width
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(preconv1)

    # Create a shortcut
    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)


    # Create residual block by loop num_blocks
    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(mainconv)
        mainconv = Add()([mainconv,y])
    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(mainconv)

    # Concat shortcut with final outputs from residual blocks
    route = Concatenate()([postconv, shortconv])

    # Use Conv 1x1 to adjust channels
    return DarknetConv2D_BN_Mish(num_filters, (1,1))(route)

#---------------------------------------------------#
#   darknet53 backbone network
#   inputs image size = 416x416x3
#   outputs 3 feature maps
#---------------------------------------------------#
def darknet_body(x):
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x
    x = resblock_body(x, 512, 8)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1,feat2,feat3
```
