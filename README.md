##                                         YoloV4 implemented via TF 2.7

### What is YoloV4 ?

 It's an one stage object detection algorithm and implemented based on YoloV3. It improved MAP to 44 while still keeps FPS same as YoloV3. Below image depicting the performance of Yolov4 compared with other algorithms.

 ![yolov4_performance](https://github.com/Qucy/yolo4-tf-27/blob/master/img/yolov4_performance.JPG)

To summary the change in YoloV4 which I thought is quite important and effective is as below

- Backbone network upgrade from DarkNet53 => CSPDarkNet53

- Using SPP and PAN in the neck network for generating enhanced feature

- Training tricks: Mosaic data argumentation, Label smoothing, CIOU and cosine annealing learning rate
- Activation function change to Mish

 So let's take a look these items one by one to see details of these changes



### 1. Backbone network update

There are 2 main updates in the backbone network, DarkNet53 => CSPDarkNet53 and activation function changed to Mish

#### 1.1 DarkNet53 => CSPDarkNet53

If you familiar with YoloV3, the backbone network is consist of multiple residual blocks with down sampling layers. In YoloV4 the residual block is replaced by CSPNet. CSPNet can be divided into 2 parts, the left part is still original input and the right part is consist of multiple residual blocks. Left part and right part will concat at final layer. Below image depicting the different networks.

![CSPDarknet](https://github.com/Qucy/yolo4-tf-27/blob/master/img/CSPDarknet.jpg)

#### 1.2 Activation function from LeakyReLU to Mish

The formula for Mish function and plot is as below, it is implemented in Tensorflow Addons, so we need to run command ```pip install tensorflow-addons ``` to install this package first.
```math
Mish = x * tanh(ln(1 + e^x))
```
![mish_activation_function](https://github.com/Qucy/yolo4-tf-27/blob/master/img/mish_activation_function.jpg)

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


    # Create residual blocks by loop num_blocks
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


def darknet_body(x):
    """
    darknet53 backbone network
    :param x: input image with shape 416x416x3
    :return: outputs 3 feature maps
    """
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

#### 1.2 FPN - neck network upgrade

FPN's network structure is as below if input image shape is 416 * 416. The main change compared to YoloV3 is:

- using SPP network in the bottom layer
- using PANet in the up sampling layers

![fpn_neck](https://github.com/Qucy/yolo4-tf-27/blob/master/img/fpn_neck.jpg)

##### 1.2.1 SPP

SPP structure append at the bottom of CSPDarknet53, after the feature maps from CSPDarknet53, it will go through a convolution block by 3 times. And after that it will go through a 5x5, a 9x9,  a 13x13 and a 1x1 max pooling layer. It will help increase the receptive filed and provide more information.

```python
maxpool1 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(P5)
maxpool2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(P5)
maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(P5)
P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
```

##### 1.2.2 PANet

PANet was first used in instance segmentation algorithm in around 2018. Below image is the original PANet structure, the most important part is it will extract features from bottom up and top down for multiple times. In the image part (a) is the traditional FPN and in part (b) it will do up sampling again.

![PANet](https://github.com/Qucy/yolo4-tf-27/blob/master/img/PANet.jpg)

In YoloV4 the PANet is using for the 3 layer as below

![fpn_panet](https://github.com/Qucy/yolo4-tf-27/blob/master/img/fpn_panet.jpg)

Source code is as below

```python
def yolo_body(input_shape, anchors_mask, num_classes):
    """
    YoloV4 FPN and head network building
    :param input_shape: image input shape
    :param anchors_mask: anchor masks
    :param num_classes: number of category classes
    :return: Model
    """
    inputs = Input(input_shape)
    # ---------------------------------------------------#
    #   Create CSPdarknet53 backbone and return 3 feature maps
    #   feat1 shape => 52,52,256
    #   feat2 shape => 26,26,512
    #   feat3 shape => 13,13,1024
    # ---------------------------------------------------#
    feat1, feat2, feat3 = darknet_body(inputs)

    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)
    P5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
    # use SPP structure here
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(P5)
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
    P5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)

    # 13,13,512 -> 13,13,256 -> 26,26,256
    P5_upsample = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(P5)
    # 26,26,512 -> 26,26,256
    P4 = DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P4, P5_upsample])

    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4, 256)

    # 26,26,256 -> 26,26,128 -> 52,52,128
    P4_upsample = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(P4)
    # 52,52,256 -> 52,52,128
    P3 = DarknetConv2D_BN_Leaky(128, (1, 1))(feat1)
    # 52,52,128 + 52,52,128 -> 52,52,256
    P3 = Concatenate()([P3, P4_upsample])

    # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    P3 = make_five_convs(P3, 128)

    # ---------------------------------------------------#
    #   y3 = (batch_size,52,52,3,85)
    # ---------------------------------------------------#
    P3_output = DarknetConv2D_BN_Leaky(256, (3, 3))(P3)
    P3_output = DarknetConv2D(len(anchors_mask[0]) * (num_classes + 5), (1, 1))(P3_output)

    # 52,52,128 -> 26,26,256
    P3_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P3)
    P3_downsample = DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(P3_downsample)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P3_downsample, P4])
    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4, 256)

    # ---------------------------------------------------#
    #   y2 = (batch_size,26,26,3,85)
    # ---------------------------------------------------#
    P4_output = DarknetConv2D_BN_Leaky(512, (3, 3))(P4)
    P4_output = DarknetConv2D(len(anchors_mask[1]) * (num_classes + 5), (1, 1))(P4_output)

    # 26,26,256 -> 13,13,512
    P4_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P4)
    P4_downsample = DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(P4_downsample)
    # 13,13,512 + 13,13,512 -> 13,13,1024
    P5 = Concatenate()([P4_downsample, P5])
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = make_five_convs(P5, 512)

    # ---------------------------------------------------#
    #   y1 = (batch_size,13,13,3,85)
    # ---------------------------------------------------#
    P5_output = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5_output = DarknetConv2D(len(anchors_mask[2]) * (num_classes + 5), (1, 1))(P5_output)

    return Model(inputs, [P5_output, P4_output, P3_output])
```



### 2. Training tricks in YoloV4

#### 2.1 Mosaic data argumentation

YoloV4 use Mosaic data argumentation in data preprocess and Mosaic data argumentation is kind of borrow the ideal from CutMix data argumentation. CutMix is using 2 images to stitch a new image, the example is shown as below:

![cutmix](https://github.com/Qucy/yolo4-tf-27/blob/master/img/cutmix.png)

For Mosaic, it uses 4 images instead of 2, according to the paper using 4 images will provide a much more meaningful and complex background in the image. And when calculate BN it will include all 4 images information. Some examples is shown as below:

![Mosaic](https://github.com/Qucy/yolo4-tf-27/blob/master/img/Mosaic.jpg)

The implementation steps for Mosaic have 3 steps,  randomly pick 4 images and scaling, resize, flip these 4 images. Put these 4 images at top left, top right, bottom left and bottom right and stitch these 4 images one by one. Below image depicting the steps for Mosaic data argumentation.![Mosaic_step](https://github.com/Qucy/yolo4-tf-27/blob/master/img/Mosaic_step.jpg)



#### 2.2 Label smoothing

Label smoothing is a quite simple and formula is as below:

```python
new_onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes
```

When label smoothing = 0.01 the formula is as below:

```python
new_onehot_labels = onehot_labels * (1 - 0.01) + 0.01 / num_classes
```

For instance, if we are doing binary classification and our ground truth label is 0 and 1. If label smoothing is 0.005 then our ground truth label will turn into  0.005 and 0.995. By doing so we allow model can be not so perfect and prevent overfitting.



#### 2.3 CIOU

Coming soon



#### 2.4 Cosine annealing learning rate

Coming soon