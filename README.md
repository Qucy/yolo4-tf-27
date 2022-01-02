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

Label smoothing formula is as below:

```python
new_onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes
```

When label smoothing = 0.01 the formula is as below:

```python
new_onehot_labels = onehot_labels * (1 - 0.01) + 0.01 / num_classes
```

For instance, if we are doing binary classification and our ground truth label is 0 and 1. If label smoothing is 0.005 then our ground truth label will turn into  0.005 and 0.995. By doing so we allow model can be not so perfect and prevent overfitting.



#### 2.3 CIoU - Complete Intersection Over Union

In YoloV3 we use bounding box x, y, width and height to calculate bounding box regression loss.  Can we use IoU to calculate bounding box loss instead of using coordinates, height and width directly.  Yes, we can but using IoU directly will have a problem, if predicted box and ground truth box have no overlap then IoU will be zero and model will become hard to train.

Then someone raised other optimized IoU losses like GIoU, DIoU and CIoU. We will not discuss all of this in this article but if you are interested, you can read this article -> [Variants of IoU](https://medium.com/nerd-for-tech/day-90-dl-variants-of-iou-giou-diou-6c0a933dd2c7)  to know more about these losses.

For CIoU the formula is as below, it will consider (a) IoU which is overlap area, (b) central point distance, (c) the aspect ratio. Based on these parameters CIoU is computed. CIoU is a variant of DIoU with additional term representing the aspect ratio.

![CIoU](https://github.com/Qucy/yolo4-tf-27/blob/master/img/CIoU.jpg)

Source code is as below

```python
def box_ciou(b1, b2):
    """
    calc ciou
    :param b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    :param b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    :return: ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # -----------------------------------------------------------#
    #   calculate predicted box top left and right bottom coordinates
    #   b1_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b1_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    # -----------------------------------------------------------#
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # -----------------------------------------------------------#
    #   calculate ground truth box top left and right bottom coordinates
    #   b2_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b2_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    # -----------------------------------------------------------#
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # -----------------------------------------------------------#
    #   calc iou between prediction and ground truth
    #   iou => (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / K.maximum(union_area, K.epsilon())

    # -----------------------------------------------------------#
    #   calc center point Euclidean distance
    #   center_distance (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # -----------------------------------------------------------#
    #   calc diagonal length
    #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal, K.epsilon())

    v = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0],
                                                                                                         K.maximum(
                                                                                                             b2_wh[
                                                                                                                 ..., 1],
                                                                                                             K.epsilon()))) / (
                    math.pi * math.pi)
    alpha = v / K.maximum((1.0 - iou + v), K.epsilon())
    ciou = ciou - alpha * v

    ciou = K.expand_dims(ciou, -1)
    return ciou
```



#### 2.4 Cosine annealing and warm restart learning scheduler

The cosine annealed warm restart learning schedule has two parts, cosine annealing and warm restarts.

- **Cosine annealing** means that the cosine function is used as the learning rate annealing function. The cosine function has been shown to perform better than alternatives like simple linear annealing in practice.

- **Warm restarts** is the interesting part: it means that every so often, the learning rate is restated.

Cosine annealing has an ideas core to a good learning rate scheduler nowadays: periods with high learning rates and periods with low ones. The function of the periods of high learning rates in the scheduler is to prevent the learner from getting stuck in a local cost minima; the function of the periods of low learning rates in the scheduler is to allow it to converge to a near-true-optimal point within the (hopefully) global minima it finds.

![consine_annel](https://github.com/Qucy/yolo4-tf-27/blob/master/img/consine_annel.jpg)

Source code is as below, we can use callback function to implement this.

```python
class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """
    Customized call back function to implement cosine decay for lr
    """
    def __init__(self, T_max, eta_min=0, verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.T_max = T_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.init_lr = 0
        self.last_epoch = 0

    def on_train_begin(self, batch, logs=None):
        self.init_lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, batch, logs=None):
        learning_rate = self.eta_min + (self.init_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        self.last_epoch += 1

        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))
```



### 3. Train your model

#### 3.1 Prepare your dataset

To train your model you need prepare your datasets first, you can use VOC datasets or COCO datasets to train your model.

For VOC datasets you can download here http://host.robots.ox.ac.uk/pascal/VOC/

For COCO datasets you can download here https://cocodataset.org/#download

But for your own data you need to install a image label tool to label your data first:

You can use pip to install LabelImage and label your own image. link -> https://pypi.org/project/labelImg/



#### 3.2 Preprocess your dataset

Before to train your model, we need to preprocess our dataset, since the the VOC or COCO dataset's annotation is in XML format. We need to process it via **voc_annotation.py**.  Change your dataset path accordingly and change annotation_mode = 2 to generate train and validation dataset.

After preprocess your dataset successfully you should see 2007_train.txt and 2007_val.txt.



#### 3.3 Train your model

By using **voc_annotation.py** we've generated our training and testing datasets. By point our train path to these 2 files and we run train.py file to kickoff the training. Of course you can change the hyper parameter in the train.py and the model weights will be saved in logs file every epoch.



#### 3.4 Make predictions !

After your model is trained, you can modify the model weights file path point to the latest weights file path in the logs folder. And input the image path or folder and run predict.py to trigger the prediction.