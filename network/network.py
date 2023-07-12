import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from network.unet_model import UNet
from network.unet_parts import *

import sys

# Specify the graphics card
# torch.cuda.set_device(7)

def my_net(modelname, in_channels, num_classes, norm_layer=nn.BatchNorm2d, pretrain_file=None, pretrain=False, bn_eps=1e-5, bn_momentum=0.1, use_ms=False):

    if modelname == 'imagenet_ResUnet':
        if pretrain:
            print("Using pretrain model")
            model = smp.Unet(
                encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights="imagenet",
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=in_channels,
                # model output channels (number of classes in your dataset)
                classes=num_classes,
            )
        else:
            model = smp.Unet(
                encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=in_channels,
                # model output channels (number of classes in your dataset)
                classes=num_classes,
            )

    elif modelname == 'ssl_ResUnet':
        if pretrain:
            print("Using pretrain model")
            model = smp.Unet(
                encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights="ssl",
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=in_channels,
                # model output channels (number of classes in your dataset)
                classes=num_classes,
            )
        else:
            model = smp.Unet(
                encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights=None,
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=in_channels,
                # model output channels (number of classes in your dataset)
                classes=num_classes,
            )

    elif modelname == 'normalUnet':
        print("Using normal Unet")
        model = UNet(
            n_channels=in_channels,
            n_classes=num_classes,
            norm_layer=norm_layer
            )
        initialize_weights(model, norm_layer=norm_layer)

    else:
        print("model name are wrong")
    return model

if __name__ == "__main__":
    x = torch.randn((2, 1, 288, 288))
    model_r = my_net(modelname='normalUnet')
    model_l = my_net(modelname='normalUnet')
    preds_r = model_r(x)
    preds_l = model_l(x)
    preds = preds_r + preds_l
    print(x.shape)
    print(preds_r.shape)
    print(preds_l.shape)
    print(preds.shape)
