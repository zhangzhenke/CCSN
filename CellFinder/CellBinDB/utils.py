import torch.nn as nn


# 这个模块用于构建卷积神经网络 (CNN) 的基础单元。
class Conv2DBlock(nn.Module):
    """ Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 dropout: float = 0,) -> None:
        super().__init__()
        
        self.block = nn.Sequential(
            # 用于提取图像的特征
            # 同态填充,使输出特征图的大小与输入图像的大小相同
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=((kernel_size - 1) // 2),),
            # 用于加速训练并提高模型的稳定性。
            nn.BatchNorm2d(out_channels),
            # 引入非线性，使模型能够学习更复杂的特征。
            nn.ReLU(True),
            # 用于防止过拟合
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


# 用于上采样或特征图放大的反卷积模块
class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dropout: float = 0,) -> None:
        super().__init__()

        self.block = nn.Sequential(
            # 将输入特征图上采样到更高的分辨率
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               output_padding=0,),
            # # 同态填充,使输出特征图的大小与输入图像的大小相同
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=((kernel_size - 1) // 2),),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)





    



