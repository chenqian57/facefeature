import timm
from functools import partial
from torch import nn
from .common import get_norm, get_activation


def build_model(cfg):
    # cfg，包含模型的各种配置信息
    
    # 使用cfg.ACT_LAYER配置的激活函数名称创建一个激活层（act_layer）
    act_layer = get_activation(name=cfg.ACT_LAYER)
    # 使用cfg.NORM_LAYER配置的归一化函数名称创建一个归一化层（norm_layer）。
    norm_layer = get_norm(name=cfg.NORM_LAYER)
    
    # 定义一个局部函数Backbone，它使用timm.create_model创建一个预训练的骨干网络模型，其中
    # model_name使用cfg.BACKBONE指定的模型名称
    # num_classes设为0，因为该模型的输出是嵌入向量而不是类别。
    # global_pool根据cfg.POOLING是否为True决定是否进行全局池化。
    # pretrained设为True，表示使用预训练的权重。
    # norm_layer和act_layer使用上面创建的 归一化层 和 激活层。
    # exportable设为True，表示模型将用于导出。

    Backbone = partial(
        timm.create_model,
        model_name=cfg.BACKBONE,
        num_classes=0,
        global_pool="avg" if cfg.POOLING else "",
        pretrained=True,
        norm_layer=norm_layer,
        act_layer=act_layer,
        # head_norm_first=True,
        exportable=True,
    )

    # 如果cfg.BACKBONE包含"convnext"，并且不包含"nano"或"tiny"，
    # 则使用Backbone(conv_mlp=True)创建一个具有ConvMLP结构的骨干网络模型，
    # 否则使用Backbone()创建一个普通的骨干网络模型。
    if "convnext" in cfg.BACKBONE and (
        "nano" not in cfg.BACKBONE and "tiny" not in cfg.BACKBONE
    ):
        backbone = Backbone(conv_mlp=True)
    else:
        backbone = Backbone()



    # 使用FaceModel类创建一个人脸识别模型：
    # backbone为上面创建的骨干网络模型。
    # num_features设为cfg.EMBEDDING_DIM，表示嵌入向量的维度。
    # drop_ratio设为cfg.get("DROP_RATIO", 0.0)，表示dropout比率，如果cfg中没有定义，则使用默认值0.0。
    # pool设为cfg.POOLING，表示是否进行全局池化。
    return FaceModel(
        backbone=backbone,
        num_features=cfg.EMBEDDING_DIM,
        drop_ratio=cfg.get("DROP_RATIO", 0.0),
        pool=cfg.POOLING,
    )



















class FaceModel(nn.Module):
    # backbone表示模型的主干网络，
    # num_features表示模型输出的特征维度，
    # drop_ratio表示Dropout的比率，
    # pool表示是否使用Pooling。
    def __init__(self, backbone, num_features=512, drop_ratio=0.0, pool=False) -> None:
        super().__init__()
        # 在构造函数中，首先计算了fc_scale和channels两个变量，
        # 分别表示最后一个全连接层的输入维度和主干网络输出的通道数。
        self.fc_scale = 1 if pool else 3 * 3

        self.channels = backbone.num_features
        self.num_features = num_features


        # 接着，定义了 backbone 和 output_layer 两个成员变量，分别表示主干网络和输出层。
        # output N, C, 3, 3
        self.backbone = backbone
        # output_layer由 BatchNorm2d、Dropout、Flatten和Linear 四个层组成，用于将 主干网络 输出的 特征图 映射为指定维度的特征向量。
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(self.channels),
            nn.Dropout(drop_ratio),
            nn.Flatten(),
            nn.Linear(self.channels * self.fc_scale, self.num_features),
            # nn.BatchNorm1d(self.num_features),
        )
        self.features = nn.BatchNorm1d(self.num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False


    # 在类的forward函数中，首先通过主干网络backbone计算特征图，然后根据特征图的维度情况，将其转换为指定维度的张量。
    # 接着，通过output_layer将特征图映射为特征向量，最后通过features进行BatchNorm操作，返回最终的特征向量。
    def forward(self, x):
        x = self.backbone(x)
        if x.dim() == 2:
            b, c = x.shape
            x = x.view(b, c, 1, 1)
        x = self.output_layer(x)
        x = self.features(x)
        return x
