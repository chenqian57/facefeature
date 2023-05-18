import torch




import sys
sys.path.append('/home/qiujing/cqwork/facefeature_202208032/src')  # /home/qiujing/cqwork/facefeature_20220803/src   ./src
print(sys.path)
from metric_trainer.core.evaluator import Evalautor
from metric_trainer.models import build_model        # /home/qiujing/cqwork/facefeature_20220803/src/metric_trainer/models/model.py


# from nets.model import Backbone
# from nets.arcface import Arcface


# from src.nets.model import Backbone
# from src.nets.arcface import Arcface



from nets.arcface import get_model

from timm import create_model
from lqcv.utils.timer import Timer
from omegaconf import OmegaConf
import argparse




def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b1",
        "--backbone",
        default="ir50",
        help="主干特征提取网络的选择, mobilefacenet,mobilenetv1,iresnet18,iresnet34,iresnet50,iresnet100,iresnet200,resnet50",
        # # ir18, ir34, ir50, ir100, ir200
        # mobilefacenet, mobilenetv1, iresnet18, iresnet34, iresnet50, iresnet100, iresnet200, resnet34, resnet50, resnet101, resnet152, vit, iresnet2060
    )


    parser.add_argument(
        "-w",
        "--weight",
        type=str,
        default="/mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth",

        # /mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r34_fp16/backbone.pth
        # /mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth
        # /mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r100_fp16/backbone.pth

        # /mnt/ssd/qiujing/arcface/arcface_mobilefacenet.pth                             # mobilefacenet
        # /mnt/ssd/qiujing/arcface/arcface_iresnet50.pth                                 # iresnet50
        # /home/qiujing/cqwork/arcface-pytorch/model_data/mobilenet_v1_backbone_weights.pth

        # /home/qiujing/cqwork/facefeature_20220803/weights/FaceFeature_112-112_20220803.p
        # /mnt/ssd/qiujing/arcface/eval/model_14w.pth   # resnet50
        help="weight path",
    )


    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/FaceFeature_112-112_20220803.yaml",
        help="config file",
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="/mnt/ssd/qiujing/glint360k",   # /mnt/ssd/qiujing/glint360k    /media/oyrq/4T/workspace/codes/face_feature/data
        help="val data path, if the argument is empty, then eval the val data from config",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )

    opt = parser.parse_args()
    return opt






if __name__ == "__main__":
    # 解析脚本的命令行参数，返回一个名为 opt 的命名空间对象
    opt = parse_opt()

    # 读取配置文件，并返回一个配置字典 cfg
    # cfg = OmegaConf.load(opt.config)

    # 输出配置中 MODEL 字段的值
    # print(cfg.MODEL)

    # 根据配置中的 MODEL 字段构建模型对象 model。
    # model = build_model(cfg.MODEL)


    print("backbone: ", opt.backbone)
    print("model_path: ", opt.weight)

    # model = Backbone(50, 0.4, 'ir_se')
    # model = Arcface(backbone=backbone, mode="predict")
    model = get_model(opt.backbone, fp16=False)
   
    # model = get_model('ir50', fp16=False)
    # ir18, ir34, ir50, ir100, ir200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(opt.weight, map_location=device), strict=False)
    model.cuda()
    model.eval()

    # 创建一个名为 testor 的 Evaluator 对象，用于在给定的数据集上进行评估。
    testor = Evalautor(
        # 指定需要评估的数据集列表。
        val_targets=["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw", "vgg2_fp"],

        # 指定数据集所在的根目录
        root_dir=opt.data,

        # 指定评估时使用的批量大小
        batch_size=opt.batch_size,
    )


    # 创建一个上下文，禁用梯度计算
    with torch.no_grad():
        # 调用 Evaluator 对象的 val() 方法，对指定的数据集进行评估，并计算识别精度。
        # model 是已经训练好的人脸识别模型，flip=True 表示在评估时使用水平翻转增强数据。
        testor.val(model, flip=False)  # True, False
