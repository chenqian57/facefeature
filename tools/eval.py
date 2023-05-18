import torch




import sys
sys.path.append('/home/qiujing/cqwork/facefeature_202208032/src')  # /home/qiujing/cqwork/facefeature_20220803/src   ./src
print(sys.path)

from src.metric_trainer.core.evaluator import Evalautor
from metric_trainer.models import build_model      # /home/qiujing/cqwork/facefeature_20220803/src/metric_trainer/models/model.py




from nets.arcface import Arcface

from timm import create_model
from lqcv.utils.timer import Timer
from omegaconf import OmegaConf
import argparse



def parse_opt():
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "-w",
        "--weight",
        type=str,
        default="/home/qiujing/cqwork/facefeature_20220803/weights/FaceFeature_112-112_20220803.pt",
        # /home/qiujing/cqwork/facefeature_20220803/weights/FaceFeature_112-112_20220803.pt
        # 

        help="weight path",
    )
    opt = parser.parse_args()
    return opt






if __name__ == "__main__":


    # 解析脚本的命令行参数，返回一个名为 opt 的命名空间对象
    opt = parse_opt()


    # 读取配置文件，并返回一个配置字典 cfg
    cfg = OmegaConf.load(opt.config)

    # 输出配置中 MODEL 字段的值
    print(cfg.MODEL)

    # 根据配置中的 MODEL 字段构建模型对象 model。
    model = build_model(cfg.MODEL)

    # torch.load(opt.weight) 加载指定的模型权重文件，返回一个字典 ckpt
    ckpt = torch.load(opt.weight)

    # 输出模型权重字典的键列表
    print(ckpt.keys())
    # model.load_state_dict(ckpt["model"])

    # 将加载的模型权重应用到 model 对象中
    # 将预训练模型的参数加载到模型中
    # 使用了 Python 字典推导式来创建一个新的字典，该字典将从预训练模型中读取的权重参数映射到当前模型中的相应权重。
    # 这里，字典的键是当前模型的参数名称（去除了 'module.' 前缀），
    # 而字典的值是从预训练模型中读取的权重参数。
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model'].items()})
    # 将模型对象移到 GPU 上运行
    model.cuda()

    # 设置模型为测试模式，用于推理阶段
    model.eval()


    # img = torch.rand((1, 3, 112, 112), dtype=torch.float32).cuda()
    # time = Timer(start=True, round=2, unit="ms")
    # for i in tqdm(range(100), total=100):
    #     model(img)
    # print(f"average inference time: {time.since_start() / 100}ms")
    # exit()
    
    # png_save_path   = "/home/qiujing/cqwork/facefeature_20220803/model_data//roc_test.png"
    # /home/qiujing/cqwork/facefeature_20220803/model_data//roc_test.png

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
        testor.val(model, flip=True)
