import torch
from omegaconf import OmegaConf

import sys
sys.path.append('/home/qiujing/cqwork/facefeature_202208031')  # /home/qiujing/cqwork/facefeature_20220803/src   ./src
from src.metric_trainer.models import build_model
import argparse





def torch2Onnx(model, dynamic=False):
    """
    pytorch转onnx
    """
    # 输入placeholder
    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_output = model(dummy_input)
    print(dummy_output.shape)

    # Export to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        "face.onnx",
        input_names=["inputs"],
        output_names=["outputs"],
        # verbose=False,
        opset_version=12,
        dynamic_axes={
            "inputs": {0: "batch"},  # shape(1,3,640,640)
            "outputs": {0: "batch", 1: "width"},  # shape(1,3,640,640)
        }
        if dynamic
        else None,
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/test.yaml",
        help="config file",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="image dir or image file",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=str,
        default="",
        help="weight path",
    )
    opt = parser.parse_args()
    return opt






if __name__ == "__main__":
    opt = parse_opt()
    cfg = OmegaConf.load(opt.config)

    model = build_model(cfg.MODEL)
    ckpt = torch.load(opt.weight, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    model.eval()

    torch2Onnx(model, dynamic=opt.dynamic)
