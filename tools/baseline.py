from metric_trainer.models.baseline_model import Backbone
import torch
import os
from typing import List
from loguru import logger
from metric_trainer.eval.verification import test, load_bin
from lqcv.utils.timer import Timer


class CallBackVerification(object):
    def __init__(self, val_targets, rec_prefix, image_size=(112, 112)):
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.init_dataset(
            val_targets=val_targets, data_dir=rec_prefix, image_size=image_size
        )

    def ver_test(self, backbone: torch.nn.Module):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = test(
                self.ver_list[i], backbone, 10, 10
            )
            logger.info(
                "[%s]XNorm: %f" % (self.ver_name_list[i], xnorm)
            )
            logger.info(
                "[%s]Accuracy-Flip: %1.5f+-%1.5f"
                % (self.ver_name_list[i], acc2, std2)
            )

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logger.info(
                "[%s]Accuracy-Highest: %1.5f"
                % (self.ver_name_list[i], self.highest_acc_list[i])
            )
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, backbone: torch.nn.Module):
        backbone.eval()
        self.ver_test(backbone)


def load_state(model, weight_path):
    state_dict = torch.load(weight_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


if __name__ == "__main__":
    model = Backbone(num_layers=50, drop_ratio=0.5, mode="ir_se")
    model = load_state(model, weight_path="weights/model_14w.pth")
    model.cuda()
    model.eval()

    # img = torch.rand((1, 3, 112, 112), dtype=torch.float32).cuda()
    # time = Timer(start=True, round=2, unit="ms")
    # for i in range(100):
    #     model(img)
    # print(f"average inference time: {time.since_start() / 100}ms")
    # exit()

    callback_verification = CallBackVerification(
        val_targets=["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw", "vgg2_fp"],
        rec_prefix="/dataset/dataset/glint360k/glint360k",
    )
    callback_verification(model)
    # LFW: 0.99683, 57s
    # cfp_fp: 0.96514, 70s
    # agedb_30: 0.97200, 60s
    # calfw: 0.95583, 60s
    # cplfw: 0.91133, 60s
    # vgg2_fp: 0.94860, 50s
