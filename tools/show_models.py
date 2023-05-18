from timm import create_model
import torch.nn as nn
import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)
pprint(model_names)
#
# model = create_model('cspresnet50', pretrained=True, act_layer=nn.SiLU)
# model.eval()

# from metric_trainer.models import build_model
# from omegaconf import OmegaConf
# import torch
#
# cfg = OmegaConf.load("configs/test_folder.yaml")
# model = build_model(cfg.MODEL)
# # model.eval()
# img = torch.randn((2, 3, 112, 112))
# output = model(img)
# print(output.shape)
#
