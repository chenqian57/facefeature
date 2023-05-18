from easydict import EasyDict as edict

# from easydict import EasyDiodelct as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.data_path = Path('/media/rachellee/新加卷7/zrm/InsightFace_Pytorch/data')
    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.input_size = [112, 112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.resize = 111
    if conf.resize == 111:
        conf.x = 7
    if conf.resize == 112:
        conf.x = 7
    elif conf.resize == 90:
        conf.x = 6
    elif conf.resize == 84:
        conf.x = 6
    elif conf.resize == 78:
        conf.x = 5
    elif conf.resize == 72:
        conf.x = 5
    elif conf.resize == 60:
        conf.x = 4
    conf.drop_ratio = 0.4 # or 0.6
    conf.net_mode = 'ir_se' # or 'ir_se'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # conf.device = "cpu"
    conf.test_transform = trans.Compose([
                    trans.Resize((conf.resize, conf.resize)),
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.glint_folder = conf.data_path/'faces_glint'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.test_folder = conf.data_path/'faces_test'
    conf.batch_size = 100 # irse net depth 50 
    conf.testdata = Path('data/faces_emore')
    
#   conf.batch_size = 200 # mobilefacenet
#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-1
        #conf.milestones = [12,15,18]
        conf.milestones = [4,8,11,14]
        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()    
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf
