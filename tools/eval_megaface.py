

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from easydict import EasyDict as edict
import time
import sys
import numpy as np
import argparse
import struct
import cv2
import sklearn
# from sklearn.preprocessing import normalize
from torch.nn.functional import normalize




import mxnet as mx
from mxnet import ndarray as nd


import sys
sys.path.append('/home/qiujing/cqwork/facefeature_202208032/src')  # /home/qiujing/cqwork/facefeature_20220803/src   ./src
from nets.arcface import get_model
# /home/qiujing/cqwork/facefeature_202208032/src/nets



def read_img(image_path):

  img = cv2.imread(image_path, cv2.IMREAD_COLOR)
  return img  # numpy数组



def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, help='', default=1)            # 批处理大小
  parser.add_argument('--image_size', type=str, help='', default='3,112,112')  # 通道数、高度和宽度
  parser.add_argument('--gpu', type=int, help='', default=0)



  parser.add_argument('--backbone', type=str, help='', default='ir34')
    # ir18, ir34, ir50, ir100, ir200


  parser.add_argument('--model_path', type=str, help='', default='/mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r34_fp16/backbone.pth')
    # /mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r34_fp16/backbone.pth
    # /mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth
    # /mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r100_fp16/backbone.pth
  # default=''
  # --model 参数指定模型的路径和参数。
  # --model './MS1MV2-mxnet-r100-ii,0000'
  # /home/qiujing/cqwork/facefeature_202208032/eval/megaface/MS1MV2-mxnet-r100-ii

  # /mnt/ssd/qiujing/arcface/megafacedata/conver1/model_r50.onnx
  # /mnt/ssd/qiujing/arcface/eval/model_14w.pth                                    # resnet50
  # /mnt/ssd/qiujing/arcface/arcface_mobilefacenet.pth                             # mobilefacenet
  # /mnt/ssd/qiujing/arcface/arcface_iresnet50.pth                                 # iresnet50
  # /home/qiujing/cqwork/arcface-pytorch/model_data/mobilenet_v1_backbone_weights.pth




  parser.add_argument('--algo', type=str, help='', default='r34')
  # default='insightface'
  # 定义使用的人脸识别算法为 ii
  # ALGO="ii" # ms1mv2   # r100ii
  # --algo "$ALGO"
  # r50
  # r100
  # r34

  parser.add_argument('--output', type=str, help='', default='/mnt/ssd/qiujing/arcface/megafacedata/feature_out1r34')                 # /mnt/ssd/qiujing/arcface/megafacedata/feature_out
  # ./feature_out
  # /mnt/ssd/qiujing/arcface/megafacedata/feature_out1
  # /mnt/ssd/qiujing/arcface/megafacedata/feature_out1r50
  # /mnt/ssd/qiujing/arcface/megafacedata/feature_out2r100
  # /mnt/ssd/qiujing/arcface/megafacedata/feature_out1r34

  parser.add_argument('--feature-dir-input', type=str, help='', default='/mnt/ssd/qiujing/arcface/megafacedata/feature_out1r34')
  # /mnt/ssd/qiujing/arcface/megafacedata/feature_out
  # ./feature_out
  # /mnt/ssd/qiujing/arcface/megafacedata/feature_out1
  # feature_out1r50
  # feature_out2r100
  # feature_out1r34

  parser.add_argument('--feature-dir-out', type=str, help='', default='/mnt/ssd/qiujing/arcface/megafacedata/feature_out1r34_clean')
  # /mnt/ssd/qiujing/arcface/megafacedata/feature_out_clean
  # ./feature_out_clean
  # /mnt/ssd/qiujing/arcface/megafacedata/feature_out1_clean
  # feature_out1r50_clean
  # feature_out2r100_clean
  # feature_out1r34_clean


  parser.add_argument('--distractor_feature_path', help='Path to MegaFace Features', 
                       default='/mnt/ssd/qiujing/arcface/megafacedata/feature_out1r34_clean/megaface')
    # 诱导答案，
    # /mnt/ssd/qiujing/arcface/megafacedata/feature_out_clean/megaface
    # feature_out1_clean
    # feature_out1r50_clean
    # feature_out2r100_clean
    # feature_out1r34_clean

    # parser.add_argument('probe_feature_path', help='Path to FaceScrub Features')
  parser.add_argument('--probe_feature_path', help='Path to FaceScrub Features', 
                       default='/mnt/ssd/qiujing/arcface/megafacedata/feature_out1r34_clean/facescrub')
    # /mnt/ssd/qiujing/arcface/megafacedata/feature_out_clean/facescrub
    # feature_out1_clean
    # feature_out1r50_clean
    # feature_out2r100_clean
    # feature_out1r34_clean

    # parser.add_argument('file_ending',help='Ending appended to original photo files. i.e. 11084833664_0.jpg_LBP_100x100.bin => _LBP_100x100.bin')
  parser.add_argument('--file_ending', help='Ending appended to original photo files. i.e. 11084833664_0.jpg_LBP_100x100.bin => _LBP_100x100.bin', 
                       default='_r34.bin' )
    # 结束附加到原始照片文件
    # _ii.bin
    # _r50.bin
    # _r100.bin
    # _r34.bin












  parser.add_argument('--facescrub-lst', type=str, help='', default='/mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/facescrub_lst')   # /mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/facescrub_lst      # ./data/facescrub_lst
  parser.add_argument('--megaface-lst', type=str, help='', default='/mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/megaface_lst')     # /mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/megaface_lst       # ./data/megaface_lst
  parser.add_argument('--facescrub-root', type=str, help='', default='/mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/facescrub_images')  # /mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/facescrub_images
  # ./data/megaface_testpack_v1.0/facescrub_images
  parser.add_argument('--megaface-root', type=str, help='', default='/mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/megaface_images')    # /mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/megaface_images
  # ./data/megaface_testpack_v1.0/megaface_images

  # parser.add_argument('--path_json', type=str, help='', default='/mnt/ssd/qiujing/arcface/megafacedata/conver1/model_r50.json')
  parser.add_argument('--nomf', default=False, action="store_true", help='')
  parser.add_argument('--facescrub-noises', type=str, help='', default='/mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/facescrub_noises.txt')
  # /mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/facescrub_noises.txt
  # ./data/facescrub_noises.txt
  parser.add_argument('--megaface-noises', type=str, help='', default='/mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/megaface_noises.txt')
  # /mnt/ssd/qiujing/arcface/megafacedata/data/megaface_testpack_v1.0/megaface_noises.txt
  # ./data/megaface_noises.txt
  parser.add_argument('--out_root', help='File output directory, outputs results files, score matrix files, and feature lists used', 
                       default='/mnt/ssd/qiujing/arcface/megafacedata/results')
    # /mnt/ssd/qiujing/arcface/megafacedata/results
  parser.add_argument('-s', '--sizes', type=int, nargs='+', help='(optional) Size(s) of feature list(s) to create. Default: 10 100 1000 10000 100000 1000000', default=1000000)
  parser.add_argument('-m', '--model', type=str, help='(optional) Scoring model to use. Default: ../models/jb_identity.bin')
  parser.add_argument('-ns','--num_sets', help='Set to change number of sets to run on. Default: 1')
  parser.add_argument('-d','--delete_matrices', dest='delete_matrices', action='store_true', help='Deletes matrices used while computing results. Reduces space needed to run test.')
  parser.add_argument('-p','--probe_list', help='Set to use different probe list. Default: ../templatelists/facescrub_features_list.json', 
                      default='/mnt/ssd/qiujing/arcface/megaface/devkit/templatelists/facescrub_features_list.json')
    # 
    # -p /home/qiujing/cqwork/facefeature_202208032/eval/megaface/devkit/templatelists/facescrub_features_list.json
    # 设置为使用不同的探测列表
  parser.add_argument('-dlp','--distractor_list_path', help='Set to change path used for distractor lists')
    # 设置为更改用于干扰列表的路径
    # , default='/home/qiujing/cqwork/facefeature_202208032/eval/megaface/devkit/templatelists/megaface_features_list.json'
    

    # 通过调用os.path.dirname()函数并传入MEGAFACE_LIST_BASENAME变量，可以获取到该文件的目录路径，将其赋值给distractor_list_path变量。
  parser.set_defaults(model=MODEL, num_sets=1, sizes=[10, 100, 1000, 10000, 100000, 1000000], probe_list=PROBE_LIST_BASENAME, distractor_list_path=os.path.dirname(MEGAFACE_LIST_BASENAME))
  return parser.parse_args(argv)





# 修改1
import torch
import numpy as np
import torch.utils.data as Data
import struct
# import sklearn.preprocessing

def get_feature(imgs, nets):
    count = len(imgs)


    for idx, img in enumerate(imgs):

        img = img[:,:,::-1] #to rgb
        img = img.transpose(2, 0, 1)
        # img = np.transpose(img, (2,0,1))

        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.clone().detach()
       # data[count+idx] = img

    # F = []
    embeddings = []
    for net in nets:
        img = img.cuda()
        img = img.float()
        img.div_(255).sub_(0.5).div_(0.5)

        img = img.unsqueeze(0)
        # print(img.shape)
        # 将数据传入模型中进行预测

        out = net.model(img)
        out = normalize(out)

        # tensor张量对象
        # out = torch.nn.functional.normalize(out)
        
        # 把 out 转为 numpy 数组
        embedding = out.detach().cpu().numpy()
        embeddings.append(embedding)
        # pbar.set_description(s)



    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


    # F = np.concatenate(F, axis=1)
    # F = sklearn.preprocessing.normalize(F)
    # return F








def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature),1,4,5))
        f.write(struct.pack("%df" % len(feature), *feature))


def get_and_write(buffer, nets):
    imgs = []
    for k in buffer:
        imgs.append(k[0])



    features = get_feature(imgs, nets)

    # print(features.shape)
    # print(len(buffer))
    assert features.shape[0] == len(buffer)

    for ik,k in enumerate(buffer):
        out_path = k[1]
        feature = features[ik].flatten()

        # feature = features[ik].flatten().numpy()
        write_bin(out_path, feature)



# 修改1
import torch
import os
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def main1(args):

  print(args)

  gpuid = args.gpu
  
  # device = torch.device("cuda:{}".format(gpuid) if torch.cuda.is_available() else "cpu")
  ctx = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")
  nets = []
  image_shape = [int(x) for x in args.image_size.split(',')]

  # 使用了 Python 中的 for 循环语法。它的作用是将字符串 args.model 按照竖杠（|）分割成多个子字符串，并针对每个子字符串执行一次循环体中的代码。
  # 具体来说，这个代码中的 args.model 可能是一个字符串，其中包含了多个模型的名称，这些名称之间用竖杠分隔。例如，args.model 可能是 "model1|model2|model3"。
  # 这个代码会先调用 split('|') 方法，将字符串按照竖杠进行分割，并将得到的多个子字符串组成一个列表。然后
  # 它使用 for 循环语句，对于这个列表中的每个子字符串，都将它赋值给变量 model。
  # 在循环体中，可以使用 model 这个变量来执行针对每个模型的操作。例如，可以调用相应的函数，或者使用这个变量来构造一个文件名或目录名。
  
  # for model in args.model:
  backbone        = args.backbone  
  # backbone        = "ir50"    # 
  # ir18, ir34, ir50, ir100, ir200

  print("backbone: ", backbone)

  model_path      = args.model_path
  print("model_path: ", model_path)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  net = edict()
  net.ctx = ctx
    # # 将模型参数、结构和辅助参数加载到字典中。
  net.model = get_model(backbone, fp16=False)
  net.model.eval()
  net.model.to(net.ctx)
  net.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    # 将字典 net 添加到 nets 列表中，以便后续使用该模型进行推理或训练。
  nets.append(net)





  facescrub_out = os.path.join(args.output, 'facescrub')
  megaface_out = os.path.join(args.output, 'megaface')

  i = 0
  succ = 0

  buffer = []
  for line in open(args.facescrub_lst, 'r'):
    if i%1000==0:
      print("writing fs",i, succ)
    i+=1
    image_path = line.strip()
    _path = image_path.split('/')
    a,b = _path[-2], _path[-1]
    out_dir = os.path.join(facescrub_out, a)


    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    image_path = os.path.join(args.facescrub_root, image_path)
    img = read_img(image_path)

    if img is None:
      print('read error:', image_path)
      continue
    out_path = os.path.join(out_dir, b+"_%s.bin"%(args.algo))
    item = (img, out_path)
    buffer.append(item)
    
    
    # 对于每个数据集，当读取的图像数(buffer)达到一定数量(args.batch_size)时，调用函数 get_and_write() 进行人脸识别，然后清空 buffer。
    if len(buffer)==args.batch_size:


      # 
      get_and_write(buffer, nets)
      buffer = []
    succ+=1
  if len(buffer)>0:
    get_and_write(buffer, nets)
    buffer = []
  print('fs stat',i, succ)
  if args.nomf:
    return



  i = 0
  succ = 0
  buffer = []


  
  for line in open(args.megaface_lst, 'r'):
    if i%1000==0:
      print("writing mf",i, succ)
    i+=1
    image_path = line.strip()
    _path = image_path.split('/')
    a1, a2, b = _path[-3], _path[-2], _path[-1]
    out_dir = os.path.join(megaface_out, a1, a2)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    image_path = os.path.join(args.megaface_root, image_path)
    img = read_img(image_path)
    if img is None:
      print('read error:', image_path)
      continue
    out_path = os.path.join(out_dir, b+"_%s.bin"%(args.algo))
    item = (img, out_path)
    buffer.append(item)
    if len(buffer)==args.batch_size:
      get_and_write(buffer, nets)
      buffer = []
    succ+=1
  if len(buffer)>0:
    get_and_write(buffer, nets)
    buffer = []

  # 最后输出每个数据集的处理状态信息，包括读取的图像数(i) 和 成功处理的图像数(succ)
  print('mf stat',i, succ)










import os
import datetime
import time
import shutil
import sys
import numpy as np
import argparse
import struct
import cv2
import mxnet as mx
from mxnet import ndarray as nd



feature_dim = 512
feature_ext = 1




# 读取二进制文件
# 从二进制文件中读取数据并返回一个 numpy 数组
def load_bin(path, fill = 0.0):
  # 函数的输入是二进制文件的路径path和填充值 fill（默认值为0.0），输出是一个 numpy数组 feature。

  # 打开文件 path，读取前 16 个字节（即4个32位整数）到变量bb中
  with open(path, 'rb') as f:
    bb = f.read(4*4)
    #print(len(bb))

    # 使用 struct模块 的 unpack函数 将bb 解包成 4个整数 并赋值给变量v
    v = struct.unpack('4i', bb)
    #print(v[0])

    # 从文件中读取 v[0]个32位 浮点数，并将它们存储在 变量 v 中
    bb = f.read(v[0]*4)
    v = struct.unpack("%df"%(v[0]), bb)

    # 使用np.full函数创建一个形状为(feature_dim+feature_ext,)的numpy数组feature，并将其所有元素初始化为fill。
    # 其中，feature_dim和feature_ext是两个全局变量。
    feature = np.full( (feature_dim+feature_ext,), fill, dtype=np.float32)

    # 将变量 v 的前 v[0] 个元素存储在 feature的前 feature_dim 个元素中。
    feature[0:feature_dim] = v
    #feature = np.array( v, dtype=np.float32)
  #print(feature.shape)
  #print(np.linalg.norm(feature))

  return feature










# 将特征写入二进制文件
# 将给定的 feature 列表写入到指定路径下的二进制文件中
def write_bin(path, feature):
  # path：一个字符串类型的参数，表示要写入二进制文件的路径。
  # feature：一个列表类型的参数，表示要写入到二进制文件中的特征数据。

  # 在函数内部，首先将 feature 转换为列表类型，然后使用 with 语句打开指定路径下的二进制文件。
  feature = list(feature)
  with open(path, 'wb') as f:
    # 使用 struct.pack() 函数将特定格式的数据打包为字节字符串并写入文件中。

    # 具体来说，使用 struct.pack('4i', len(feature),1,4,5) 打包了 4 个整数，
    # 其中第一个整数为 len(feature)，表示 feature 列表的长度；
    # 第二个整数为 1，表示数据类型为单精度浮点数；
    # 第三个整数为 4，表示每个单精度浮点数的大小为 4 字节；
    # 第四个整数为 5，表示字节序为小端序。
    f.write(struct.pack('4i', len(feature),1,4,5))

    # 使用 struct.pack("%df"%len(feature), *feature) 打包 feature 列表中的所有浮点数，并将其写入到文件中。
    # 其中，"%df"%len(feature) 表示使用 %d 占位符将会被 feature 列表的长度替换，从而形成格式字符串 %4f，该格式字符串表示每个浮点数占用 4 个字节。
    # 最后，使用 * 运算符展开 feature 列表中的所有元素，并作为参数传递给 struct.pack() 函数
    f.write(struct.pack("%df"%len(feature), *feature))












# 字典 fs_noise_map

def main2(args):


  fs_noise_map = {}
  # 首先读取一个名为args.facescrub_noises的文件，该文件包含了需要处理的噪声数据，通过循环读取文件中的每一行，
  # 并且忽略以#开头的注释行，将文件名的前缀作为字典中的键，将文件名去掉扩展名中最后一个下划线 及其 后面的部分 作为字典中的值，将这些键值对存储在名为 fs_noise_map 的字典中。
  for line in open(args.facescrub_noises, 'r'):
    if line.startswith('#'):
      continue
    line = line.strip()
    fname = line.split('.')[0]
    p = fname.rfind('_')
    fname = fname[0:p]
    fs_noise_map[line] = fname


  print(len(fs_noise_map))


  i=0
  fname2center = {}
  noises = []


  # 接着读取一个名为 args.facescrub_lst 的文件，该文件包含了需要处理的人脸图像数据，
  # 通过循环读取文件中的每一行，并且根据文件路径读取特征数据
  for line in open(args.facescrub_lst, 'r'):
    if i%1000==0:
      print("reading fs",i)
    i+=1
    image_path = line.strip()
    _path = image_path.split('/')
    a, b = _path[-2], _path[-1]
    feature_path = os.path.join(args.feature_dir_input, 'facescrub', a, "%s_%s.bin"%(b, args.algo))
    feature_dir_out = os.path.join(args.feature_dir_out, 'facescrub', a)
    if not os.path.exists(feature_dir_out):
      os.makedirs(feature_dir_out)
    feature_path_out = os.path.join(feature_dir_out, "%s_%s.bin"%(b, args.algo))
    #print(b)
    # 如果该图像不是噪声数据，则将其特征数据写入名为 feature_path_out 的文件中，并且将该特征数据累加到名为 fname2center 的字典中对应的值中，
    if not b in fs_noise_map:
      #shutil.copyfile(feature_path, feature_path_out)
      feature = load_bin(feature_path)
      write_bin(feature_path_out, feature)
      if not a in fname2center:
        fname2center[a] = np.zeros((feature_dim+feature_ext,), dtype=np.float32)
      fname2center[a] += feature

    # 否则将该图像的路径信息存储在名为 noises 的列表中
    else:
      #print('n', b)
      noises.append( (a,b) )
  print(len(noises))






  # 对于名为noises的列表中的每一个元素，将其作为 键值对(a,b) 提取出来，通过 键a 在字典 fname2center 中查找对应的值，
  # 该值表示 正常数据 的 中心特征向量，然后随机生成一个 偏移向量，将其加到中心向量上得到新的特征向量，

  for k in noises:
    a,b = k
    assert a in fname2center
    center = fname2center[a]
    g = np.zeros( (feature_dim+feature_ext,), dtype=np.float32)
    g2 = np.random.uniform(-0.001, 0.001, (feature_dim,))
    g[0:feature_dim] = g2

    f = center+g

  # 然后进行归一化操作并写入名为 feature_path_out 的文件中
    _norm=np.linalg.norm(f)
    f /= _norm
    feature_path_out = os.path.join(args.feature_dir_out, 'facescrub', a, "%s_%s.bin"%(b, args.algo))
    write_bin(feature_path_out, f)


  mf_noise_map = {}



  # 接着读取一个名为 args.megaface_noises 的文件，该文件包含了需要处理的噪声数据，
  # 通过循环读取文件中的每一行，并且忽略以#开头的注释行，将文件名作为字典中的键，将值设为1，
  # 将这些键值对存储在名为 mf_noise_map 的字典中。
  for line in open(args.megaface_noises, 'r'):
    if line.startswith('#'):
      continue
    line = line.strip()
    _vec = line.split("\t")
    if len(_vec)>1:
      line = _vec[1]
    mf_noise_map[line] = 1

  print(len(mf_noise_map))


  i=0
  nrof_noises = 0



  # 最后读取一个名为args.megaface_lst的文件，该文件包含了需要处理的人脸图像数据(一系列图像路径信息)，通过循环读取文件中的每一行，并且根据文件路径读取特征数据。
  for line in open(args.megaface_lst, 'r'):
    if i%1000==0:
      print("reading mf",i)
    i+=1
    image_path = line.strip()
    _path = image_path.split('/')
    # 解析图像的路径，将其分解为 a1、a2 和 b 三个部分
    a1, a2, b = _path[-3], _path[-2], _path[-1]

    # 构造 输入特征文件路径 和 输出特征文件夹 路径
    feature_path = os.path.join(args.feature_dir_input, 'megaface', a1, a2, "%s_%s.bin"%(b, args.algo))    # /mnt/ssd/qiujing/arcface/megafacedata/feature_out
    feature_dir_out = os.path.join(args.feature_dir_out, 'megaface', a1, a2)                               # /mnt/ssd/qiujing/arcface/megafacedata/feature_out_clean


    # 先判断feature_dir_out目录是否存在，如果不存在则创建该目录
    if not os.path.exists(feature_dir_out):
      os.makedirs(feature_dir_out)
    feature_path_out = os.path.join(feature_dir_out, "%s_%s.bin"%(b, args.algo))
    bb = '/'.join([a1, a2, b])
    # print(b)



    # 如果该图像不是噪声数据，则将其特征数据写入名为 feature_path_out 的文件中，
    if not bb in mf_noise_map:
      feature = load_bin(feature_path)
      write_bin(feature_path_out, feature)
      #shutil.copyfile(feature_path, feature_path_out)

    # 否则根据名为 feature_path 读取特征数据 并且 加上一个随机生成的 偏移向量 得到新的 特征向量，然后进行 归一化操作 并写入名为 feature_path_out 的文件中
    else:
      feature = load_bin(feature_path, 100.0)
      write_bin(feature_path_out, feature)
      #g = np.random.uniform(-0.001, 0.001, (feature_dim,))
      #print('n', bb)
      #write_bin(feature_path_out, g)
      nrof_noises+=1

  # 打印 噪声数据 的数量
  print(nrof_noises)








import argparse
import os
import sys
import json
import random
import subprocess


MODEL = os.path.join('/mnt/ssd/qiujing/arcface/megaface/devkit', 'models', 'jb_identity.bin')
IDENTIFICATION_EXE = os.path.join('/mnt/ssd/qiujing/arcface/megaface/devkit', 'bin', 'Identification')
FUSE_RESULTS_EXE = os.path.join('/mnt/ssd/qiujing/arcface/megaface/devkit', 'bin', 'FuseResults')
MEGAFACE_LIST_BASENAME = os.path.join('/mnt/ssd/qiujing/arcface/megaface/devkit','templatelists','megaface_features_list.json')  # 'megaface_features_list.json'
# 'devkit'
# /home/qiujing/cqwork/facefeature_202208032/eval/megaface/devkit/templatelists/megaface_features_list.json_10_1

PROBE_LIST_BASENAME = os.path.join('/mnt/ssd/qiujing/arcface/megaface/devkit','templatelists','facescrub_features_list.json')







def main3(args):
    print(args)
    distractor_feature_path = args.distractor_feature_path
    out_root = args.out_root
    probe_feature_path = args.probe_feature_path
    model = args.model
    num_sets = args.num_sets
    sizes = args.sizes
    file_ending = args.file_ending
    alg_name = file_ending.split('.')[0].strip('_')
    delete_matrices = args.delete_matrices
    probe_list_basename = args.probe_list
    megaface_list_basename = os.path.join(args.distractor_list_path,os.path.basename(MEGAFACE_LIST_BASENAME))
    set_indices = range(1,int(num_sets) + 1)



    print(distractor_feature_path)
    assert os.path.exists(distractor_feature_path)
    assert os.path.exists(probe_feature_path)
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    if(not os.path.exists(os.path.join(out_root, "otherFiles"))):
        os.makedirs(os.path.join(out_root, "otherFiles"))
    other_out_root = os.path.join(out_root, "otherFiles")

    probe_name = os.path.basename(probe_list_basename).split('_')[0]
    distractor_name = os.path.basename(megaface_list_basename).split('_')[0]




    #Create feature lists for megaface for all sets and sizes and verifies all features exist
    missing = False
    for index in set_indices:
        for size in sizes:
            print('Creating feature list of {} photos for set {}'.format(size,str(index)))
            cur_list_name = megaface_list_basename + "_{}_{}".format(str(size), str(index))

            with open(cur_list_name) as fp:
                featureFile = json.load(fp)
                path_list = featureFile["path"]
                for i in range(len(path_list)):
                    path_list[i] = os.path.join(distractor_feature_path,path_list[i] + file_ending)
                    if(not os.path.isfile(path_list[i])):
                        print(path_list[i] + " is missing")
                        missing = True
                    if (i % 10000 == 0 and i > 0):
                        print(str(i) + " / " + str(len(path_list)))
                featureFile["path"] = path_list
                json.dump(featureFile, open(os.path.join(
                    other_out_root, '{}_features_{}_{}_{}'.format(distractor_name,alg_name,size,index)), 'w'), sort_keys=True, indent=4)
    if(missing):
        sys.exit("Features are missing...")
    





    #Create feature list for probe set
    with open(probe_list_basename) as fp:
        featureFile = json.load(fp)
        path_list = featureFile["path"]
        for i in range(len(path_list)):
            path_list[i] = os.path.join(probe_feature_path,path_list[i] + file_ending)
            if(not os.path.isfile(path_list[i])):
                print(path_list[i] + " is missing")
                missing = True
        featureFile["path"] = path_list
        json.dump(featureFile, open(os.path.join(
            other_out_root, '{}_features_{}'.format(probe_name,alg_name)), 'w'), sort_keys=True, indent=4)
        probe_feature_list = os.path.join(other_out_root, '{}_features_{}'.format(probe_name,alg_name))
    if(missing):
        sys.exit("Features are missing...")


    print('Running probe to probe comparison')
    probe_score_filename = os.path.join(
        other_out_root, '{}_{}_{}.bin'.format(probe_name, probe_name, alg_name))

    proc = subprocess.Popen(
        [IDENTIFICATION_EXE, model, "path", probe_feature_list, probe_feature_list, probe_score_filename])
    proc.communicate()




    for index in set_indices:
        for size in sizes:
            print('Running test with size {} images for set {}'.format(
                str(size), str(index)))
            args = [IDENTIFICATION_EXE, model, "path", os.path.join(other_out_root, '{}_features_{}_{}_{}'.format(distractor_name,alg_name,size,index)
                ), probe_feature_list, os.path.join(other_out_root, '{}_{}_{}_{}_{}.bin'.format(probe_name, distractor_name, alg_name, str(size),str(index)))]
            proc = subprocess.Popen(args)
            proc.communicate()

            print('Computing test results with {} images for set {}'.format(
                str(size), str(index)))
            args = [FUSE_RESULTS_EXE]
            args += [os.path.join(other_out_root, '{}_{}_{}_{}_{}.bin'.format(
                probe_name, distractor_name, alg_name, str(size), str(index)))]
            args += [os.path.join(other_out_root, '{}_{}_{}.bin'.format(
                probe_name, probe_name, alg_name)), probe_feature_list, str(size)]
            args += [os.path.join(out_root, "cmc_{}_{}_{}_{}_{}.json".format(
                probe_name, distractor_name, alg_name, str(size), str(index)))]
            args += [os.path.join(out_root, "matches_{}_{}_{}_{}_{}.json".format(
                probe_name, distractor_name, alg_name, str(size), str(index)))]
            proc = subprocess.Popen(args)
            proc.communicate()

            if(delete_matrices):
                os.remove(os.path.join(other_out_root, '{}_{}_{}_{}_{}.bin'.format(
                    probe_name, distractor_name, alg_name, str(size), str(index))))
















if __name__ == '__main__':
  main1(parse_arguments(sys.argv[1:]))
  main2(parse_arguments(sys.argv[1:]))
  main3(parse_arguments(sys.argv[1:]))



