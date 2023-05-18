<<<<<<< HEAD
# FaceFeature


# MetricTrainer
- 如果使用之前的CspResNet50模型, 则需要`git checkout 60ca7bc7ffb97d89df1fa8436d8ab02fa4e67a06`才支持, 因为后面添加convnext，稍微优化了一下声明模型的代码
- `configs/partial_glint360k.yaml`为之前(60ca7bc7ffb97d89df1fa8436d8ab02fa4e67a06)的配置文件
- `configs/partial_glint360k_adaface.yaml`为现在较新的配置文件，主要是增加了一些模型的定义
- 由于暂时还未进行特征比对，所以单独进行特征比对的代码未编写
- 该仓库是基于timm库的，因此可以调用timm库的任意模型，可能少数模型会存在单独设置的地方，则需要单独处理



## Quick Start

<details open>
<summary>Installation</summary>

Clone repo and install [requirements.txt](https://github.com/Laughing-q/yolov5-q/blob/master/requirements.txt) in a
**Python>=3.7.0** environment, including**PyTorch>=1.7.1**.

```python
pip install timm pytorch-metric-learning
pip install git+https://github.com/Laughing-q/lqcv.git
# 如果要使用convnext作为backbone，则不能安装官方的timm，官方的目前不支持替换激活函数, 只能采用下面的命令安装timm
pip install git+https://github.com/Laughing-q/pytorch-image-models.git
git clone http://192.168.2.196:11110/Train/facefeature.git
git clone https://192.168.100.15/ModelTrains/facefeature.git

cd facefeature
pip install -r requirements.txt
pip install -e .
```

```
安装
libnccl.so.2
libtbb.so.2
libopencv_core.so.2.4
```

- 也可以直接使用10.10.8.217上的face环境
  ```shell
  conda activate face
  ```

</details>

<details open>
<summary>Training</summary>

- 数据1, 来自insightface官方下载的1000W+的glint360k
```plain
├── root_dir
│   ├── lfw.bin           # 评估集lfw
│   ├── cfp_fp.bin        # 评估集cfp_fp
│   ├── agedb_30.bin      # 评估集agedb_30
│   ├── calfw.bin         # 评估集calfw.bin
│   ├── cplfw.bin         # 评估集cplfw.bin
│   ├── vgg2_fp.bin       # 评估集vgg2_fp.bin
│   ├── train.idx         # 训练集标签信息
│   ├── train.rec         # 训练集图片
```


- 数据2, 来自insightface官方下载的评估集megaface
```plain
├── megaface
│   ├── data              # 评估集megaface
│   ├── devkit            # 评估集megaface工具包
```


- 数据3, 来自insightface官方下载的预训练权重
```plain
├── aarcface_torch
│   ├── ms1mv3_arcface_r18_fp16      # ir18预训练权重
│   ├── ms1mv3_arcface_r34_fp16      # ir34预训练权重
│   ├── ms1mv3_arcface_r50_fp16      # ir50预训练权重
│   ├── ms1mv3_arcface_r100_fp16     # ir100预训练权重
│   ├── ms1mv3_arcface_r2060         # ir2060预训练权重
```


- 准备config.yaml如下, `configs/`查看更多实例.
  ```python
  MODEL:
    BACKBONE: 'convnext_base_in22ft1k'  # backbone, 依赖于timm库，具体有哪些backbone可以运行python tools/show_models.py查看
    ACT_LAYER: 'silu'                   # 激活函数，目前只支持silu, lrelu(leakyrelu), relu
    NORM_LAYER: 'bn'                    # 归一化层，目前只支持bn, gn, ln
    POOLING: False                      # backbone最后一层生成的特征图，是否采用pooling
    EMBEDDING_DIM: 512                  # 输出的特征向量维度
    LOSS: 'partial_fc_adaface'          # 损失函数, 目前支持partial_fc_adaface, partial_fc, 其他一些损失函数代码里有，但是测试效果不好
    NUM_CLASS: 360232                   # 人脸id数
    SAMPLE_RATE: 0.1                    # partial_fc的采样率

  DATASET:
    TYPE: 'glint360k'                   # dataset的类型，支持glint360k和folder，folder是从文件夹读取图片，由于该项目主要就是训练的glint360k数据集，这个不用改
    TRAIN: '/dataset/dataset/glint360k/glint360k'   # 训练数据路径
    VAL: '/dataset/dataset/glint360k/glint360k'     # 测试数据路径
    VAL_TARGETS: ['lfw', 'cfp_fp', "agedb_30"]      # 测试数据, 详细查看上面的数据路径描述
    IMG_SIZE: 112                                   # 图片输入分辨率
    NUM_IMAGES: 17091657                            # 总图片数量
    TRANSFORM:                                      # 数据增强的参数，随机crop，随机左右翻转，随意变化亮度对比度
      RandomResizedCrop: 0.5
      HorizontalFlip: 0.5
      RandomBrightnessContrast: 0.5

  SOLVER:
    OPTIM: 'sgd'                                   # 优化器，目前仅支持sgd
    BATCH_SIZE_PER_GPU: 64                         # 每张GPU上的batch-size
    BASE_LR: 0.4                                   # 基础学习率
    WARMUP_EPOCH: 2                                # 热身训练的轮次
    NUM_EPOCH: 20                                  # 训练总轮次
    MOMENTUM: 0.9                                  # 优化器参数
    WEIGHT_DECAY: 0.0005                           # 优化器参数
    FP16: False                                    # 是否开启半精度训练
  OUTPUT: 'runs'                                   # 保存模型以及日志的路径
  NUM_WORKERS: 8                                   # 加载dataloader时的workers数量
  ```

- `partial_fc` 依赖`DDP`模式, 所以如果是单卡训练则设置下面的命令n=1即可.
- Multi GPU(DDP)
```shell
python -m torch.distributed.run --nproc_per_node=n tools/train.py -c configs/test.yaml
```
</details>

<details open>
<summary>Eval1-glint360k</summary>

- `-c` 传入训练模型时保存的cfg.yaml(在上面配置文件中的`OUTPUT`中)
- `-w` 传入目录中保存的模型
- `-d` 传入评估集的root目录(在上面`数据1`中)
```shell
python tools/eval.py -c configs/partial_glint360k.yaml -d root_dir \
    -w runs/CosFace_noaug/best.pt
```
</details>

<details open>
<summary>Eval2-ms1mv3</summary>

- `--backbone`            设置backbone
- `--model_path`          传入权重路径(在上面`数据3`中)
- `-d`                    传入评估集的root目录(在上面`数据1`中)
```shell
python tools/eval2.py --backbone --model_path -d
```
</details>

<details open>
<summary>Eval3-ms1mv3-megaface</summary>

- `--backbone`                设置backbone
- `--model_path`              传入权重路径(在上面`数据3`中)
- `--algo`                    设置算法名称
- `--output`                  设置bin文件输出目录
- `--feature-dir-input`       传入要进行噪声处理的bin文件输入目录，和 `--output` 保持一致
- `--feature-dir-out`         设置经过噪声处理的bin文件输出目录
- `--distractor_feature_path` 传入经过噪声处理的megaface目录
- `--probe_feature_path`      传入经过噪声处理的facescrub目录
- `--file_ending`             设置文件后缀名(和 `--algo` 对应)
- `--out_root`                文件输出目录，输出结果文件，分数矩阵文件，以及使用的特性列表

- `--facescrub-lst`           传入评估集megaface的data目录(在上面`数据2`中)
- `--megaface-lst`
- `--facescrub-root`
- `--megaface-root`
- `--facescrub-noises`
- `--megaface-noises`

- `MODEL`、`IDENTIFICATION_EXE`、`FUSE_RESULTS_EXE`、`MEGAFACE_LIST_BASENAME`、`PROBE_LIST_BASENAME`，传入megaface devkit工具包路径，(在上面`数据2`中)


```shell
python tools/eval_megaface.py --backbone --model_path --algo --output --feature-dir-input --feature-dir-out --distractor_feature_path --probe_feature_path --file_ending
```
</details>


## 🍩Export
- to onnx
  ```shell
  python export/pt2onnx.py -c configs/test.yaml -w runs/CosFace_noaug/best.pt
  ```




## Tips
- I haven't test resume at all, maybe there will get some bug.
>>>>>>> first commit
















## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://192.168.100.15/ModelTrains/facefeature.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://192.168.100.15/ModelTrains/facefeature/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***
# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
=======




