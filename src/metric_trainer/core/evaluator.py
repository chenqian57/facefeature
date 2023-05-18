from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import List
import torch
import sklearn
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from ..utils.pair import parse_pair
from ..utils.dist import get_rank
from ..data.dataset import FaceValData, ValBinData
from ..eval.verification import evaluate


from scipy import interpolate
from sklearn.model_selection import KFold


# class Evalautor:
#     def __init__(self, pair_path, batch_size, img_size=112) -> None:
#         imgl, imgr, flags, folds = parse_pair(pair_path=pair_path)
#         self.flags = flags
#         self.folds = folds
#         self.dataset = FaceValData(imgl, imgr, img_size)
#         self.val_loader = DataLoader(self.dataset, batch_size)
#
#     def eval(self, model):
#         model.cuda()
#         pbar = tqdm(self.val_loader, total=len(self.val_loader))
#         for imgl, imgr in pbar:
#             imgl = imgl.cuda()
#             imgl = imgl / 255.
#             imgr = imgr.cuda()
#             imgr = imgr / 255.
#
#     def single_inference(self, img, model):
#         img = img / 255.


class Evalautor:
    """
    Eval image pairs from prepared *.bin file which download from insightface repo.
    """

    def __init__(
        self, val_targets, root_dir, image_size=112, batch_size=8, flip=True
    ) -> None:
        self.rank: int = get_rank()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.var_data_list: List[object] = []
        self.val_issame_list: List[object] = []
        self.var_name_list: List[str] = []
        self.flip = flip
        if self.rank == 0:
            self.init_dataset(
                val_targets=val_targets,
                data_dir=root_dir,
                image_size=image_size,
                batch_size=batch_size,
            )





    def init_dataset(self, val_targets, data_dir, image_size, batch_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                logger.info(f"loading val data {name}...")


                
                valdataset = ValBinData(path, image_size, rgb=True)
                valdataloader = DataLoader(
                    dataset=valdataset,
                    batch_size=batch_size,
                    # num_workers=nw,
                    shuffle=False,
                )
                self.var_data_list.append(valdataloader)
                self.val_issame_list.append(valdataset.issame_list)
                self.var_name_list.append(name)
                logger.info(f"load {name}: {len(valdataset) // 2} image pairs Done!")
            else:
                logger.info(f"Not found {name}!")







    # # test
    # def val(self, model, nfolds=10, flip=True):
    #     model.cuda()
    #     accs, stds = [], []
    #     for i, val_data in enumerate(self.var_data_list):
    #         acc, std = self.val_one_data(
    #             val_data,
    #             self.val_issame_list[i],
    #             model,
    #             flip,
    #             nfolds,
    #         )
    #         accs.append(acc)
    #         stds.append(std)
    #         pf = "%20s" + "%20s" * 1
    #         print(pf % (self.var_name_list[i], '%1.5f+-%1.5f' % (acc, std)))
    #     return accs, stds






    # test
    def val(self, model, nfolds=10, flip=True):
        model.cuda()
        accs, stds , best_thresholds = [], [], []
        for i, val_data in enumerate(self.var_data_list):
            acc, std, best_threshold = self.val_one_data(
                val_data,
                self.val_issame_list[i],
                model,
                flip,
                nfolds,
            )

            # 将一个名为acc的变量的值添加到名为accs的列表中
            accs.append(acc)
            stds.append(std)
            best_thresholds.append(best_threshold)

            best_threshold2 = best_threshold/4

            # 定义了一个格式化字符串pf，其中包含两个占位符%20s和%20s，
            # 第一个占位符是用来输出字符串类型的变量值（比如self.var_name_list[i]），它的宽度为20个字符；
            # 第二个占位符用来输出数值类型的变量值（比如acc和std），它的宽度也为20个字符。其中，* 1 表示第二个占位符重复1次，即只需要一个。
            pf = "%20s" + "%20s" + "%20s" * 1
            print(pf % (self.var_name_list[i], '%1.5f+-%1.5f' % (acc, std), '%1.5f' % best_threshold2 ))
            # print('Best_thresholds: %2.5f' % best_thresholds)
        return accs, stds, best_thresholds














   # def evaluate
    def val_one_data(self, val_data, issame_list, model, flip, nfolds):
        # 五个参数，分别是val_data，issame_list，model，flip，和nfolds
        # 该方法的作用是 评估模型 在 验证集 上的性能，并返回 平均准确率 和 标准差。
        # 使用了torch和numpy库，其中torch是深度学习框架，numpy是用于数值计算的Python库。
       
        # 字符串"Eval"和"Accuracy"会被格式化为20个字符的字符串，用于显示进度条的标签，flip为True时在"Accuracy"后面添加"-Flip"
        s = ("%20s" + "%20s" + "%20s"  * 1) % (
            f"Eval",
            f"Accuracy{'-Flip' if flip else ''}",
            f"Best_thresholds",
        )


        # 定义一个空列表embeddings，用于存储每个样本的嵌入向量
        embeddings = []



        #    ###########################arcface_20220803##########################################
        # # 对于每个样本，将其转换为CUDA Tensor，并将其除以255进行归一化。
        # # 如果flip为True，将样本的水平翻转添加到样本集合中，形成一个大小为2N的张量。
        # # 然后，将样本输入模型中，得到512维的嵌入向量，
        # # 如果flip为True，将得到的嵌入向量按照原始和翻转的顺序拆分为两个大小为N的嵌入向量，
        # # 再将这两个嵌入向量相加，形成一个2xNx512的张量。
        # pbar = tqdm(enumerate(val_data), total=len(val_data))
        # for i, imgs in pbar:
        #     imgs = imgs.cuda()
        #     imgs = imgs / 255.0
        #     if flip:
        #         imgs = torch.cat([imgs, imgs.flip(dims=[-1])], dim=0)  # (2N, C, H, W)
        #     out = model(imgs)  # (N, 512)
        #     if flip:
        #         out = torch.split(out, out.shape[0] // 2, dim=0)  # ((N, 512), (N, 512))
        #         out = out[0] + out[1]  # (2, N, 512)


        #     # normalize
        #     # 对得到的嵌入向量进行归一化，并将其转换为Numpy数组类型。
        #     # 将每个样本的嵌入向量追加到列表embeddings中
        #     out = normalize(out)
        #     embedding = out.detach().cpu().numpy()

        #     # 对列表embeddings中的所有嵌入向量进行拼接，
        #     # 得到一个大小为Nx512的Numpy数组embeddings















        pbar = tqdm(enumerate(val_data), total=len(val_data))
        for i, images in pbar:
            imgs, srcs = images

            ###########################arcface##########################################
            imgs = imgs.cuda()


            imgs = imgs.float()
            imgs.div_(255).sub_(0.5).div_(0.5)
            out = model(imgs)


            # if flip:
            #     imgs = torch.cat([imgs, imgs.flip(dims=[-1])], dim=0)  # (2N, C, H, W)
            # out = model(imgs)  # (N, 512)
            # if flip:
            #     out = torch.split(out, out.shape[0] // 2, dim=0)  # ((N, 512), (N, 512))
            #     out = out[0] + out[1]  # (2, N, 512)




            out = normalize(out)
            embedding = out.detach().cpu().numpy()









            embeddings.append(embedding)
            pbar.set_description(s)
        embeddings = np.concatenate(embeddings, axis=0)

        # 使用evaluate方法对embeddings和issame_list进行比较，返回评估结果，其中nrof_folds参数指定交叉验证的折数。
        _, _, accuracy, _, _, _, best_thresholds = evaluate(embeddings, issame_list, nrof_folds=nfolds)
        
        # 计算评估结果的平均准确率和标准差，并返回。
        # np.mean(accuracy) 计算了 accuracy 数组中所有元素的平均值，即将所有元素相加并除以数组长度。然后，这个平均值被存储在变量 acc 中
        # np.std(accuracy) 计算了 accuracy 数组中所有元素的标准差，即一组数据的离散程度。标准差表示数据值偏离平均值的程度。标准差越小，表示数据越接近平均值。然后，这个标准差被存储在变量 std 中。
        acc, std = np.mean(accuracy), np.std(accuracy)
        return acc, std, best_thresholds





    # # def test改
    # def val(self, model, png_save_path, nfolds=10, flip=True ):

    #     model.cuda()
    #     accuracy, val_std, best_thresholds = [], []
    #     # for i, val_data in enumerate(self.var_data_list):


    #     # labels      = np.array([sublabel for label in labels for sublabel in label])
    #     # distances   = np.array([subdist for dist in distances for subdist in dist])
    
    #     for i, val_data in enumerate(self.var_data_list):
    #         acc, std = self.val_one_data(
    #             val_data,
    #             self.val_issame_list[i],
    #             model,
    #             flip,
    #             nfolds,
    #         )
    #         tpr, fpr, accuracy, val, val_std, far, best_thresholds = self.val_one_data(distances,labels)
        
    #     accuracy.append(accuracy)
    #     val_std.append(val_std)
    #     best_thresholds.append(best_thresholds)
    #     pf = "%20s" + "%20s" * 1
    #     print(pf % (self.var_name_list[i], '%1.5f+-%1.5f' % (acc, std)))
    #     return accuracy, val_std, best_thresholds
        







        # print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        # print('Best_thresholds: %2.5f' % best_thresholds)
        # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        # self.plot_roc(fpr, tpr, figure_name = png_save_path)




        # for i, val_data in enumerate(self.var_data_list):
        #     acc, std = self.val_one_data(
        #         val_data,
        #         self.val_issame_list[i],
        #         model,
        #         flip,
        #         nfolds,
        #     )





        #     accs.append(acc)
        #     stds.append(std)
        #     pf = "%20s" + "%20s" * 1
        #     print(pf % (self.var_name_list[i], '%1.5f+-%1.5f' % (acc, std)))
        # return accs, stds
















    def plot_roc(fpr, tpr, figure_name = "roc.png"):
        import matplotlib.pyplot as plt
        from sklearn.metrics import auc, roc_curve
        roc_auc = auc(fpr, tpr)
        fig = plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        fig.savefig(figure_name, dpi=fig.dpi)










    
    
    # # def evaluate 改
    # def val_one_data(self, val_data, issame_list, model, distances, labels,flip, nfolds):
        
    #     thresholds = np.arange(0, 4, 0.01)

    #     tpr, fpr, accuracy, best_thresholds = self.calculate_roc(thresholds, distances,
    #     labels, nrof_folds=nfolds)


    #     thresholds = np.arange(0, 4, 0.001)

    #     # calculate_val函数接受五个参数：thresholds、distances、labels、far和nrof_folds（默认值为10）
    #     # 它计算了验证率（Verification Rate）、验证率标准差（Verification Rate Standard Deviation）和虚警率（False Alarm Rate）
    #     val, val_std, far = self.calculate_val(thresholds, distances,
    #     labels, 1e-3, nrof_folds=nfolds)

    #     # 函数的返回值是一个元组，包含了计算出来的评估指标：真正率、假正率、准确率、验证率、验证率标准差、虚警率和最佳阈值
    #     return tpr, fpr, accuracy, val, val_std, far, best_thresholds

  
        # s = ("%20s" + "%20s" * 1) % (
        #     f"Eval",
        #     f"Accuracy{'-Flip' if flip else ''}",
        # )



        
        # embeddings = []
        # pbar = tqdm(enumerate(val_data), total=len(val_data))
        # for i, imgs in pbar:
        #     imgs = imgs.cuda()
        #     imgs = imgs / 255.0
        #     if flip:
        #         imgs = torch.cat([imgs, imgs.flip(dims=[-1])], dim=0)  # (2N, C, H, W)
        #     out = model(imgs)  # (N, 512)
        #     if flip:
        #         out = torch.split(out, out.shape[0] // 2, dim=0)  # ((N, 512), (N, 512))
        #         out = out[0] + out[1]  # (2, N, 512)
        #     # normalize
        #     out = normalize(out)
        #     embedding = out.detach().cpu().numpy()
        #     embeddings.append(embedding)
        #     pbar.set_description(s)
        # embeddings = np.concatenate(embeddings, axis=0)
        # _, _, accuracy, _, _, _ = evaluate(embeddings, issame_list, nrof_folds=nfolds)
        # acc, std = np.mean(accuracy), np.std(accuracy)
        # return acc, std
        # 
    
    














#     def calculate_roc(self, thresholds, distances, labels, nrof_folds=10):

#         nrof_pairs = min(len(labels), len(distances))
#         nrof_thresholds = len(thresholds)
#         k_fold = KFold(n_splits=nrof_folds, shuffle=False)

#         tprs = np.zeros((nrof_folds,nrof_thresholds))
#         fprs = np.zeros((nrof_folds,nrof_thresholds))
#         accuracy = np.zeros((nrof_folds))

#         indices = np.arange(nrof_pairs)

#         for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

#             # Find the best threshold for the fold
#             acc_train = np.zeros((nrof_thresholds))
#             for threshold_idx, threshold in enumerate(thresholds):
#                 _, _, acc_train[threshold_idx] = self.calculate_accuracy(threshold, distances[train_set], labels[train_set])

#             best_threshold_index = np.argmax(acc_train)
#             for threshold_idx, threshold in enumerate(thresholds):
#                 tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = self.calculate_accuracy(threshold, distances[test_set], labels[test_set])
#             _, _, accuracy[fold_idx] = self.calculate_accuracy(thresholds[best_threshold_index], distances[test_set], labels[test_set])
#             tpr = np.mean(tprs,0)
#             fpr = np.mean(fprs,0)


#     # 函数最终返回四个数组，分别为：
#     # tpr: 平均TPR的数组
#     # fpr: 平均FPR的数组
#     # accuracy: 每个折叠的准确率数组
#     # thresholds[best_threshold_index]: 最佳阈值
#             return tpr, fpr, accuracy, thresholds[best_threshold_index]








#     def calculate_accuracy(threshold, dist, actual_issame):
#         predict_issame = np.less(dist, threshold)
#         tp = np.sum(np.logical_and(predict_issame, actual_issame))
#         fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
#         tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
#         fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

#         tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
#         fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
#         acc = float(tp+tn)/dist.size
#         return tpr, fpr, acc






# # thresholds、distances、labels、far_target（默认值为1e-3）和nrof_folds（默认值为10）
# # 函数的主要功能是计算验证率（Verification Rate）、验证率标准差（Verification Rate Standard Deviation）和虚警率（False Alarm Rate）
#     def calculate_val(self, thresholds, distances, labels, far_target=1e-3, nrof_folds=10):

#     # 计算pairs的数量（labels和distances的最小长度）以及阈值的数量。
#         nrof_pairs = min(len(labels), len(distances))
#         nrof_thresholds = len(thresholds)

#         # 通过KFold方法将数据集分成nrof_folds个折叠，每个折叠用于验证模型性能。
#         k_fold = KFold(n_splits=nrof_folds, shuffle=False)

#         # 初始化val和far两个数组，用于存储每个折叠（fold）的验证率和虚警率。
#         val = np.zeros(nrof_folds)
#         far = np.zeros(nrof_folds)

#         indices = np.arange(nrof_pairs)

#         for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

#         # Find the threshold that gives FAR = far_target
#         # 对于每个折叠，计算训练集的far_train，以找到使得FAR=far_target的阈值。
#             far_train = np.zeros(nrof_thresholds)
#             for threshold_idx, threshold in enumerate(thresholds):
#                 _, far_train[threshold_idx] = self.calculate_val_far(threshold, distances[train_set], labels[train_set])
        
#         # 如果 far_train 的最大值大于等于far_target，则使用插值法（slinear）找到使得 fdFAR=far_target的阈值。
#             if np.max(far_train)>=far_target:
#                 f = interpolate.interp1d(far_train, thresholds, kind='slinear')
#                 threshold = f(far_target)
#             else:
#                 # 否则，将阈值设置为0.0
#                 threshold = 0.0


#             # 函数中使用了一个辅助函数：calculate_val_far。calculate_val_far函数接受三个参数：threshold、distances和labels。
#             # 它计算了给定阈值下的验证率和虚警率。
#             # 对于测试集，计算给定阈值下的验证率和虚警率，并将结果存储到val和far数组中
#             val[fold_idx], far[fold_idx] = self.calculate_val_far(threshold, distances[test_set], labels[test_set])


#         # 计算所有折叠的验证率均值、虚警率均值和验证率标准差，并返回这三个值作为函数的输出。
#         val_mean = np.mean(val)
#         far_mean = np.mean(far)
#         val_std = np.std(val)
#         return val_mean, val_std, far_mean







# # 计算计算验证 和 假识别率
# # 接受三个参数，分别是 threshold、dist 和 actual_issame
# # threshold 是一个阈值，它用于决定两张人脸是否属于同一人
# # dist 是一个数组，它包含了所有人脸对之间的距离
# # actual_issame 是一个布尔类型的数组，它指示每个人脸对是否属于同一人
#     def calculate_val_far(threshold, dist, actual_issame):

#         # 首先通过比较 dist 和 threshold 来得到 predict_issame 数组，该数组表示每个人脸对是否被分类为同一人。
#         predict_issame = np.less(dist, threshold)

#         # 计算出 true_accept 和 false_accept 两个变量，它们分别表示 正确接受 和 错误接受 的人脸对的数量
#         true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
#         false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

#         # 计算出 n_same 和 n_diff，它们分别表示属于同一人和不同人的人脸对的数量
#         n_same = np.sum(actual_issame)
#         n_diff = np.sum(np.logical_not(actual_issame))

#         # 检查 n_diff 和 n_same 是否为零，并在它们为零时对它们进行修正，以避免除以零错误。
#         if n_diff == 0:
#             n_diff = 1
#         if n_same == 0:
#             return 0,0

#         # 函数计算出验证率（val）和假识别率（far），并将它们作为一个元组返回。
#         # val 表示被正确接受的同一人脸对的比率，far 表示被错误接受的不同人脸对的比率。
#         val = float(true_accept) / float(n_same)
#         far = float(false_accept) / float(n_diff)
#         return val, far











































 