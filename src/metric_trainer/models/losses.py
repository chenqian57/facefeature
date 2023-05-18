import torch
import math
from pytorch_metric_learning import losses, reducers
from .partial_fc import PartialFC


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def build_metric(name, embedding_dim, num_class, sample_rate, fp16):
    reducer = reducers.MeanReducer()
    if name == "arcface":
        loss_func = losses.ArcFaceLoss(
            num_classes=num_class, embedding_size=embedding_dim, reducer=reducer
        )
    elif name == "circleloss":
        loss_func = losses.CircleLoss(m=0.25, gamma=256, reducer=reducer)
    elif name == "tripletloss":
        loss_func = losses.TripletMarginLoss(reducer=reducer)
    elif "partial_fc" in name:
        if "adaface" in name:
            margin_loss = AdaFace()
        else:
            margin_loss = CombinedMarginLoss(
                64,
                1.0,
                0.0,
                0.4,
                # 0.5, 
                # 0.0,
            )
        loss_func = PartialFC(
            margin_loss,
            embedding_dim,
            num_class,
            sample_rate,
            fp16,
        )
    return loss_func


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, s, m1, m2, m3, interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones(
                    [index_positive.size(0), logits.size(1)], device=logits.device
                )
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = (
                target_logit * self.cos_m - sin_theta * self.sin_m
            )  # cos(target+margin)
            if self.easy_margin:
                final_target_logit = torch.where(
                    target_logit > 0, cos_theta_m, target_logit
                )
            else:
                final_target_logit = torch.where(
                    target_logit > self.theta, cos_theta_m, target_logit - self.sinmm
                )
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits


class ArcFace(torch.nn.Module):
    """ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):"""

    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = (
            target_logit * self.cos_m - sin_theta * self.sin_m
        )  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit
            )
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm
            )

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


class AdaFace(torch.nn.Module):
    def __init__(
        self,
        m=0.4,
        h=0.333,
        s=64.0,
        t_alpha=1.0,
    ):
        super(AdaFace, self).__init__()
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer("t", torch.zeros(1))
        self.register_buffer("batch_mean", torch.ones(1) * (20))
        self.register_buffer("batch_std", torch.ones(1) * 100)

        # print("\n\AdaFace with the following property")
        # print("self.m", self.m)
        # print("self.h", self.h)
        # print("self.s", self.s)
        # print("self.t_alpha", self.t_alpha)

    def forward(self, logits, labels, embeddings):
        # partial fc
        index_positive = torch.where(labels != -1)[0]
        target_logit = logits[index_positive, labels[index_positive].view(-1)]
        embeddings = embeddings[index_positive]

        norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        target_logit = target_logit.clamp(-1 + self.eps, 1 - self.eps)  # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100)  # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (
            self.batch_std + self.eps
        )  # 66% between -1, 1
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        margin_scaler = margin_scaler.squeeze_()

        # g_angular
        g_angular = self.m * margin_scaler * -1
        theta = target_logit.acos()
        theta_m = torch.clip(theta + g_angular, min=self.eps, max=math.pi - self.eps)
        target_logit = theta_m.cos()

        # g_additive
        g_add = self.m + (self.m * margin_scaler)
        final_target_logit = target_logit - g_add

        logits[index_positive, labels[index_positive].view(-1)] = final_target_logit

        # scale
        logits = logits * self.s
        return logits


class AdaFaceLayers(torch.nn.Module):
    def __init__(
        self,
        embedding_size=512,
        classnum=70722,
        m=0.4,
        h=0.333,
        s=64.0,
        t_alpha=1.0,
    ):
        super(AdaFaceLayers, self).__init__()
        self.classnum = classnum
        self.kernel = torch.nn.Parameter(torch.Tensor(embedding_size, classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer("t", torch.zeros(1))
        self.register_buffer("batch_mean", torch.ones(1) * (20))
        self.register_buffer("batch_std", torch.ones(1) * 100)

        print("\n\AdaFace with the following property")
        print("self.m", self.m)
        print("self.h", self.h)
        print("self.s", self.s)
        print("self.t_alpha", self.t_alpha)

    def forward(self, embeddings, label):
        norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True)

        kernel_norm = l2_norm(self.kernel, axis=0)
        embeddings = torch.div(embeddings, norms)

        cosine = torch.mm(embeddings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)  # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100)  # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (
            self.batch_std + self.eps
        )  # 66% between -1, 1
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m
