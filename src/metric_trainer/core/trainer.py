from tqdm import tqdm
from torch import optim
from omegaconf import OmegaConf

# from loguru import logger
import torch
import time
import torch.nn as nn
import numpy as np
import os.path as osp
import os
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from .evaluator import Evalautor
from ..data.dataset import FaceTrainData, Glint360Data, get_dataloader
from ..utils.callbacks import CallBackSaveLog
from ..utils.lr_scheduler import PolyScheduler
from ..utils.dist import get_world_size, get_rank
from ..utils.metric import AverageMeter
from ..utils.plots import plot_results
from ..utils.general import colorstr, strip_optimizer
from ..models import build_metric, build_model

from ..utils.logger import setup_logger


def build_dataset(data, *args, **kwargs):
    if data == "glint360k":
        return Glint360Data(*args, **kwargs)
    elif data == "folder":
        return FaceTrainData(*args, **kwargs)


class Trainer:
    def __init__(self, cfg, local_rank) -> None:
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = local_rank

        self.cfg = cfg
        self.backbone = cfg.MODEL.BACKBONE
        self.embedding_dim = cfg.MODEL.EMBEDDING_DIM
        self.num_classes = cfg.MODEL.NUM_CLASS
        # batch size for one gpu
        self.batch_size = cfg.SOLVER.BATCH_SIZE_PER_GPU
        self.fp16 = cfg.SOLVER.FP16
        self.max_epoch = cfg.SOLVER.NUM_EPOCH
        self.save_dir = cfg.OUTPUT
        self.img_size = cfg.DATASET.IMG_SIZE
        self.resume_dir = cfg.get("RESUME_DIR", None)

        self.dataset = build_dataset(
            cfg.DATASET.TYPE, cfg.DATASET.TRAIN, cfg.DATASET.TRANSFORM, self.img_size
        )
        self.best_fitness = 0
        self.start_epoch = 0
        self.last = osp.join(self.save_dir, "last.pt")
        self.best = osp.join(self.save_dir, "best.pt")
        self.partial_fc = "partial_fc" in cfg.MODEL.LOSS

        # class-aware args
        self.sample_rate = cfg.MODEL.SAMPLE_RATE

    def before_train(self):
        setup_logger(
            self.save_dir,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )
        logger.info(colorstr("Creating Model: ") + f"{self.backbone}...")
        model = build_model(self.cfg.MODEL)
        logger.info(
            colorstr("Creating Loss Function and Optimizer: ")
            + f"{self.cfg.MODEL.LOSS}..."
        )
        loss_func = build_metric(
            self.cfg.MODEL.LOSS,
            self.embedding_dim,
            self.num_classes,
            self.sample_rate,
            self.fp16,
        )
        self.optimizer = optim.SGD(
            params=[
                {"params": model.parameters()},
                {"params": loss_func.parameters()},
            ],
            lr=self.cfg.SOLVER.BASE_LR,
            momentum=self.cfg.SOLVER.MOMENTUM,
            weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
        )
        model, loss_func = self.resume_train(model, loss_func)
        model = model.cuda()
        loss_func = loss_func.cuda()
        if self.is_distributed:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                broadcast_buffers=False,
                bucket_cap_mb=16,
                find_unused_parameters=True,
            )
            model._set_static_graph()
        self.model = model
        self.model.train()
        self.loss_func = loss_func
        self.loss_func.train()

        logger.info(colorstr("Creating DataLoader: ") + f"{self.cfg.DATASET.TYPE}...")
        self.train_loader = get_dataloader(
            self.dataset, self.is_distributed, self.batch_size, self.cfg.NUM_WORKERS
        )
        self.max_iter = len(self.train_loader)

        self.scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

        total_batch_size = self.cfg.SOLVER.BATCH_SIZE_PER_GPU * get_world_size()
        # steps of one epoch
        steps = self.cfg.DATASET.NUM_IMAGES // total_batch_size
        warmup_step = steps * self.cfg.SOLVER.WARMUP_EPOCH
        total_step = steps * self.cfg.SOLVER.NUM_EPOCH
        # NOTE:this lr_scheduler reduce lr by steps, not epochs
        self.lr_scheduler = PolyScheduler(
            optimizer=self.optimizer,
            base_lr=self.cfg.SOLVER.BASE_LR,
            max_steps=total_step,
            warmup_steps=warmup_step,
        )
        self.lr_scheduler.last_epoch = self.start_epoch * steps - 1  # do not move

        logger.info(
            colorstr("Creating Evalautor: ") + f"{self.cfg.DATASET.VAL_TARGETS}..."
        )
        self.evaluator = Evalautor(
            val_targets=self.cfg.DATASET.VAL_TARGETS,
            root_dir=self.cfg.DATASET.VAL,
            batch_size=self.batch_size,
        )
        self.loss_am = AverageMeter()

        self.callback = CallBackSaveLog(
            save_dir=self.save_dir, val_targets=self.cfg.DATASET.VAL_TARGETS
        )

        os.makedirs(self.save_dir, exist_ok=True)

        with open(osp.join(self.save_dir, "cfg.yaml"), "w") as f:
            OmegaConf.save(self.cfg, f)
        logger.info(colorstr("Train: ") + "Begin training...")
        self.ts = time.time()

    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()

    def after_train(self):
        if self.rank != 0:
            return
        self.te = time.time()
        if osp.exists(self.save_dir):
            plot_results(
                file=osp.join(self.save_dir, "results.csv")
            )  # save results.png
        logger.info(
            colorstr("Train: ")
            + f"Finish training with {(self.te - self.ts) / 3600:.3f} hours..."
        )
        for f in [self.last, self.best]:
            if not osp.exists(f):
                continue
            strip_optimizer(f)  # strip optimizers
        # TODO
        with open(osp.join(self.save_dir, "train.txt"), "a") as f:
            f.write(f"Finish training with {(self.te - self.ts) / 3600:.3f} hours...")

    def before_epoch(self):
        self.model.train()
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(self.epoch)

        # message
        s = ("\n" + "%10s" * 5) % ("Epoch", "gpu_mem", "loss", "lr", "img_size")
        logger.info(s)
        self.pbar = enumerate(self.train_loader)
        if self.rank == 0:
            self.pbar = tqdm(self.pbar, total=len(self.train_loader))

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def after_epoch(self):
        self.save_loss()
        if self.rank != 0:
            return
        self.model.eval()
        with torch.no_grad():
            accs, stds = self.evaluator.val(self.model, flip=True)

        log_vals = [
            self.epoch,
            self.img_size,
            self.lr_scheduler.get_last_lr()[0],
            self.loss_am.avg,
        ] + accs
        self.callback(log_vals)

        fi = np.array(accs).mean()
        if fi > self.best_fitness:
            self.best_fitness = fi
        self.save_ckpt(save_path=self.last, best=self.best_fitness == fi)

    def train_in_iter(self):
        for self.iter, (imgs, labels) in self.pbar:
            imgs = imgs.cuda()
            imgs = imgs / 255.0
            labels = labels.cuda()

            embeddings = self.model(imgs)
            loss = self.compute_loss(embeddings, labels)
            self.optimizer.zero_grad()
            if self.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

            self.lr_scheduler.step()
            with torch.no_grad():
                self.loss_am.update(loss.item())
                self.after_iter()

    def after_iter(self):
        if self.rank != 0:
            return
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        self.pbar.set_description(
            (
                ("%10s" * 2 + "%10.4g" * 3)
                % (
                    f"{self.epoch}/{self.max_epoch - 1}",
                    mem,
                    self.loss_am.avg,
                    self.lr_scheduler.get_last_lr()[0],
                    self.img_size,
                )
            )
        )

    def save_ckpt(self, save_path, best=False):
        model = (
            self.model.module
            if type(self.model)
            in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
            else self.model
        )
        ckpt = {
            "epoch": self.epoch,
            "model": model.state_dict(),
            "best_fitness": self.best_fitness,
            "optimizer": self.optimizer.state_dict(),
        }
        if self.rank == 0:
            torch.save(ckpt, save_path)
            if best:
                torch.save(ckpt, self.best)

    def save_loss(self):
        if self.partial_fc:
            path_pfc = osp.join(self.save_dir, "softmax_fc_gpu_{}.pt".format(self.rank))
            torch.save(self.loss_func.state_dict(), path_pfc)
        elif self.rank == 0:
            path_loss = osp.join(self.save_dir, "loss_func.pt")
            torch.save(self.loss_func.state_dict(), path_loss)

    def resume_train(self, model, loss):
        if self.resume_dir is None or len(self.resume_dir) == 0:
            return model, loss
        logger.info(colorstr("resume: ") + f"Resuming from {self.resume_dir}...")
        model_path = osp.join(self.resume_dir, "last.pt")
        ckpt = torch.load(model_path)
        # Optimizer
        if ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_fitness = ckpt["best_fitness"]

        # Epochs
        self.start_epoch = ckpt["epoch"] + 1
        assert (
            self.start_epoch > 0
        ), f"{model_path} training to {self.epochs} epochs is finished, nothing to resume."
        model.load_state_dict(ckpt["model"], strict=False)  # load
        del ckpt

        loss_path = osp.join(self.resume_dir, f"softmax_fc_gpu_{self.rank}.pt")
        ckpt = torch.load(loss_path)
        loss.load_state_dict(ckpt)
        del ckpt
        return model, loss

    def compute_loss(self, embeddings, labels):
        if self.partial_fc:
            return self.loss_func(embeddings, labels, self.optimizer)
        return self.loss_func(embeddings, labels)

    @property
    def global_iter(self):
        return self.iter + self.max_iter * self.epoch
