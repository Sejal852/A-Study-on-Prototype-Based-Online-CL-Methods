"""OnPro implementation using Avalanche 0.6 with rotation-based augmentation, OPE loss, APF sampling,
and SlimResNet18 ('slimnet') backbone from avalanche.models."""

import random
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Replay

# NEW: import SlimResNet18 ("slimnet") from Avalanche
from avalanche.models import SlimResNet18  # Avalanche 0.6.x slimmed ResNet-18


# ---------------------------------------------------------------------------
# Functions and layers from utils and modules
# ---------------------------------------------------------------------------

def rot_inner_all(x):
    num = x.shape[0]
    c, h, w = x.shape[1], x.shape[2], x.shape[3]

    R = x.repeat(4, 1, 1, 1)
    a = x.permute(0, 1, 3, 2)
    a = a.view(num, c, 2, h // 2, w)

    a = a.permute(2, 0, 1, 3, 4)

    s1 = a[0]
    s2 = a[1]
    s1_1 = torch.rot90(s1, 2, (2, 3))
    s2_2 = torch.rot90(s2, 2, (2, 3))

    R[num:2 * num] = (
        torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
        .reshape(num, c, h, w)
        .permute(0, 1, 3, 2)
    )
    R[3 * num:] = (
        torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2)
        .reshape(num, c, h, w)
        .permute(0, 1, 3, 2)
    )
    R[2 * num:3 * num] = (
        torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2)
        .reshape(num, c, h, w)
        .permute(0, 1, 3, 2)
    )
    return R


def Rotation(x):
    X = rot_inner_all(x)
    return torch.cat(
        (
            X,
            torch.rot90(X, 2, (2, 3)),
            torch.rot90(X, 1, (2, 3)),
            torch.rot90(X, 3, (2, 3)),
        ),
        dim=0,
    )


class RandomResizedCropLayer(nn.Module):
    def __init__(self, size=None, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        super().__init__()
        _eye = torch.eye(2, 3)
        self.size = size
        self.register_buffer("_eye", _eye)
        self.scale = scale
        self.ratio = ratio

    def forward(self, inputs, whbias=None):
        _device = inputs.device
        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)

        if whbias is None:
            whbias = self._sample_latent(inputs)

        _theta[:, 0, 0] = whbias[:, 0]
        _theta[:, 1, 1] = whbias[:, 1]
        _theta[:, 0, 2] = whbias[:, 2]
        _theta[:, 1, 2] = whbias[:, 3]

        grid = F.affine_grid(_theta, inputs.size(), align_corners=False).to(_device)
        grid = torch.tensor(grid, dtype=inputs.dtype).to(_device)
        output = F.grid_sample(inputs, grid, padding_mode="reflection", align_corners=False)

        if self.size is not None:
            h, w, c = self.size
            output = F.adaptive_avg_pool2d(output, (h, w))
        return output

    def _sample_latent(self, inputs):
        _device = inputs.device
        N, _, width, height = inputs.shape

        area = width * height
        target_area = np.random.uniform(*self.scale, N * 10) * area
        log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
        aspect_ratio = np.exp(np.random.uniform(*log_ratio, N * 10))

        w = np.round(np.sqrt(target_area * aspect_ratio))
        h = np.round(np.sqrt(target_area / aspect_ratio))
        cond = (0 < w) * (w <= width) * (0 < h) * (h <= height)
        w = w[cond]
        h = h[cond]
        cond_len = w.shape[0]
        if cond_len >= N:
            w = w[:N]
            h = h[:N]
        else:
            w = np.concatenate([w, np.ones(N - cond_len) * width])
            h = np.concatenate([h, np.ones(N - cond_len) * height])

        w_bias = np.random.randint(w - width, width - w + 1) / width
        h_bias = np.random.randint(h - height, height - h + 1) / height
        w = w / width
        h = h / height

        whbias = np.column_stack([w, h, w_bias, h_bias])
        whbias = torch.tensor(whbias, device=_device)

        return whbias


class HorizontalFlipLayer(nn.Module):
    def __init__(self):
        super().__init__()
        _eye = torch.eye(2, 3)
        self.register_buffer("_eye", _eye)

    def forward(self, inputs):
        _device = inputs.device
        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)
        r_sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        _theta[:, 0, 0] = r_sign
        grid = F.affine_grid(_theta, inputs.size(), align_corners=False).to(_device)
        grid = torch.tensor(grid, dtype=inputs.dtype).to(_device)
        inputs = F.grid_sample(inputs, grid, padding_mode="reflection", align_corners=False)
        return inputs


class RandomColorGrayLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.prob = p
        _weight = torch.tensor([[0.299, 0.587, 0.114]])
        self.register_buffer("_weight", _weight.view(1, 3, 1, 1))

    def forward(self, inputs, aug_index=None):
        if aug_index == 0:
            return inputs

        l = F.conv2d(inputs, self._weight)
        gray = torch.cat([l, l, l], dim=1)

        if aug_index is None:
            _prob = inputs.new_full((inputs.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            gray = inputs * (1 - _mask) + gray * _mask

        return gray


class OPELoss(nn.Module):
    def __init__(self, class_per_task, temperature=0.5, only_old_proto=False):
        super().__init__()
        self.class_per_task = class_per_task
        self.temperature = temperature
        self.only_old_proto = only_old_proto

    def cal_prototype(self, z1, z2, y, current_task_id):
        start_i = 0
        end_i = (current_task_id + 1) * self.class_per_task
        dim = z1.shape[1]
        current_classes_mean_z1 = torch.zeros((end_i, dim), device=z1.device)
        current_classes_mean_z2 = torch.zeros((end_i, dim), device=z1.device)
        for i in range(start_i, end_i):
            indices = y == i
            if not any(indices):
                continue
            t_z1 = z1[indices]
            t_z2 = z2[indices]
            mean_z1 = torch.mean(t_z1, dim=0)
            mean_z2 = torch.mean(t_z2, dim=0)
            current_classes_mean_z1[i] = mean_z1
            current_classes_mean_z2[i] = mean_z2

        return current_classes_mean_z1, current_classes_mean_z2

    def forward(self, z1, z2, labels, task_id, is_new=False):
        prototype_z1, prototype_z2 = self.cal_prototype(z1, z2, labels, task_id)

        if not self.only_old_proto or is_new:
            non_zero_rows = torch.abs(prototype_z1).sum(dim=1) > 0
            non_zero_proto_z1 = prototype_z1[non_zero_rows]
            non_zero_proto_z2 = prototype_z2[non_zero_rows]
        else:
            old_z1 = prototype_z1[: task_id * self.class_per_task]
            old_z2 = prototype_z2[: task_id * self.class_per_task]
            non_zero_rows = torch.abs(old_z1).sum(dim=1) > 0
            non_zero_proto_z1 = old_z1[non_zero_rows]
            non_zero_proto_z2 = old_z2[non_zero_rows]

        if non_zero_proto_z1.size(0) == 0:
            return torch.tensor(0.0, device=z1.device), prototype_z1, prototype_z2

        non_zero_proto_z1 = F.normalize(non_zero_proto_z1)
        non_zero_proto_z2 = F.normalize(non_zero_proto_z2)

        device = non_zero_proto_z1.device

        class_num = non_zero_proto_z1.size(0)
        z = torch.cat((non_zero_proto_z1, non_zero_proto_z2), dim=0)

        logits = torch.einsum("if, jf -> ij", z, z) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        pos_mask = torch.zeros((2 * class_num, 2 * class_num), dtype=torch.bool, device=device)
        pos_mask[:, class_num:].fill_diagonal_(True)
        pos_mask[class_num:, :].fill_diagonal_(True)

        logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        loss = -mean_log_prob_pos.mean()

        return loss, prototype_z1, prototype_z2


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


# ---------------------------------------------------------------------------
# loss functions
# ---------------------------------------------------------------------------

def Supervised_NT_xent_n(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8):
    device = sim_matrix.device
    labels1 = labels.repeat(2)
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()
    B = sim_matrix.size(0) // chunk
    eye = torch.zeros((B * chunk, B * chunk), dtype=torch.bool, device=device)
    eye[:, :].fill_diagonal_(True)
    sim_matrix = torch.exp(sim_matrix / temperature) * (~eye)
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)
    labels1 = labels1.contiguous().view(-1, 1)
    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    loss2 = 2 * torch.sum(Mask1 * sim_matrix) / (2 * B)
    loss1 = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)
    return loss1 + loss2


# ---------------------------------------------------------------------------
# model (SlimResNet18 wrapper with projection head)
# ---------------------------------------------------------------------------

class OnProSlimNet(nn.Module):
    """
    Wraps Avalanche SlimResNet18 to:
    - keep a 'classifier' attribute (read-only property) compatible with your strategy
    - expose a projection head for contrastive losses
    - support forward(..., use_proj=True) -> (feats, proj)
    """
    def __init__(self, num_classes: int, nf: int = 20, proj_dim: int = 128):
        super().__init__()
        self.backbone = SlimResNet18(nclasses=num_classes, nf=nf)
        # Use the backbone head's input size for the projection layer
        feat_dim = self.backbone.linear.in_features
        self.proj_head = nn.Linear(feat_dim, proj_dim)

    @property
    def classifier(self) -> nn.Linear:
        # Expose a classifier attribute like in your original ResNet class
        return self.backbone.linear

    def forward(self, x, use_proj: bool = False):
        # Reproduce SlimResNet18 forward up to the penultimate features
        bsz = x.size(0)
        out = F.relu(self.backbone.bn1(self.backbone.conv1(x.view(bsz, 3, 32, 32))))
        out = self.backbone.layer1(out)
        out = self.backbone.layer2(out)
        out = self.backbone.layer3(out)
        out = self.backbone.layer4(out)
        out = F.avg_pool2d(out, out.size(2))  # adaptive to spatial size
        feats = out.view(out.size(0), -1)

        if use_proj:
            proj = F.normalize(self.proj_head(feats), dim=1)
            return feats, proj
        return self.backbone.linear(feats)


# ---------------------------------------------------------------------------
# strategy
# ---------------------------------------------------------------------------

class OnProStrategy(Replay):
    def __init__(
        self,
        *,
        model,
        optimizer,
        mem_size,
        train_mb_size,
        train_epochs,
        eval_mb_size,
        device,
        evaluator,
        alpha: float = 0.25,
        mixup_alpha: float = 0.4,
        mixup_base_rate: float = 0.75,
        ins_t: float = 0.07,
        proto_t: float = 0.5,
        replay_bsize: int = 64,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            mem_size=mem_size,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            evaluator=evaluator,
        )

        self.alpha = alpha
        self.mixup_alpha = mixup_alpha
        self.mixup_base_rate = mixup_base_rate
        self.replay_bsize = replay_bsize
        self.ins_t = ins_t
        self.ope_loss = OPELoss(class_per_task=2, temperature=proto_t)

        self.oop = 16
        self.oop_base = model.classifier.out_features  # unchanged

        self.transform = nn.Sequential(
            HorizontalFlipLayer(),
            RandomColorGrayLayer(p=0.25),
            RandomResizedCropLayer(size=[32, 32, 3], scale=(0.3, 1.0)),
        ).to(self.device)
        self.rotation = Rotation

        self.prototype_distances: Optional[torch.Tensor] = None
        self.prob_distribution: Optional[torch.Tensor] = None

    def before_training_exp(self, strategy, **kwargs):
        self.cur_task = strategy.experience.current_experience
        super().before_training_exp(strategy, **kwargs)

    # ------------------------------------------------------------------
    def compute_prototype_distances(self, prototypes: torch.Tensor) -> Optional[torch.Tensor]:
        n = prototypes.size(0) // 2
        if n < 2:
            return None
        proto_avg = (prototypes[:n] + prototypes[n:]) / 2
        non_zero = torch.abs(proto_avg).sum(dim=1) > 0
        proto_avg = proto_avg[non_zero]
        if proto_avg.size(0) < 2:
            return None
        return torch.cdist(proto_avg, proto_avg, p=2)

    def compute_probability_distribution(self, distances: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if distances is None:
            return None
        probs = torch.exp(-distances ** 2)
        probs *= ~torch.eye(probs.size(0), dtype=torch.bool, device=probs.device)
        s = probs.sum()
        return probs / s if s > 0 else None

    def sample_replay_data(self, replay_buffer: Optional[AvalancheDataset]):
        if replay_buffer is None or len(replay_buffer) == 0:
            return None, None

        mb_size = self.replay_bsize
        n_apf = int(self.alpha * mb_size)
        n_uniform = mb_size - n_apf
        indices = []

        if n_apf > 0 and self.prob_distribution is not None and self.prob_distribution.numel():
            flat_probs = self.prob_distribution.reshape(-1)
            n_classes = self.prob_distribution.size(0)
            class_pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
            pair_ids = torch.multinomial(flat_probs, min(n_apf, len(class_pairs)), replacement=True)
            selected = [class_pairs[idx] for idx in pair_ids]

            class_to_ids = {}
            for idx, (_, y) in enumerate(replay_buffer):
                class_to_ids.setdefault(int(y), []).append(idx)

            for i, j in selected:
                if class_to_ids.get(i) and class_to_ids.get(j):
                    indices.extend([
                        random.choice(class_to_ids[i]),
                        random.choice(class_to_ids[j]),
                    ])

        if n_uniform > 0:
            indices.extend(random.sample(range(len(replay_buffer)), min(n_uniform, len(replay_buffer))))

        indices = list(dict.fromkeys(indices))[:mb_size]
        while len(indices) < mb_size:
            indices.append(random.randrange(len(replay_buffer)))

        x_rep = torch.stack([replay_buffer[i][0] for i in indices]).to(self.device)
        y_rep = torch.tensor([replay_buffer[i][1] for i in indices], device=self.device)
        return x_rep, y_rep

    def apply_mixup(self, x, y, x_rep, y_rep):
        if x_rep is None or random.random() > self.mixup_base_rate or len(x_rep) < 2:
            return x, y

        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        batch_size = x.size(0)
        idx = torch.randperm(x_rep.size(0))[:batch_size]
        x_mix = lam * x + (1 - lam) * x_rep[idx]
        return x_mix, y

    def forward(self):
        if not self.is_training:
            self.logits = self.model(self.mb_x)
            return self.logits

        x = self.mb_x
        rot_x = self.rotation(x)
        rot_x_aug = self.transform(rot_x)
        buffer = getattr(self, "buffer", None)

        x_rep, y_rep = self.sample_replay_data(buffer)

        y_main = torch.cat([self.mb_y + self.oop_base * i for i in range(self.oop)], dim=0)

        if x_rep is not None:
            rot_rep = self.rotation(x_rep)
            rot_rep_aug = self.transform(rot_rep)
            y_rep_r = torch.cat([y_rep + self.oop_base * i for i in range(self.oop)], dim=0)
            x_mix, _ = self.apply_mixup(rot_x, y_main, rot_rep, y_rep_r)
            imgs_main = torch.cat([x_mix, rot_x_aug])
            labels_main = y_main
            imgs_rep = torch.cat([rot_rep, rot_rep_aug])
            labels_rep = y_rep_r
        else:
            imgs_main = torch.cat([rot_x, rot_x_aug])
            labels_main = y_main
            imgs_rep = None
            labels_rep = None

        all_imgs = imgs_main if imgs_rep is None else torch.cat([imgs_main, imgs_rep])
        feats, proj = self.model(all_imgs, use_proj=True)

        len_main = imgs_main.size(0)
        self.feats_main = feats[:len_main]
        self.proj_main = proj[:len_main]
        if imgs_rep is not None:
            self.feats_rep = feats[len_main:]
            self.proj_rep = proj[len_main:]
            self.labels_rep = labels_rep
        else:
            self.feats_rep = None
            self.proj_rep = None
            self.labels_rep = None

        self.labels_main = labels_main
        self.current_y_main = self.mb_y

        self.logits = self.model(self.transform(x))
        return self.logits

    def criterion(self):
        if not self.is_training:
            return F.cross_entropy(self.logits, self.mb_y)

        outputs = self.logits
        ce = F.cross_entropy(outputs, self.mb_y)

        proj_n = F.normalize(self.proj_main, dim=1)
        feat_n = F.normalize(self.feats_main, dim=1)

        dim_diff = feat_n.shape[1] - proj_n.shape[1]
        if dim_diff >= 0:
            dim_begin = torch.randperm(dim_diff + 1, device=feat_n.device)[0]
            feat_chunk = feat_n[:, dim_begin : dim_begin + proj_n.shape[1]]
        else:
            pad = proj_n.shape[1] - feat_n.shape[1]
            feat_chunk = F.pad(feat_n, (0, pad))

        sim_m = torch.matmul(proj_n, feat_chunk.t()) + torch.mm(proj_n, proj_n.t())
        ins = Supervised_NT_xent_n(sim_m, self.labels_main, temperature=self.ins_t)

        if self.proj_rep is not None:
            proj_rn = F.normalize(self.proj_rep, dim=1)
            feat_rn = F.normalize(self.feats_rep, dim=1)
            dim_diff_r = feat_rn.shape[1] - proj_rn.shape[1]
            if dim_diff_r >= 0:
                start_r = torch.randperm(dim_diff_r + 1, device=feat_rn.device)[0]
                feat_chunk_r = feat_rn[:, start_r : start_r + proj_rn.shape[1]]
            else:
                pad_r = proj_rn.shape[1] - feat_rn.shape[1]
                feat_chunk_r = F.pad(feat_rn, (0, pad_r))
            sim_m_r = torch.matmul(proj_rn, feat_chunk_r.t()) + torch.mm(proj_rn, proj_rn.t())
            ins += Supervised_NT_xent_n(sim_m_r, self.labels_rep, temperature=self.ins_t)

        B_oop = self.oop * self.mb_x.size(0)
        z = self.proj_main[:B_oop]
        zt = self.proj_main[B_oop: 2 * B_oop]
        ope, p1, p2 = self.ope_loss(
            z[: self.mb_x.size(0)],
            zt[: self.mb_x.size(0)],
            self.mb_y,
            getattr(self, "cur_task", 0),
            True
        )
        if p1 is not None:
            protos = torch.cat([p1, p2])
            self.prototype_distances = self.compute_prototype_distances(protos)
            self.prob_distribution = self.compute_probability_distribution(self.prototype_distances)

        return ce + ins + ope


# ---------------------------------------------------------------------------
# evaluation metric
# ---------------------------------------------------------------------------

class OnProAccuracyAndForgetting(PluginMetric):
    def __init__(self):
        super().__init__()
        self.best_acc = {}
        self.last_acc = {}
        self._latest_results = []
        self.exp_id = None
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0
        self.last_acc.clear()
        self._latest_results = []

    def result(self):
        return self._latest_results

    def before_eval(self, strategy):
        self.reset()

    def before_eval_exp(self, strategy):
        self.exp_id = strategy.experience.current_experience
        self.correct = 0
        self.total = 0

    def after_eval_iteration(self, strategy):
        preds = strategy.mb_output.argmax(dim=1)
        self.correct += (preds == strategy.mb_y).sum().item()
        self.total += len(strategy.mb_y)

    def after_eval_exp(self, strategy):
        acc = 100.0 * self.correct / self.total
        exp_id = self.exp_id
        self.last_acc[exp_id] = acc
        self.best_acc[exp_id] = max(self.best_acc.get(exp_id, 0.0), acc)
        mv = MetricValue(self, f"Acc_Exp{exp_id}", acc, exp_id)
        self._latest_results = [mv]
        return [mv]

    def after_eval(self, strategy):
        metrics = []
        for exp_id, curr in self.last_acc.items():
            forget = self.best_acc[exp_id] - curr
            metrics.append(MetricValue(self, f"Forget_Exp{exp_id}", forget, exp_id))
        if self.last_acc:
            avg_acc = sum(self.last_acc.values()) / len(self.last_acc)
            avg_fgt = sum(self.best_acc[e] - self.last_acc[e] for e in self.last_acc) / len(self.last_acc)
            metrics.append(MetricValue(self, "Avg_Acc_Stream", avg_acc, None))
            metrics.append(MetricValue(self, "Avg_Forget_Stream", avg_fgt, None))
        self._latest_results = metrics
        return metrics

    def __str__(self):
        return "OnProAccuracyAndForgetting"


# ---------------------------------------------------------------------------
# data and training
# ---------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

benchmark = SplitCIFAR10(
    n_experiences=5,
    seed=0,
    fixed_class_order=list(range(10)),
    train_transform=train_transform,
    eval_transform=eval_transform,
)

interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    OnProAccuracyAndForgetting(),
    loggers=[interactive_logger],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_MB_SIZE = 10
EVAL_MB_SIZE = 100
REPLAY_BATCH_SIZE = 64
INS_T = 0.07
PROTO_T = 0.5

# Parameter grid tuned for the CIFAR-10 baseline with memory size 100, 200 and 500
param_grid = {
    "lr": [5e-4],
    "train_epochs": [1],
    "alpha": [0.25],
    "mixup_alpha": [0.4],
    "mixup_base_rate": [0.75],
    "mem_size": [100, 200, 500],
}

results = []

for lr, epochs, alpha, mixup_a, mix_base, M in product(*param_grid.values()):

    if M > 100 and epochs == 2:
        continue

    print(f"\n▶ lr={lr:.0e}  ep={epochs}  α={alpha}  β={mixup_a}  mix={mix_base}  M={M}")

    # UPDATED: Use OnProSlimNet (SlimResNet18 backbone) instead of the custom ResNet
    model = OnProSlimNet(num_classes=10)  # nf defaults to 20 in SlimResNet18
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    strategy = OnProStrategy(
        model=model,
        optimizer=optimiser,
        mem_size=M,
        train_mb_size=TRAIN_MB_SIZE,
        train_epochs=epochs,
        eval_mb_size=EVAL_MB_SIZE,
        device=device,
        evaluator=eval_plugin,
        alpha=alpha,
        mixup_alpha=mixup_a,
        mixup_base_rate=mix_base,
        replay_bsize=REPLAY_BATCH_SIZE,
        ins_t=INS_T,
        proto_t=PROTO_T,
    )

    for exp in benchmark.train_stream:
        strategy.train(exp, verbose=False)
        strategy.eval(benchmark.test_stream[: exp.current_experience + 1], verbose=False)

    strategy.eval(benchmark.test_stream, verbose=False)
    metrics = strategy.evaluator.get_last_metrics()

    row = {
        "lr": lr,
        "train_epochs": epochs,
        "alpha": alpha,
        "mixup_alpha": mixup_a,
        "mixup_base_rate": mix_base,
        "mem_size": M,
        "Avg_Acc_Stream": metrics["Avg_Acc_Stream"],
        "Avg_Forget_Stream": metrics["Avg_Forget_Stream"],
    }
    for i in range(5):
        row[f"Forget_Exp{i}"] = metrics.get(f"Forget_Exp{i}", 0.0)

    results.append(row)
    print(
        f"   ↳ Avg Acc {metrics['Avg_Acc_Stream']:.2f}%  "
        f"Avg Fgt {metrics['Avg_Forget_Stream']:.2f}%"
    )
