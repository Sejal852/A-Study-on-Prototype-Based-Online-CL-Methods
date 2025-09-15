import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import avalanche as avl
from avalanche.training import CoPE
from avalanche.models import SlimResNet18
from avalanche.benchmarks import SplitCIFAR10, data_incremental_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, StreamForgetting
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandomErasing
import random
import numpy as np
import itertools

# -------------------------
# Feature extractor + projection (SlimResNet18)
# -------------------------
class SlimResNet18CoPE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.backbone = SlimResNet18(10)
        self.backbone.classifier = nn.Identity()
        # determine feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            feat = self.backbone(dummy)
        self.projection = nn.Linear(feat.shape[1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.projection(x)
        return x / torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)

# -------------------------
# CoPE with APF (prioritized sampling)
# -------------------------
class CoPEWithAPF(CoPE):
    def __init__(
        self,
        *,
        apf_temp: float = 0.2,
        **kwargs
    ):
        self.apf_temp = apf_temp
        # build CoPE criterion
        def loss_fn(emb, tgt):
            logits = emb @ self.prototypes.T / self.T
            return F.cross_entropy(logits, tgt)
        kwargs.pop('criterion', None)
        super().__init__(criterion=loss_fn, **kwargs)

    def before_training_exp(self, strategy, num_workers=0, shuffle=True, **kwargs):
        # if empty buffer, use default behavior
        buf = strategy.storage_policy.buffer
        if len(buf) == 0:
            return

        # 1) loader for current new-data batch
        new_loader = DataLoader(
            strategy.adapted_dataset,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        # 2) embed all buffer samples to compute risk
        risks = []
        with torch.no_grad():
            for x_b, y_b in DataLoader(buf, batch_size=256, num_workers=num_workers):
                z = strategy.model(x_b.to(strategy.device)).cpu()
                protos = self.prototypes.cpu()[y_b]
                cos_vals = F.cosine_similarity(z, protos, dim=1)
                risks.append((1.0 - cos_vals).clamp(min=0.0))
        risks = torch.cat(risks)

        # 3) sampling weights via softmax
        weights = torch.softmax(risks / self.apf_temp, dim=0)

        # 4) weighted sampler + memory loader
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        mem_loader = DataLoader(
            buf,
            batch_size=strategy.train_mb_size,
            sampler=sampler,
            num_workers=num_workers,
        )

        # 5) combine new-data and prioritized-memory loaders
        class CombinedLoader:
            def __init__(self, new_dl, mem_dl):
                self.new_dl = new_dl
                self.mem_dl = mem_dl

            def __iter__(self):
                self.iter_new = iter(self.new_dl)
                self.iter_mem = iter(self.mem_dl)
                return self

            def __next__(self):
                try:
                    x1, y1 = next(self.iter_new)
                except StopIteration:
                    self.iter_new = iter(self.new_dl)
                    x1, y1 = next(self.iter_new)
                try:
                    x2, y2 = next(self.iter_mem)
                except StopIteration:
                    self.iter_mem = iter(self.mem_dl)
                    x2, y2 = next(self.iter_mem)
                xb = torch.cat([x1, x2], dim=0)
                yb = torch.cat([y1, y2], dim=0)
                return xb, yb

        strategy.model.train()
        strategy.dataloader = CombinedLoader(new_loader, mem_loader)

# -------------------------
# Runner: seed, transforms, benchmark, training loop
# -------------------------
def run_cope_apf(
    seed: int,
    mem_size: int,
    n_iter: int = 1,
    apf_temp: float = 0.1
):
    # 1) set RNGs
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 10

    # 2) data augmentations
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std  = (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
        RandomErasing(p=0.25, scale=(0.02,0.2), ratio=(0.3,3.3))
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 3) benchmark
    task_scenario = SplitCIFAR10(
        n_experiences=5,
        return_task_id=False,
        fixed_class_order=list(range(n_classes)),
        train_transform=train_transform,
        eval_transform=eval_transform
    )
    benchmark = data_incremental_benchmark(
        task_scenario,
        experience_size=10
    )

    # 4) model & optimizer
    model = SlimResNet18CoPE(latent_dim=128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 5) logger & evaluator
    logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=True),
        loss_metrics(stream=True),
        StreamForgetting(),
        loggers=[logger]
    )

    # 6) strategy
    strategy = CoPEWithAPF(
        model=model,
        optimizer=optimizer,
        mem_size=mem_size,
        alpha=0.99,
        p_size=128,
        n_classes=n_classes,
        T=0.25,
        apf_temp=apf_temp,
        train_mb_size=10,
        train_epochs=1,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin
    )

    print(f"\n=== Seed {seed} | Memory {mem_size} ===")
    for exp in benchmark.train_stream:
        for i in range(n_iter):
            print(f"[Exp {exp.current_experience} | Iter {i+1}/{n_iter}]")
            strategy.train(exp)

    res = strategy.eval(benchmark.test_stream)
    print(res)
    return res

if __name__ == "__main__":
    seeds = [0,1,42]
    memory_sizes = [100,200, 500]
    for seed, mem in itertools.product(seeds, memory_sizes):
        run_cope_apf(seed, mem, n_iter=1)