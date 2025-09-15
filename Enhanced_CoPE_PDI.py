import torch
import torch.nn as nn
import avalanche as avl
import torch.nn.functional as F
from torch.distributions.beta import Beta
import random
import numpy as np
import itertools
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandomErasing
from avalanche.models import SlimResNet18
from avalanche.training import CoPE
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, StreamForgetting
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

# -------------------------
# SlimResNet18+CoPE PDI
# -------------------------
class SlimResNet18CoPE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.backbone = SlimResNet18(10)
        self.backbone.classifier = nn.Identity()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            feat = self.backbone(dummy)
        self.projection = nn.Linear(feat.shape[1], latent_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x / torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)

def get_apf_prototypes(targets, prototypes, sigma=0.4):
    mixed, beta = [], Beta(sigma, sigma)
    for y in targets:
        y = y.item()
        p_c = prototypes[y]
        s = random.choice([i for i in range(len(prototypes)) if i != y])
        lam = beta.sample().to(p_c.device)
        mixed.append(F.normalize((1-lam)*p_c + lam*prototypes[s], p=2, dim=0))
    return torch.stack(mixed)

class CoPEWithAPFPrototypes(CoPE):
    def __init__(self, *, sigma=0.35, **kwargs):
        self.apf_sigma = sigma
        def loss_fn(emb, tgt):
            logits = emb @ self.prototypes.T / self.T
            return F.cross_entropy(logits, tgt)
        kwargs.pop('criterion', None)
        super().__init__(criterion=loss_fn, **kwargs)

    def _before_backward(self, **kwargs):
        if hasattr(self, 'prototypes') and hasattr(self, 'mb_y'):
            self.prototypes = get_apf_prototypes(self.mb_y, self.prototypes, self.apf_sigma)

# -------------------------
#Main Code
# -------------------------
def run_apf_cope(seed: int, mem_size: int, n_iter: int = 1):
    # 1) reset RNGs
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 10

    # 2) define transforms
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std  = (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),

        # strong augmentations:
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomAffine(degrees=10,
                                translate=(0.1,0.1),
                                scale=(0.9,1.1),
                                shear=5),

        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),

        RandomErasing(p=0.25,
                      scale=(0.02,0.2),
                      ratio=(0.3,3.3))
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 3) build benchmark with transforms
    task_scenario = avl.benchmarks.SplitCIFAR10(
        n_experiences=5,
        return_task_id=False,
        fixed_class_order=list(range(n_classes)),
        train_transform=train_transform,
        eval_transform=eval_transform
    )
    benchmark = avl.benchmarks.data_incremental_benchmark(
        task_scenario,
        seed = seed,
        experience_size=10
    )

    # 4) model & optimizer
    model = SlimResNet18CoPE(latent_dim=128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 5) logger & evaluation plugin
    logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False,
                         experience=True, stream=True),
        loss_metrics(stream=True),
        StreamForgetting(),
        loggers=[logger]
    )

    # 6) CoPE strategy with APF
    strategy = CoPEWithAPFPrototypes(
        model=model,
        optimizer=optimizer,
        criterion=None,
        mem_size=mem_size,    
        alpha=0.99,
        p_size=128,
        n_classes=n_classes,
        T=0.25,
        sigma=0.35,
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
    seeds = [42, 123, 999]           #  three seeds
    memory_sizes = [100, 200, 500]   # three memory sizes
    for seed, mem in itertools.product(seeds, memory_sizes):
        run_apf_cope(seed, mem, n_iter=1)
