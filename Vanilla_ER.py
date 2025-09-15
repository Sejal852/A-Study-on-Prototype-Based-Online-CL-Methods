    # ==========================
#  CIFAR-10 Replay Sweep: Memory Size and Iterations
# ==========================

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torchvision import transforms, models
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.training.supervised import Replay
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin

# -------------------------
# Set random seeds
# -------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# -------------------------
# CIFAR-10 Experiment Runner
# -------------------------
def run_cifar10_experiment(
    mem_size: int,
    n_iter: int = 1,
    train_mb_size: int = 64,
    train_epochs: int = 1
):
    """
    Runs a Replay strategy on SplitCIFAR10.
    :param mem_size: Replay buffer size
    :param n_iter: Number of times to repeat training per experience
    :param train_mb_size: Mini-batch size for training (fixed)
    :param train_epochs: Epochs per call (1 when manual looping)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # normalization stats
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    benchmark = SplitCIFAR10(
        n_experiences=5,
        seed=seed,
        train_transform=train_transform,
        eval_transform=eval_transform
    )

    # model setup
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[InteractiveLogger()]
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    lr_scheduler_plugin = LRSchedulerPlugin(scheduler)

    strategy = Replay(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        mem_size=mem_size,
        train_mb_size=train_mb_size,
        train_epochs=train_epochs,
        eval_mb_size=64,
        device=device,
        evaluator=eval_plugin,
        plugins=[lr_scheduler_plugin]
    )

    print(f"\n=== Replay: mem_size={mem_size}, iterations={n_iter}, mb_size={train_mb_size} ===")
    for exp_id, experience in enumerate(benchmark.train_stream):
        print(f"--- Experience {exp_id+1}/{len(benchmark.train_stream)} ---")
        print("Classes:", experience.classes_in_this_experience)
        for it in range(n_iter):
            print(f"Iteration {it+1}/{n_iter}")
            strategy.train(experience)
        metrics = strategy.eval(benchmark.test_stream)
        print("Metrics:", metrics)

if __name__ == "__main__":
   
    mem_sizes = [100, 200, 500]
    n_iters = [1,2]

    for mem in mem_sizes:
        for n in n_iters:
            run_cifar10_experiment(mem_size=mem, n_iter=n)
