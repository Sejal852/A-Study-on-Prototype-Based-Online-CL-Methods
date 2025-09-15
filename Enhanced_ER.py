# ==========================
#  CIFAR-10 Replay + APF Plugin 
# ==========================

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torchvision import transforms, models
from avalanche.benchmarks import SplitCIFAR10
from avalanche.training.supervised import Replay
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin, SupervisedPlugin
from collections import defaultdict

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
# Define Dynamic APF Replay Plugin
# -------------------------
class DynamicAPFReplayPlugin(SupervisedPlugin):
    def __init__(self, mem_size=500, num_classes=10,
                 init_replay_ratio=0.8, final_replay_ratio=0.6,
                 total_experiences=5,
                 use_mixup=False, use_prioritized_sampling=True,
                 use_kl_loss=False, apf_ratio=0.25, sigma=0.5):
        super().__init__()
        self.mem_size = mem_size
        self.buffer = []
        self.num_classes = num_classes
        self.init_replay_ratio = init_replay_ratio
        self.final_replay_ratio = final_replay_ratio
        self.total_experiences = total_experiences
        self.current_experience = 0
        self.use_prioritized_sampling = use_prioritized_sampling
        self.apf_ratio = apf_ratio
        self.sigma = sigma
        self.forgetting_scores = {c: 1.0 for c in range(num_classes)}

    def update_forgetting_scores(self, eval_results):
        eps = 1e-6
        for cls, acc in eval_results.items():
            self.forgetting_scores[cls] = 1.0 / (acc + eps)

    def _get_dynamic_replay_ratio(self):
        ratio = self.init_replay_ratio - (
            (self.init_replay_ratio - self.final_replay_ratio)
            * (self.current_experience / max(1, self.total_experiences - 1)))
        return max(ratio, self.final_replay_ratio)

    def after_training_exp(self, strategy, **kwargs):
        self.current_experience = strategy.experience.current_experience
        data = [(x.cpu(), int(y)) for x,y,*_ in strategy.experience.dataset]
        per_class = defaultdict(list)
        for x,y in data:
            per_class[y].append((x,y))
        target = self.mem_size // max(1, len(per_class))
        new_buffer = []
        for samples in per_class.values():
            idx = np.random.choice(len(samples), min(target, len(samples)), replace=False)
            new_buffer += [samples[i] for i in idx]
        self.buffer = (self.buffer + new_buffer)[-self.mem_size:]

    def before_training_iteration(self, strategy, **kwargs):
        if len(self.buffer) < 1: return
        x,y,*rest = strategy.mbatch
        bs = x.size(0)
        ratio = self._get_dynamic_replay_ratio()
        rbs = max(1, int(ratio * bs))
        # prioritized sampling
        if self.use_prioritized_sampling:
            weights = np.array([self.forgetting_scores[l] for _,l in self.buffer])
            weights /= weights.sum()
        else:
            weights = None
        n_apf = int(self.apf_ratio * rbs)
        n_uni = rbs - n_apf
        idx_apf = np.random.choice(len(self.buffer), n_apf, p=weights) if weights is not None else np.random.choice(len(self.buffer), n_apf)
        idx_uni = np.random.choice(len(self.buffer), n_uni)
        idx = np.concatenate([idx_apf, idx_uni])
        rx = torch.stack([self.buffer[i][0] for i in idx]).to(x.device)
        ry = torch.tensor([self.buffer[i][1] for i in idx], device=x.device)
        strategy.mbatch = (torch.cat([x, rx]), torch.cat([y, ry]), *rest)

# -------------------------
# Experiment Runner
# -------------------------
def run_cifar10_experiment(mem_size: int, n_iter: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    norm_mean = (0.4914,0.4822,0.4465)
    norm_std = (0.2470,0.2435,0.2616)
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    eval_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    benchmark = SplitCIFAR10(
        n_experiences=5,
        shuffle=True,
        fixed_class_order=list(range(10)),
        train_transform=train_t,
        eval_transform=eval_t
    )

    # model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    optim_ = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    eval_pl = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[InteractiveLogger()]
    )
    sched = optim.lr_scheduler.StepLR(optim_, step_size=10, gamma=0.5)
    lr_pl = LRSchedulerPlugin(sched)

    # APF plugin
    apf = DynamicAPFReplayPlugin(
        mem_size=mem_size,
        total_experiences=benchmark.n_experiences
    )
    # dummy update
    apf.update_forgetting_scores({i:0.9 for i in range(10)})

    strategy = Replay(
        model=model,
        optimizer=optim_,
        criterion=crit,
        mem_size=mem_size,
        train_mb_size=64,
        train_epochs=1,
        eval_mb_size=64,
        device=device,
        evaluator=eval_pl,
        plugins=[lr_pl, apf]
    )

    print(f"\n=== Replay+APF: mem_size={mem_size}, iterations={n_iter} ===")
    for exp in benchmark.train_stream:
        print(f"--- Experience {exp.current_experience}/{benchmark.n_experiences} ---")
        for i in range(n_iter):
            print(f"Iteration {i+1}/{n_iter}")
            strategy.train(exp)
        res = strategy.eval(benchmark.test_stream)
        print(res)

if __name__ == '__main__':
    mem_sizes = [100, 200, 500]
    n_iters = [1]
    for m in mem_sizes:
        for it in n_iters:
            run_cifar10_experiment(mem_size=m, n_iter=it)
