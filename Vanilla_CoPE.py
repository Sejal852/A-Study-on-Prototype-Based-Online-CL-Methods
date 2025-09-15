import random
import numpy as np
import torch
import torch.nn as nn
import avalanche as avl

class SlimResNet18CoPE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        # Backbone for CIFAR-10
        self.backbone = avl.models.SlimResNet18(num_classes)
        self.backbone.classifier = nn.Identity()

        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            feats = self.backbone(dummy)
            feat_dim = feats.shape[1]

        self.projection = nn.Linear(feat_dim, latent_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        # ℓ₂-normalize
        return x / x.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cope_cifar10(seed: int, mem_size: int):
    """
    Run CoPE on SplitCIFAR10 with given seed and memory size.
    Trains each experience for one epoch.
    """
    # Ensure reproducibility
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 10

    # Split CIFAR-10 into 10 experiences, one class per experience
    task_scenario = avl.benchmarks.SplitCIFAR10(
        n_experiences=5,
        seed=seed,
        return_task_id=False,
        fixed_class_order=list(range(n_classes))
    )

    # Online data-incremental benchmark: each batch is one experience
    benchmark = avl.benchmarks.data_incremental_benchmark(
        task_scenario,
        experience_size=100  # ~100 samples per class for CIFAR-10
    )

    print(f"Seed {seed} | Mem {mem_size}: {benchmark.n_experiences} experiences.")

    model = SlimResNet18CoPE(latent_dim=256, num_classes=n_classes).to(device)

    logger = avl.logging.InteractiveLogger()
    eval_plugin = avl.training.plugins.EvaluationPlugin(
        avl.evaluation.metrics.accuracy_metrics(experience=True, stream=True),
        avl.evaluation.metrics.loss_metrics(stream=True),
        avl.evaluation.metrics.StreamForgetting(),
        loggers=[logger]
    )

    cope = avl.training.plugins.CoPEPlugin(
        mem_size=mem_size,
        alpha=0.99,
        p_size=256,
        n_classes=n_classes,
        T=0.15
    )

    cl_strategy = avl.training.Naive(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        cope.ppp_loss,
        train_mb_size=10,
        train_epochs=1,  # single epoch per experience
        eval_mb_size=100,
        device=device,
        plugins=[cope],
        evaluator=eval_plugin
    )

    # Training loop: one epoch per experience
    for experience in benchmark.train_stream:
        print(f"[Seed {seed} | Mem {mem_size} | Exp {experience.current_experience}]")
        cl_strategy.train(experience)

    # Evaluate on test stream
    res = cl_strategy.eval(benchmark.test_stream)
    return res


if __name__ == '__main__':
    seeds = [0, 1, 42]
    mem_sizes = [100,200, 500]
    all_results = {}

    for seed in seeds:
        for mem in mem_sizes:
            print("\n" + "="*40)
            print(f"Running CoPE CIFAR-10 with seed={seed}, mem_size={mem}")
            print("="*40)
            result = cope_cifar10(seed, mem)
            all_results[(seed, mem)] = result

    # Summary of results
    print("\nSummary of results:")
    for (seed, mem), res in all_results.items():
        print(f"Seed {seed} | Mem {mem}: {res}")
