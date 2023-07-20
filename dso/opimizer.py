import torch


class PolicyOptimizer(torch.optim.Optimizer):
    pass


class PGPolicyOptimizer(PolicyOptimizer):
    pass


class PPOPolicyOptimizer(PolicyOptimizer):
    pass


class PQTPolicyOptimizer(PolicyOptimizer):
    pass
