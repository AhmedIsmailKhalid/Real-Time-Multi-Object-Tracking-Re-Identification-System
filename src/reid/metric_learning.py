"""
Metric learning losses for Re-ID training.
"""

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.

    Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)

    where:
    - anchor: Reference sample
    - positive: Sample from same class as anchor
    - negative: Sample from different class
    - d: Distance function (typically Euclidean)
    """

    def __init__(self, margin: float = 0.3):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            inputs: Feature vectors (N, feature_dim)
            targets: Class labels (N,)

        Returns:
            Triplet loss value
        """
        n = inputs.size(0)

        # Compute pairwise distance matrix
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # For numerical stability

        # For each anchor, find hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        dist_ap = []
        dist_an = []

        for i in range(n):
            # Hardest positive (furthest positive sample)
            pos_dists = dist[i][mask[i]]
            if len(pos_dists) > 1:  # Need at least 2 samples from same class
                dist_ap.append(pos_dists.max().unsqueeze(0))
            else:
                dist_ap.append(torch.zeros(1).to(inputs.device))

            # Hardest negative (closest negative sample)
            neg_dists = dist[i][mask[i] == 0]
            if len(neg_dists) > 0:
                dist_an.append(neg_dists.min().unsqueeze(0))
            else:
                dist_an.append(torch.ones(1).to(inputs.device) * 1e6)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Helps prevent overfitting by not pushing the model to be
    overconfident about its predictions.
    """

    def __init__(self, num_classes: int, epsilon: float = 0.1):
        """
        Initialize label smoothing loss.

        Args:
            num_classes: Number of classes
            epsilon: Smoothing parameter
        """
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross entropy loss with label smoothing.

        Args:
            inputs: Class logits (N, num_classes)
            targets: Class labels (N,)

        Returns:
            Loss value
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()

        return loss
