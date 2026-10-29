"""The linear head for Path A: logistic regression on the average patch vector.

score = weight . average_patch_vector + bias. A higher score means "more likely AI".
Because it is linear, the same weights applied to each patch give the evidence map, and the
average of those patch scores is exactly the image score.
"""

import torch
import torch.nn.functional as F


class LinearHead:
    def __init__(self, weight=None, bias=None):
        self.weight = weight  # [1024]
        self.bias = bias  # scalar tensor

    def fit(self, x, y, weight_decay=1e-3, max_iter=200):
        """x: [N, 1024] float features, y: [N] with 1 = AI, 0 = real."""
        # Standardise each feature so all 1,024 numbers are on a similar scale (more stable training).
        mean, std = x.mean(dim=0), x.std(dim=0) + 1e-6
        z = (x - mean) / std
        # Give real and AI the same total weight, even if one class has more images.
        n_ai = y.sum()
        sample_weight = torch.where(y == 1, len(y) / (2 * n_ai), len(y) / (2 * (len(y) - n_ai)))

        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(z @ w + b, y, weight=sample_weight)
            loss = loss + weight_decay * (w ** 2).sum() / 2  # weight decay: discourages huge weights
            loss.backward()
            return loss

        optimizer.step(closure)
        # Fold the standardisation back into the weights, so the head stays linear in the raw
        # average patch vector (and therefore in every single patch).
        with torch.no_grad():
            self.weight = w / std
            self.bias = (b - (w * mean / std).sum()).squeeze()
        return self

    def logits(self, x):
        return x @ self.weight + self.bias

    def save(self, path):
        torch.save({"weight": self.weight, "bias": self.bias}, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        return cls(state["weight"], state["bias"])
