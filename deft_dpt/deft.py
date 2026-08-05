"""
DEFT-DPT — DEFT Module
======================
Dynamic Egocentric Feature Transformation: an affine spatial transform
predicted from the frame, followed by a radial (centre-biased) weighting.

STATUS — read before using
--------------------------
`identity_init=True` zeroes fc2.weight and sets its bias to the identity
affine. While fc2.weight stays zero, theta equals that bias for EVERY input,
i.e. the localization branch is inactive and only the radial weighting has
any effect. This matches what the feature-extraction notebooks do today.

Training DEFT requires a downstream task loss backpropagated through it
(frame -> DEFT -> backbone -> graph -> classifier -> CE). A standalone
reconstruction loss MSE(warped, original) is degenerate: its optimum is the
identity. See train.py for where joint training would hook in.

Pass identity_init=False for a small-random init suitable for joint training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SIGMA = 0.5
DEFAULT_LAMBDA = 0.5


class LocalizationNetwork(nn.Module):
    """
    Predicts a 2x3 affine matrix from a frame.

    Args:
        input_channels : 3 for RGB, 1 for a single flow channel
        input_hw       : (H, W) so fc1 is built eagerly. The notebooks built
                         fc1 lazily inside forward(), which means it is absent
                         from state_dict() until a forward pass has run and is
                         therefore silently dropped from any checkpoint saved
                         before that. Always pass input_hw when training.
        identity_init  : see module docstring.
    """

    def __init__(self, input_channels=3, input_hw=None, identity_init=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 10, kernel_size=5)
        self.fc2 = nn.Linear(32, 6)

        if input_hw is not None:
            self.fc1 = nn.Linear(self._flat_size(input_channels, input_hw), 32)
            self.computed_fc1 = True
        else:
            self.fc1 = nn.Linear(1, 32)   # placeholder, rebuilt on first forward

        if identity_init:
            # Frozen identity: theta is constant regardless of input.
            self.fc2.weight.data.zero_()
            self.fc2.bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            # Small-random init: theta starts near identity but is
            # input-dependent, so gradients can shape it.
            self.fc2.weight.data.normal_(mean=0.0, std=1e-3)
            self.fc2.bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
                + 1e-4 * torch.randn(6))

    def _flat_size(self, c, hw):
        with torch.no_grad():
            d = torch.zeros(1, c, *hw)
            d = self.pool(F.relu(self.conv1(d)))
            d = self.pool(F.relu(self.conv2(d)))
            return d.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if not hasattr(self, "computed_fc1"):
            self.fc1 = nn.Linear(x.view(x.shape[0], -1).shape[1], 32).to(x.device)
            self.computed_fc1 = True
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).view(-1, 2, 3)


class WeightingModule(nn.Module):
    """Radial weight  w(r) = 1 + lambda * exp(-r^2 / (2*sigma^2))  over the grid."""

    def __init__(self, sigma=DEFAULT_SIGMA, lambda_init=DEFAULT_LAMBDA):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(float(lambda_init)))
        self.sigma = sigma

    def forward(self, grid):
        dist2 = grid[..., 0] ** 2 + grid[..., 1] ** 2
        w = 1 + self.lambda_param * torch.exp(-dist2 / (2 * self.sigma ** 2))
        return w.unsqueeze(-1)


class DEFTModule(nn.Module):
    def __init__(self, input_channels=3, sigma=DEFAULT_SIGMA,
                 lambda_init=DEFAULT_LAMBDA, input_hw=None, identity_init=True):
        super().__init__()
        self.localization = LocalizationNetwork(input_channels, input_hw,
                                                identity_init)
        self.weighting = WeightingModule(sigma, lambda_init)

    def forward(self, x, return_parts=False):
        theta = self.localization(x)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        weight = self.weighting(grid)
        x_warped = F.grid_sample(x, grid, align_corners=False)

        if x.shape[1] > 1:
            w = weight.expand(-1, x.shape[2], x.shape[3],
                              x.shape[1]).permute(0, 3, 1, 2)
        else:
            w = weight.permute(0, 3, 1, 2)

        out = x_warped * w
        if return_parts:
            return out, theta, x_warped, weight.squeeze(-1)
        return out


def theta_is_identity(theta, tol=1e-5):
    eye = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return torch.allclose(theta.detach().cpu(), eye, atol=tol)


def report_theta(names, theta, wmap=None):
    """Print whether the localization branch is input-dependent."""
    print("\n" + "=" * 64)
    print("  Predicted affine parameters (theta)")
    print("=" * 64)
    n_ident = 0
    for name, th in zip(names, theta):
        flat = ", ".join(f"{v:+.4f}" for v in th.flatten().tolist())
        ident = theta_is_identity(th)
        n_ident += ident
        print(f"  {name}: [{flat}]{'   <- IDENTITY' if ident else ''}")
    spread = (theta.max(0).values - theta.min(0).values).abs().max().item()
    print(f"\n  Max spread across frames : {spread:.3e}")
    if wmap is not None:
        print(f"  Weight map range         : {wmap.min():.3f} (edge) "
              f"-> {wmap.max():.3f} (centre)")
    if n_ident == len(theta):
        print("\n  [WARN] theta is the identity for every frame: the localization\n"
              "         branch is inactive; only the radial weighting applies.")
    print("=" * 64 + "\n")
