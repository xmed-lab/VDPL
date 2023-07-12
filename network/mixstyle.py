import random
from contextlib import contextmanager
import torch
import torch.nn as nn
from utils.data_utils import colorful_spectrum_mix, fourier_transform_GPU, colorful_spectrum_mix_GPU

def deactivate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(False)


def activate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(True)


def random_mixstyle(m):
    if type(m) == MixStyle:
        m.update_mix_method("random")


def crossdomain_mixstyle(m):
    if type(m) == MixStyle:
        m.update_mix_method("crossdomain")


@contextmanager
def run_without_mixstyle(model):
    # Assume MixStyle was initially activated
    try:
        model.apply(deactivate_mixstyle)
        yield
    finally:
        model.apply(activate_mixstyle)


@contextmanager
def run_with_mixstyle(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == "random":
        model.apply(random_mixstyle)

    elif mix == "crossdomain":
        model.apply(crossdomain_mixstyle)

    try:
        model.apply(activate_mixstyle)
        yield
    finally:
        model.apply(deactivate_mixstyle)

# add our method based on Mistyle
class MixStyle(nn.Module):

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random", norm_type='random', fourier_type='AM', fourier=True):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.fourier = fourier
        self.norm_type = norm_type
        self.fourier_type = fourier_type

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def normalization(self, norm_type):
        # decide how to compute the moments
        if norm_type == 'instance_norm':
            norm_dims = [2, 3]
        elif norm_type == 'layer_norm':
            norm_dims = [1, 2, 3]
        elif norm_type == 'random':
            p = random.random()
            if p < 0.5:
                norm_dims = [2, 3]
            else:
                norm_dims = [1, 2, 3]
        return norm_dims

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)
        device = x.device

        probability = random.random()

        if self.mix == "random":
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == "crossdomain":
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        if self.fourier and probability < 0.3:
            # print("Using fourier")
            x_tar = x[perm]
            new_x = x.clone()
            x = x.detach()
            x_tar = x_tar.detach()
            new_x = new_x.detach()

            for i in range(B):
                if self.fourier_type=="AS":
                    aug_img, _ = fourier_transform_GPU(x[i], x_tar[i], L=0.01, i=1)
                elif self.fourier_type == 'AM':
                    aug_img, _ = colorful_spectrum_mix_GPU(x[i], x_tar[i], alpha=0.3)

                new_x[i] = aug_img

            new_x = new_x.to(device)
            return_x = new_x

        else:
            # print("Using random TA")
            norm_dims = self.normalization(self.norm_type)
            mu = x.mean(dim=norm_dims, keepdim=True)
            var = x.var(dim=norm_dims, keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x-mu) / sig

            lmda = self.beta.sample((B, 1, 1, 1))
            lmda = lmda.to(x.device)

            mu2, sig2 = mu[perm], sig[perm]
            mu_mix = mu*lmda + mu2 * (1-lmda)
            sig_mix = sig*lmda + sig2 * (1-lmda)
            return_x = x_normed*sig_mix + mu_mix

        return return_x