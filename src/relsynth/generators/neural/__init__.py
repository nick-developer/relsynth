# SPDX-License-Identifier: Apache-2.0
"""Neural generator plug-ins (optional extra ``relsynth[neural]``, PyTorch).

* ``tvae``: a conditional tabular variational autoencoder (Xu et al. 2019).
* ``ddpm``: a conditional Gaussian-multinomial diffusion model (TabDDPM,
  Kotelnikov et al. 2023), conditioned on the row's latent group as
  ClavaDDPM (Pang et al. 2024) conditions on parent clusters.

Importing this package does not import torch; fitting does.  See docs/neural.md.
"""
from .ddpm import DDPMGenerator
from .tvae import TVAEGenerator

__all__ = ["DDPMGenerator", "TVAEGenerator"]
