# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# PyTorch Hub entry point. Lets this repo serve the official V-JEPA 2 and
# V-JEPA 2.1 encoder+predictor backbones directly:
#
#   import torch
#   enc, pred = torch.hub.load("/path/to/VJEPA-Policy", "vjepa2_vit_giant",
#                              source="local", pretrained=True)
#   enc, pred = torch.hub.load("/path/to/VJEPA-Policy", "vjepa2_1_vit_giant_384",
#                              source="local", pretrained=True)
#
# `pretrained=True` downloads weights from https://dl.fbaipublicfiles.com/vjepa2
# (override the host with the VJEPA2_WEIGHTS_URL env var, e.g. a local mirror).
#
# torch.hub puts the repo ROOT on sys.path; the package lives under src/, so add
# it here before importing.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from vjepa_policy.hub.backbones import (  # noqa: E402
    vjepa2_ac_vit_giant,
    vjepa2_vit_giant,
    vjepa2_vit_giant_384,
    vjepa2_vit_huge,
    vjepa2_vit_large,
    vjepa2_1_vit_base_384,
    vjepa2_1_vit_giant_384,
    vjepa2_1_vit_gigantic_384,
    vjepa2_1_vit_large_384,
)

dependencies = ["torch", "timm", "einops"]

__all__ = [
    "vjepa2_vit_large",
    "vjepa2_vit_huge",
    "vjepa2_vit_giant",
    "vjepa2_vit_giant_384",
    "vjepa2_ac_vit_giant",
    "vjepa2_1_vit_base_384",
    "vjepa2_1_vit_large_384",
    "vjepa2_1_vit_giant_384",
    "vjepa2_1_vit_gigantic_384",
]
