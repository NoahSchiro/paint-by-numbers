# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Noah Schiro, 2025 -- substantial modifications
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToImage


def get_train_transform():
    transform_list = [
        ToImage(),
        RandomHorizontalFlip(),
        ToDtype(torch.float32, scale=True),
    ]
    return Compose(transform_list)
