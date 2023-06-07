# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

def quat_apply_yaw_inverse(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw[:, 2] = -quat_yaw[:, 2]
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

def unravel_indices(tensor, shape, out=None, in_place=True):
    """Converts flat indices to multi-dim indices. Warning: behaviour different from that of numpy.unravel_index.
    - tensor: shape [...], containing the indices in the flat tensor
    - shape (tuple). If the indices are incompatible with the given shape, the result is unspecified.
    - out (optional tensor, shape [..., len(shape)]): if not None, the indices will be stored
    in it instead of allocating a new tensor
    - in_place (bool): if True, 'tensor' will be modified in place during the conversion.
    The state of the input 'tensor' when this function returns is then unspecified."""
    if not in_place: tensor = tensor.clone()
    if out is None:
        out = torch.zeros(*tensor.shape, len(shape), device=tensor.device, dtype=torch.long)
    
    for i, dim in enumerate(reversed(shape)):
        out[...,-i-1] = tensor % dim
        tensor.div_(dim, rounding_mode="floor")

    return out

def ravel_indices(tensor, shape, out=None, in_place=True):
    """Converts indices in a multi-dim tensor to the corresponding indices in the flattened tensor.
     Warning: behaviour different from that of numpy.ravel_multi_index.
    - tensor: shape [..., K], containing the K-dim indices to convert
    - shape (tuple with K elements): the shape of the original tensor
    - out (optional tensor, shape [...]): if not None, the indices will be stored
    in it instead of allocating a new tensor
    - in_place (bool): if True, 'tensor' will be modified in place during the conversion.
    The state of the input 'tensor' when this function returns is then unspecified."""
    if not in_place: tensor = tensor.clone()
    if out is None:
        out = torch.zeros(*tensor.shape[:-1], device=tensor.device, dtype=torch.long)
    else:
        out[:] = 0
    
    for i, dim in enumerate(reversed(shape)):
        out[:].add_(tensor[..., -i-1])
        tensor.mul_(dim)

    return out