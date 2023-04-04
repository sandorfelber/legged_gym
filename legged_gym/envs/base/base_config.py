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

import inspect
import yaml

class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key=="__class__":
                continue
            # get the corresponding attribute object
            var =  getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)

    @staticmethod
    def _serialize_rec(obj) -> dict:
        s = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue

            var = getattr(obj,key)
            if callable(var) or "numpy" in str(type(var)):
                continue
            if isinstance(var, (int, float, bool, str, list, tuple, dict, yaml.YAMLObject)):
                s[key] = var
            else:
                s[key] = BaseConfig._serialize_rec(var)
        return s

    def serialize(self):
        return yaml.dump(BaseConfig._serialize_rec(self), default_flow_style=False)
    
    def export(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            file.write(self.serialize())
    
    @staticmethod
    def _update_children(obj, config: dict):
        for key in config:
            if isinstance(config[key], dict):                
                if not hasattr(obj, key):
                    print("Warning: unsupported config namespace given in YAML file: %s (ignored)" % key)
                    continue
                var = getattr(obj, key)
                if isinstance(var, dict):
                    setattr(obj, key, config[key])
                else:
                    BaseConfig._update_children(getattr(obj, key), config[key])
            else:
                setattr(obj, key, config[key])
    
    def update(self, config_str):
        config_dict = yaml.safe_load(config_str)
        BaseConfig._update_children(self, config_dict)
        
    def update_from(self, path):
        try:
            with open(path) as file:
                config_str = file.read()
                self.update(config_str)
        except IOError as e:
            print("Cannot load config from previous run:", e)