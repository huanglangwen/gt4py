# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import abc
import copy
import functools
import numbers
import os
import types

import jinja2
import numpy as np

from gt4py import analysis as gt_analysis
from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from gt4py.utils import text as gt_text


class CXXOptExtGenerator(gt_backend.GTPyExtGenerator):
    TEMPLATE_FILES = copy.deepcopy(gt_backend.GTPyExtGenerator.TEMPLATE_FILES)
    TEMPLATE_FILES["computation.src"] = "new_computation.src.in"


@gt_backend.register
class CXXOptBackend(gt_backend.BaseGTBackend):
    MODULE_GENERATOR_CLASS = CXXOptExtGenerator
    GT_BACKEND_T = "x86"

    name = "cxxopt"
    options = gt_backend.BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": gt_backend.make_x86_layout_map,
        "is_compatible_layout": gt_backend.x86_is_compatible_layout,
        "is_compatible_type": gt_backend.gtcpu_is_compatible_type,
    }

    @classmethod
    def generate_extension(cls, stencil_id, definition_ir, options, **kwargs):
        return cls._generic_generate_extension(
            stencil_id, definition_ir, options, uses_cuda=False, **kwargs
        )
