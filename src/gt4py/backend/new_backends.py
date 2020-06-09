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


class _MaxKOffsetExtractor(gt_ir.IRNodeVisitor):
    @classmethod
    def apply(cls, root_node):
        return cls()(root_node)

    def __init__(self):
        self.max_offset = 2

    def __call__(self, node):
        self.visit(node)
        return self.max_offset

    def visit_AxisBound(self, node: gt_ir.AxisBound):
        self.max_offset = max(self.max_offset, abs(node.offset) + 1)


class CXXOptExtGenerator(gt_backend.GTPyExtGenerator):
    OP_TO_CPP = gt_backend.GTPyExtGenerator.OP_TO_CPP
    DATA_TYPE_TO_CPP = gt_backend.GTPyExtGenerator.DATA_TYPE_TO_CPP

    TEMPLATE_FILES = copy.deepcopy(gt_backend.GTPyExtGenerator.TEMPLATE_FILES)
    TEMPLATE_FILES["computation.hpp"] = "new_computation.hpp.in"
    TEMPLATE_FILES["computation.src"] = "new_computation.src.in"

    ITERATORS = ("i", "j", "k")

    def __init__(self, class_name, module_name, gt_backend_t, options):
        super().__init__(class_name, module_name, gt_backend_t, options)

    def visit_FieldRef(self, node: gt_ir.FieldRef, **kwargs):
        assert node.name in self.apply_block_symbols
        offset = [node.offset.get(name, 0) for name in self.domain.axes_names]

        iter_tuple = []
        for i in range(len(offset)):
            iter = CXXOptExtGenerator.ITERATORS[i]
            if offset[i] != 0:
                oper = ""
                if offset[i] > 0:
                    oper = "+"
                iter_tuple.append(iter + oper + str(offset[i]))
            else:
                iter_tuple.append(iter)

        return "{name}({idx})".format(name=node.name, idx=", ".join(iter_tuple))

    def visit_VarRef(self, node: gt_ir.VarRef, *, write_context=False):
        assert node.name in self.apply_block_symbols

        if write_context and node.name not in self.declared_symbols:
            self.declared_symbols.add(node.name)
            source = self._make_cpp_type(self.apply_block_symbols[node.name].data_type) + " "
        else:
            source = ""

        idx = ", ".join(str(i) for i in node.index) if node.index else ""
        if len(idx) > 0:
            idx = f"({idx})"
        if node.name in self.impl_node.parameters:
            source += "{name}{idx}".format(name=node.name, idx=idx)
        else:
            source += "{name}".format(name=node.name)
            if idx:
                source += "[{idx}]".format(idx=idx)

        return source

    def visit_StencilImplementation(self, node: gt_ir.StencilImplementation):
        offset_limit = _MaxKOffsetExtractor.apply(node)
        k_axis = {"n_intervals": 1, "offset_limit": offset_limit}
        max_extent = functools.reduce(
            lambda a, b: a | b, node.fields_extents.values(), gt_definitions.Extent.zeros()
        )
        halo_sizes = tuple(max(lower, upper) for lower, upper in max_extent.to_boundary())
        constants = {}
        if node.externals:
            for name, value in node.externals.items():
                value = self._make_cpp_value(name)
                if value is not None:
                    constants[name] = value

        arg_fields = []
        tmp_fields = []
        storage_ids = []
        max_ndim = 0
        for name, field_decl in node.fields.items():
            if name not in node.unreferenced:
                max_ndim = max(max_ndim, len(field_decl.axes))
                field_attributes = {
                    "name": field_decl.name,
                    "dtype": self._make_cpp_type(field_decl.data_type),
                }
                if field_decl.is_api:
                    if field_decl.layout_id not in storage_ids:
                        storage_ids.append(field_decl.layout_id)
                    field_attributes["layout_id"] = storage_ids.index(field_decl.layout_id)
                    arg_fields.append(field_attributes)
                else:
                    tmp_fields.append(field_attributes)

        parameters = [
            {"name": parameter.name, "dtype": self._make_cpp_type(parameter.data_type)}
            for name, parameter in node.parameters.items()
            if name not in node.unreferenced
        ]

        steps = []
        multi_stages = []

        for multi_stage in node.multi_stages:
            for group in multi_stage.groups:
                interval = []
                for stage in group.stages:
                    stage_start = stage.apply_blocks[0].interval.start
                    start_level = "min" if stage_start.level == gt_ir.LevelMarker.START else "max"
                    stage_end = stage.apply_blocks[0].interval.end
                    end_level = "min" if stage_end.level == gt_ir.LevelMarker.START else "max"
                    interval = [
                        dict(level=start_level, offset=stage_start.offset),
                        dict(level=end_level, offset=stage_end.offset),
                    ]

                    extents = []
                    compute_extent = stage.compute_extent
                    for i in range(compute_extent.ndims):
                        extents.extend(
                            [compute_extent.lower_indices[i], compute_extent.upper_indices[i]]
                        )

                    step = self.visit(stage)
                    step["extents"] = extents
                    steps.append(step)

            multi_stages.append(
                {
                    "exec": str(multi_stage.iteration_order).lower(),
                    "interval": interval,
                    "steps": steps,
                }
            )

        template_args = dict(
            arg_fields=arg_fields,
            constants=constants,
            gt_backend=self.gt_backend_t,
            halo_sizes=halo_sizes,
            k_axis=k_axis,
            module_name=self.module_name,
            multi_stages=multi_stages,
            parameters=parameters,
            stencil_unique_name=self.class_name,
            tmp_fields=tmp_fields,
            max_ndim=max_ndim,
        )

        sources = {}
        for key, template in self.templates.items():
            sources[key] = template.render(**template_args)

        return sources


@gt_backend.register
class CXXOptBackend(gt_backend.BaseGTBackend):
    PYEXT_GENERATOR_CLASS = CXXOptExtGenerator
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
