# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from eve import NodeTranslator
from . import cuir

class DependencyAnalysis(NodeTranslator):
    """
    Dependency analysis for CUIR kernels, store dependency array in Program node
    """

    def visit_Program(self, node: cuir.Program) -> cuir.Program:
        if len(node.dependency) == len(node.kernels):
            dependency = node.dependency
        else:
            dependency = list(range(-1, len(node.kernels)-1)) # because of kernel fusion
        writes = []
        reads = []
        for i in range(len(node.kernels)):
            kernel = node.kernels[i]
            writes.append(
                kernel.iter_tree()
                .if_isinstance(cuir.AssignStmt)
                .getattr("left")
                .if_isinstance(cuir.FieldAccess)
                .getattr("name")
                .to_set()
            )
            reads.append(
                kernel.iter_tree()
                .if_isinstance(cuir.FieldAccess)
                .getattr("name")
                .to_set() - writes[i]
            )
            for j in range(i, 0, -1):
                # R -> W, W -> W
                if writes[i].intersection(writes[j].union(reads[j])):
                    break
                # W -> R
                if reads[i].intersection(writes[j]):
                    break
                dependency[i] = j - 1
        return cuir.Program(
            name=node.name, params=node.params, temporaries=node.temporaries, kernels=node.kernels, dependency=dependency
        )