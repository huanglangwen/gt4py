# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

"""GTGraph decorator
Interface functions to define the 'gtgraph' decorator to construct dataflow
graphs from functions that call gtscript stencils.
"""

import ast
import inspect
import networkx as nx
import types
import astor
from typing import Any, Callable, Dict, Tuple
from gt4py.stencil_object import StencilObject


_graphs: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

def get_stencil_in_context():
    ctx = globals()
    return {k: ctx[k] for k in ctx if isinstance(ctx[k], StencilObject)}

def gtgraph(definition=None, **stencil_kwargs):
    def decorator(definition) -> Callable[..., None]:
        def_name = f"{definition.__module__}.{definition.__name__}"
        stencil_ctx = get_stencil_in_context()
        graph, meta_data = GraphMaker.apply(definition, stencil_ctx)
        _graphs[def_name] = (graph, meta_data)

        return _graphs[def_name]

    if definition is None:
        return decorator
    else:
        return decorator(definition)

class InsertAsync(ast.NodeTransformer):
    @classmethod
    def apply(cls, definition, ctx):
        maker = cls(definition, ctx)
        return astor.to_source(maker.visit(maker.ast_root))

    def __init__(self, definition, ctx):
        self.ast_root = astor.code_to_ast(definition)
        self.stencil_ctx = {k: ctx[k] for k in ctx if isinstance(ctx[k], StencilObject)}

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            keywords = [i.arg for i in node.keywords]
            if (func_name in self.stencil_ctx) and ('async_launch' not in keywords):
                return ast.copy_location(ast.Call(func=node.func,
                                                  args=node.args,
                                                  keywords=node.keywords+[ast.keyword(arg='async_launch',
                                                                                      value=ast.Constant(value=True, kind=None))]), node)
        return node



class GraphMaker(ast.NodeVisitor):
    @classmethod
    def apply(cls, definition, stencil_ctx: Dict[str, StencilObject]):
        maker = cls(definition, stencil_ctx)
        maker(maker.ast_root)
        return maker.graph, maker.meta_data

    def __init__(self, definition, stencil_ctx: Dict[str, StencilObject]):
        assert isinstance(definition, types.FunctionType)
        self.definition = definition
        self.filename = inspect.getfile(definition)
        self.source = inspect.getsource(definition)
        self.ast_root = ast.parse(self.source)
        self.graph = nx.DiGraph()
        self.meta_data: Dict[str, Any] = {}
        self.stencil_ctx: Dict[str, StencilObject] = stencil_ctx

    def __call__(self, func_node: ast.FunctionDef):
        self.visit(func_node)

    def __contains__(self, node_name: str) -> bool:
        return node_name in self.meta_data

    def _add_node(self, node_name: str, node_meta: Any) -> None:
        self.graph.add_node(node_name)
        self.meta_data[node_name] = dict(node=node_meta)

    def visit(self, node, **kwargs):
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, **kwargs)

    def visit_FunctionDef(self, node: ast.FunctionDef, **kwargs):
        for arg in node.args.args:
            self.visit(arg)
        for entry in node.body:
            self.visit(entry)

    def visit_arg(self, node: ast.arg, **kwargs):
        arg_name = node.arg
        if isinstance(node.annotation, ast.Subscript):
            container_type = node.annotation.value.id.lower()
            if container_type == "dict":
                arg_type = node.annotation.slice.value.elts[1].value
        else:
            arg_type = node.annotation.id
        if "Field" in arg_type:
            self._add_node(arg_name, node)

    def visit_Assign(self, node: ast.Assign, **kwargs):
        if isinstance(node.value, ast.Call):
            func = node.value.func
            func_name = func.attr if isinstance(func, ast.Attribute) else func.id
            # TODO: This is very fv3core-specific, need to generalize for gt4py...
            if func_name == "copy" or func_name.startswith("make_storage"):
                for target in node.targets:
                    self._add_node(target.id, node)

    def visit_Call(self, node: ast.Call, **kwargs):
        func = node.func
        is_attr = isinstance(func, ast.Attribute)
        func_name = func.attr if is_attr else func.id
        if "Exception" not in func_name:
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    arg_name = arg.id
                    if arg_name in self:
                        if func_name not in self:
                            self._add_node(func_name, node)
                        # TODO: How to determine if in/out?
                        self.graph.add_edge(arg.id, func_name)

            for keyword in node.keywords:
                arg_name = keyword.arg
                if arg_name in self:
                    if func_name not in self:
                        self._add_node(func_name, node)
                    self.graph.add_edge(arg.id, func_name)
                elif arg_name == "domain" or arg_name == "origin":
                    self.meta_data[func_name][arg_name] = keyword.value