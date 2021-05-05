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

import cupy.cuda
import networkx as nx
import types
import astor
from typing import Any, Callable, Dict, Tuple, List, Deque
from gt4py.stencil_object import StencilObject
from gt4py import AccessKind
from collections import deque
from dataclasses import dataclass
from time import sleep

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

@dataclass
class InvokedStencil():
    stencil: StencilObject
    done_event: cupy.cuda.Event

class AsyncContext():
    def __init__(self, num_streams, max_invoked_stencils = 50):
        self.stream_pool: List[cupy.cuda.Stream] = []
        self.add_streams(num_streams)
        self.invoked_stencils: Deque[InvokedStencil] = deque()
        self.max_invoked_stencils: int = max_invoked_stencils

    def add_streams(self, num_streams):
        self.stream_pool.extend(cupy.cuda.Stream(non_blocking=True) for _ in range(num_streams))

    def allocate_streams(self, num_streams):
        streams = []
        n = 0
        for stream in self.stream_pool:
            if n < num_streams:
                if stream.done:
                    streams.append(stream)
                    n += 1
            else:
                break
        if n < num_streams:
            self.add_streams(num_streams - n)
            streams.extend(self.stream_pool[n - num_streams:])
        return streams

    def free_streams(self):
        del self.stream_pool

    def free_finished_stencils(self):
        #https://stackoverflow.com/questions/8037455/how-to-modify-python-collections-by-filtering-in-place
        for _ in range(len(self.invoked_stencils)):
            stencil = self.invoked_stencils.popleft()
            if not stencil.done_event.done:
                self.invoked_stencils.append(stencil)

    def _get_field_access_info(self, stencil: StencilObject):
        reads = set(v for v in stencil.field_info if stencil.field_info[v].access == AccessKind.READ_ONLY)
        writes = set(v for v in stencil.field_info if stencil.field_info[v].access == AccessKind.READ_WRITE)
        return reads, writes

    def get_dependencies(self, stencil: StencilObject) -> List[cupy.cuda.Event]:
        # R -> W, W -> W, W -> R
        dep_events = []
        reads, writes = self._get_field_access_info(stencil)
        for stencil_i in self.invoked_stencils:
            if not stencil_i.done_event.done:
                reads_i, writes_i = self._get_field_access_info(stencil_i.stencil)
                if writes.intersection(reads_i): # R -> W
                    dep_events.append(stencil_i.done_event)
                    continue
                if writes.intersection(writes_i): # W -> W
                    dep_events.append(stencil_i.done_event)
                    continue
                if reads.intersection(writes_i): # W -> R
                    dep_events.append(stencil_i.done_event)
                    continue
        return dep_events

    def add_invoked_stencil(self, stencil: StencilObject, done_event: cupy.cuda.Event):
        while len(self.invoked_stencils) >= self.max_invoked_stencils:
            sleep(0.5) # wait 0.5s for gpu computation
            self.free_finished_stencils()
        self.invoked_stencils.append(InvokedStencil(stencil=stencil, done_event=done_event))

    def wait_finish(self):
        while len(self.invoked_stencils) > 0:
            sleep(0.5)
            self.free_finished_stencils()
        self.free_streams()

def async_invoke(async_context: AsyncContext, stencil: StencilObject, *args, **kwargs):
    """
    Step 1: remove finished calls & free streams
    Step 2: analyse dependency
    Step 3: allocate streams
    Step 4: insert start & wait events in streams
    Step 5: invoke stencil
    Step 6: insert stop & wait events
    """
    async_context.free_finished_stencils()
    dep_events = async_context.get_dependencies(stencil)
    has_dependency_info = False
    if (hasattr(stencil, 'pyext_module')):
        num_kernels = stencil.pyext_module.num_kernels()
        has_dependency_info = stencil.pyext_module.has_dependency_info()
    else:
        raise TypeError(f"The stencil object {stencil.__module__}.{stencil.__name__} is not generated by GTC:CUDA backend")
    # count how many streams needed
    num_streams = num_kernels if has_dependency_info else 1  # TODO: reduce unnecessary streams
    stream_pool = async_context.allocate_streams(num_streams)
    # insert events
    for stream in stream_pool:
        for dep_event in dep_events:
            stream.wait_event(dep_event)
    # Launch stencil
    if num_streams == 1:
        streams = stream_pool * num_kernels
    else:
        streams = stream_pool
    stream_ptrs = [stream.ptr for stream in streams]
    stencil(*args, async_launch=True, streams=stream_ptrs, **kwargs)
    # insert events
    done_events = [cupy.cuda.Event(block=False, disable_timing=True) for _ in range(num_streams)]
    for i in range(1, num_streams):
        done_events[i].record(stream_pool[i])
        stream_pool[0].wait_event(done_events[i])
    done_events[0].record(stream_pool[0])
    # update async_ctx
    async_context.add_invoked_stencil(stencil, done_events[0])

class InsertAsync(ast.NodeTransformer):
    @classmethod
    def apply(cls, definition, ctx, num_streams_init = 20):
        maker = cls(definition, ctx, num_streams_init)
        maker.ast_root = maker.visit(maker.ast_root)
        maker.ast_root = maker.insert_init(maker.ast_root)
        return astor.to_source(maker.ast_root, add_line_information=True)

    def __init__(self, definition, ctx, num_streams_init):
        # check AsyncContext and async_invoke is in ctx as well
        #if ("AsyncContext" not in ctx) and ("async_invoke" not in ctx):
        #    raise ValueError("Please import `AsyncContext` and `async_invoke` first")
        self.ast_root = astor.code_to_ast(definition)
        self.stencil_ctx = {k: ctx[k] for k in ctx if isinstance(ctx[k], StencilObject) and
                                hasattr(ctx[k], "pyext_module")}
        self.num_streams_init = num_streams_init

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            keywords = [i.arg for i in node.keywords]
            if (func_name in self.stencil_ctx) and ('async_launch' not in keywords):
                return ast.copy_location(ast.Call(func=ast.Name(id="async_invoke", ctx=ast.Load()),
                                                  args=[ast.Name(id="async_context", ctx=ast.Load()), node.func]+node.args,
                                                  keywords=node.keywords), node)
        return node

    def insert_init(self, node: ast.FunctionDef):
        # num_streams_guess = min(20, sum(eval(stencil).pyext_module.num_kernels() for stencil in self.stencil_ctx))
        import_node = ast.ImportFrom(module='gt4py.gtgraph', names=[ast.alias(name='AsyncContext', asname=None),
                                                                    ast.alias(name='async_invoke', asname=None)],
                                     level=0, lineno=node.body[0].lineno)
        call_node = ast.Call(func=ast.Name(id="AsyncContext", ctx=ast.Load()),
                             args=[ast.Constant(value=self.num_streams_init, kind=None)],
                             keywords=[])
        start_node = ast.Assign(targets=[ast.Name(id='async_context', ctx=ast.Store())],
                                value=call_node, lineno=node.body[0].lineno)
        end_node = ast.Expr(value=ast.Call(func=ast.Attribute(
                                    value=ast.Name(id='async_context', ctx=ast.Load()),
                                attr='wait_finish', ctx=ast.Load()), args=[], keywords=[]),
                            lineno=node.body[-1].lineno)
        new_node = ast.copy_location(ast.FunctionDef(name=node.name, args=node.args,
                                                     body=[import_node, start_node]+node.body+[end_node],
                                                     decorator_list=node.decorator_list,
                                                     returns=node.returns,
                                                     type_comment=node.type_comment), node)
        return new_node