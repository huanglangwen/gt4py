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
from typing import Any, Callable, Dict, Tuple, List, Deque, Optional
from gt4py.stencil_object import StencilObject
from gt4py import AccessKind
from gt4py.storage import Storage
from collections import deque
from dataclasses import dataclass
from time import sleep
from graphviz import Digraph
from uuid import uuid4

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
    access_info: Dict[str, AccessKind]
    done_event: cupy.cuda.Event
    id: int

class AsyncContext():
    def __init__(self, num_streams, max_invoked_stencils = 50, blocking = False, sleep_time = 0.5, graph_record = False, name = None):
        self.stream_pool: List[Optional[cupy.cuda.Stream]] = []
        self.add_streams(num_streams)
        self.invoked_stencils: Deque[InvokedStencil] = deque()
        self.max_invoked_stencils: int = max_invoked_stencils
        self.known_num_fields: int = 0
        self.blocking: bool = blocking # synchronize all stencils, debug only
        self.sleep_time: float = sleep_time # (second) longer sleep saves cpu cycle but lower resolution in timing
        self._graph_record: bool = False
        self._stencil_id_ticket = 0
        if name is None:
            name = uuid4().hex[:5] # get a (almost) unique name
        self.name: str = name
        if graph_record:
            self.graph_record()
            
    def get_stencil_id(self):
        stencil_id = self._stencil_id_ticket
        self._stencil_id_ticket += 1
        return stencil_id

    def get_node_name(self, node_name: str, stencil_id: int):
        return f'{node_name}_cluster_{stencil_id}'

    def graph_record(self, filename: Optional[str] = None):
        self._graph_record = True
        if filename is None:
            filename = f"{self.name}_graph.gv"
        self._graph = Digraph(name=f"{self.name}_graph", filename=filename)

    def graph_add_stencil(self, stencil: StencilObject, access_info: Dict[str, AccessKind], stencil_id: int):
        args_name = ",".join(k for k in access_info)
        stencil_name = f"{stencil.options['name']}_{stencil_id}({args_name})" # {stencil.options['module']}_
        row_ind, col_ind = self.get_kernel_dependencies(stencil)
        num_kernels = len(row_ind) - 1
        with self._graph.subgraph(name=f'cluster_{stencil_id}') as c:
            c.attr(style='filled', color='lightgrey')
            c.node_attr.update(style='filled', color='white')
            start_name = self.get_node_name('start', stencil_id)
            end_name = self.get_node_name('end', stencil_id)
            c.node(start_name, label='start', shape='box')
            c.node(end_name, label='end', shape='box')
            c.attr(label=stencil_name)
            for i in range(num_kernels):
                cols = col_ind[row_ind[i]: row_ind[i+1]]
                name_i = self.get_node_name(f'kernel{i}', stencil_id)
                c.node(name_i, label=f"kernel {i}")
                c.edge(start_name, name_i)
                c.edge(name_i, end_name)
                for j in cols:
                    name_j = self.get_node_name(f'kernel{j}', stencil_id)
                    c.edge(name_j, name_i)

    def graph_add_stencil_dependency(self, stencil_id_i: int, stencil_id_j: int):
        """
         i is dependent on j, j -> i
        """
        node_name_i = self.get_node_name('start', stencil_id_i)
        node_name_j = self.get_node_name("end", stencil_id_j)
        self._graph.edge(node_name_j, node_name_i, style='bold', color='blue')

    def graph_stop_record(self):
        self._graph_record = False

    def graph_view(self, cleanup=True):
        if isinstance(self._graph, Digraph):
            self._graph.view(cleanup=cleanup)

    def graph_save(self, cleanup=True):
        if isinstance(self._graph, Digraph):
            self._graph.render(cleanup=cleanup, format="pdf")

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

    def free_finished_stencils(self):
        #https://stackoverflow.com/questions/8037455/how-to-modify-python-collections-by-filtering-in-place
        for _ in range(len(self.invoked_stencils)):
            stencil = self.invoked_stencils.popleft()
            if not stencil.done_event.done:
                self.invoked_stencils.append(stencil)

    def set_field_name(self, field: Storage, field_name: str):
        field._field_name = field_name

    def get_field_name(self, field: Storage):
        if not hasattr(field, '_field_name'):
            name = f"field_{self.known_num_fields}"
            self.known_num_fields += 1
            field._field_name = name
        return field._field_name

    def get_field_access_info(self, stencil: StencilObject, *args, **kwargs) -> Dict[str, AccessKind]:
        access_info = {}
        stencil_sig_bind = inspect.signature(stencil).bind(*args, **kwargs)
        for k, v in stencil_sig_bind.arguments.items():
            if isinstance(v, Storage):
                field_name = self.get_field_name(v)
                access_kind = stencil.field_info[k].access
                access_info[field_name] = access_kind
        return access_info

    def get_kernel_dependencies(self, stencil: StencilObject) -> Tuple[List[int], List[int]]: #(row_ind, col_ind)
        assert hasattr(stencil, "pyext_module")
        num_kernels = stencil.pyext_module.num_kernels()
        col_ind = []
        row_ind = [0] * (num_kernels + 1)
        if stencil.pyext_module.has_dependency_info():
            row_ind = stencil.pyext_module.dependency_row_ind()
            col_ind = stencil.pyext_module.dependency_col_ind()
            assert len(row_ind) == num_kernels + 1, "CSR format in dependency data is broken"
        return row_ind, col_ind

    def get_dependencies(self, access_info: Dict[str, AccessKind], stencil_id: int) -> List[cupy.cuda.Event]:
        # R -> W, W -> W, W -> R
        dep_events = []
        reads = {k for k in access_info if access_info[k] == AccessKind.READ_ONLY}
        writes = {k for k in access_info if access_info[k] == AccessKind.READ_WRITE}
        for stencil_i in self.invoked_stencils:
            if not stencil_i.done_event.done:
                access_info_i = stencil_i.access_info
                reads_i = {k for k in access_info_i if access_info_i[k] == AccessKind.READ_ONLY}
                writes_i = {k for k in access_info_i if access_info_i[k] == AccessKind.READ_WRITE}
                dep_flag = False
                if writes.intersection(reads_i): # R -> W
                    dep_flag = True
                if writes.intersection(writes_i): # W -> W
                    dep_flag = True
                if reads.intersection(writes_i): # W -> R
                    dep_flag = True
                if dep_flag:
                    dep_events.append(stencil_i.done_event)
                    if self._graph_record:
                        self.graph_add_stencil_dependency(stencil_id, stencil_i.id)
        return dep_events

    def add_invoked_stencil(self, stencil: StencilObject, access_info: Dict[str, AccessKind], done_event: cupy.cuda.Event, stencil_id: int):
        while len(self.invoked_stencils) >= self.max_invoked_stencils:
            sleep(self.sleep_time) # wait for gpu computation
            self.free_finished_stencils()
        self.invoked_stencils.append(InvokedStencil(stencil=stencil, access_info=access_info, done_event=done_event, id=stencil_id))

    def wait(self):
        while len(self.invoked_stencils) > 0:
            sleep(self.sleep_time)
            self.free_finished_stencils()

    def wait_finish(self):
        self.wait()
        for i in range(len(self.stream_pool)):
            self.stream_pool[i] = None

    def schedule(self, stencil: StencilObject, *args, **kwargs):
        if self.blocking:
            stencil(*args, **kwargs)
        else:
            self.async_schedule(stencil, *args, **kwargs)

    def async_schedule(self, stencil: StencilObject, *args, **kwargs):
        """
        Step 0: remove finished calls & free streams
        Step 1: mark fields if first meet
        Step 2: analyse dependency
        Step 3: allocate streams
        Step 4: insert start & wait events in streams
        Step 5: invoke stencil
        Step 6: insert stop & wait events
        """
        # remove finished calls
        self.free_finished_stencils()

        # check stencil obj is generated by the right backend
        has_kernel_dependency_info = False

        assert hasattr(stencil, 'pyext_module'), f"The stencil object {stencil.__module__}.{stencil.__name__} is not generated by GTC:CUDA backend"
        num_kernels = stencil.pyext_module.num_kernels()
        has_kernel_dependency_info = stencil.pyext_module.has_dependency_info()

        # resolve dependency
        access_info = self.get_field_access_info(stencil, *args, **kwargs)
        stencil_id = self.get_stencil_id()
        if self._graph_record:
            self.graph_add_stencil(stencil, access_info, stencil_id)
        dep_events = self.get_dependencies(access_info, stencil_id)

        # count how many streams needed
        num_streams = num_kernels if has_kernel_dependency_info else 1  # TODO: reduce unnecessary streams
        stream_pool = self.allocate_streams(num_streams)

        # insert events for waiting dependencies
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

        # insert events to record when the stencil finishes
        done_events = [cupy.cuda.Event(block=False, disable_timing=True) for _ in range(num_streams)]
        for i in range(1, num_streams):
            done_events[i].record(stream_pool[i])
            stream_pool[0].wait_event(done_events[i])
        done_events[0].record(stream_pool[0])

        # update async_ctx
        self.add_invoked_stencil(stencil, access_info, done_events[0], stencil_id)

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
                return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='async_context', ctx=ast.Load()),
                                                                     attr='schedule', ctx=ast.Load()),
                                                  args=[ast.Name(id="async_context", ctx=ast.Load()), node.func]+node.args,
                                                  keywords=node.keywords), node)
        return node

    def insert_init(self, node: ast.FunctionDef):
        # num_streams_guess = min(20, sum(eval(stencil).pyext_module.num_kernels() for stencil in self.stencil_ctx))
        import_node = ast.ImportFrom(module='gt4py.gtgraph', names=[ast.alias(name='AsyncContext', asname=None)],
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