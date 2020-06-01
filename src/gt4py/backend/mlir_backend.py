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

import sys
import abc
import inspect
import numbers
import os
import re
import types

import enum
import jinja2
import numpy as np
import subprocess as sub

from collections import deque
from collections import OrderedDict

from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils

DEFAULT_FIELD_SIZE = 64
DEFAULT_HALO_SIZE = 4
DEFAULT_DIMENSIONS = 3

DOMAIN_AXES = gt_definitions.CartesianSpace.names


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Intent(enum.Enum):
    IN = 0
    OUT = 1
    INOUT = 2


class MLIRConverter(gt_ir.IRNodeVisitor):
    @classmethod
    def apply(cls, definition_ir):
        return cls()(definition_ir)

    def __call__(
        self,
        definition_ir,
        indent="  ",
        field_size=DEFAULT_FIELD_SIZE,
        halo_size=DEFAULT_HALO_SIZE,
    ):
        self.fields_ = []
        self.indent_ = indent
        self.field_size_ = field_size
        self.halo_size_ = halo_size

        self.stack_ = deque()
        self.body_ = []
        self.op_counts_ = {}
        self.out_counts_ = {}
        self.symbols_ = {}
        self.operations_ = OrderedDict()
        self.field_refs_ = OrderedDict()
        self.global_variables_ = OrderedDict()
        self.max_arg_ = 0

        return self.visit(definition_ir)

    def _add_operation(self, operation: AttrDict):
        key = str(operation)
        if key in self.symbols_:
            id = self.symbols_[key]
        else:
            type = operation.node_type
            prefix = "exp"
            if type == gt_ir.ScalarLiteral:
                prefix = "cst"
            elif type == gt_ir.FieldRef:
                prefix = "acc"
            elif type == gt_ir.VarRef:
                prefix = "var"
            elif type == gt_ir.TernaryOpExpr:
                prefix = "sel"

            if not prefix in self.op_counts_:
                self.op_counts_[prefix] = 0

            id = "%s%d" % (prefix, self.op_counts_[prefix])
            self.op_counts_[prefix] += 1
            self.operations_[id] = operation

        self.stack_.append(id)
        print("push('%s')" % id)

        return id

    def _emit_operation(self, id):
        operation = self.operations_[id]
        key = str(operation)
        if not key in self.symbols_:
            self.symbols_[key] = id
            type = operation.node_type

            if type == gt_ir.ScalarLiteral:
                # %cst0 = constant -4.000000e+00 : f64
                if operation.data_type.startswith('i'):
                    op_val = "%d" % int(operation.value)
                else:
                    op_val = "%e" % float(operation.value)
                code = f"%{id} = constant {op_val} : {operation.data_type}"

            elif type == gt_ir.FieldRef:
                # %a0 = stencil.access %arg1[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
                field = self._get_field(operation.name)
                field_ref = self.field_refs_[field.name]
                arg_name = "arg%d" % field_ref
                offset = ", ".join([str(index) for index in operation.offset])
                temp_type = f"!stencil.temp<{field.dimensions}{field.data_type}>"
                code = f"%{id} = stencil.access %{arg_name}[{offset}] : ({temp_type}) -> {operation.data_type}"

            elif type == gt_ir.UnaryOpExpr:
                operator = operation.op
                if operator == "-":
                    op_name = "neg"
                else:
                    raise NotImplementedError(f"Unimplemented unary operator '{operator}'")
                op_name += operation.data_type[0]
                code = f"%{id} = {op_name} %{operation.operand} : {operation.data_type}"

            elif type == gt_ir.BinOpExpr:
                # Use type of LHS for now...
                lhs = self.operations_[operation.lhs]
                data_type = lhs.data_type
                comp = ""
                operator = operation.op

                if operator == "+":
                    op_name = "add"
                elif operator == "-":
                    op_name = "sub"
                elif operator == "*":
                    op_name = "mul"
                elif operator == "/":
                    op_name = "div"
                elif operator == ">":
                    op_name = "cmp"
                    comp = "ogt"
                elif operator == "<":
                    op_name = "cmp"
                    comp = "olt"
                elif operator == ">=":
                    op_name = "cmp"
                    comp = "oge"
                elif operator == "<=":
                    op_name = "cmp"
                    comp = "ole"
                elif operator == "==":
                    op_name = "cmp"
                    comp = "eq"
                elif operator == "!=":
                    op_name = "cmp"
                    comp = "ne"
                else:
                    raise NotImplementedError(f"Unimplemented binary operator '{op_name}'")

                op_name += data_type[0]
                if len(comp) > 0:
                    op_name += ' "%s",' % comp
                code = f"%{id} = {op_name} %{operation.lhs}, %{operation.rhs} : {data_type}"

            elif type == gt_ir.TernaryOpExpr:
                # %s0 = select %e3, %c0, %e0 : f64
                code = f"%{id} = select %{operation.cond}, %{operation.lhs}, %{operation.rhs} : {operation.data_type}"

            else:  # Expr
                raise NotImplementedError("Unimplemented node type '%s'" % str(type))

            self.body_.append(code)
            print(code)

        return operation

    def _get_field(self, name):
        for field in self.fields_:
            if field.name == name:
                return field
        return None

    def _make_global_variables(self, parameters: list, externals: dict):
        global_variables = self.global_variables_
        for param in parameters:
            global_variables[param.name] = AttrDict(
                is_constexpr=False,
                value=None,
                data_type=param.data_type,
            )
            if param.data_type in [gt_ir.DataType.BOOL]:
                global_variables[param.name].value = param.init or False
            elif param.data_type in [
                gt_ir.DataType.INT8,
                gt_ir.DataType.INT16,
                gt_ir.DataType.INT32,
                gt_ir.DataType.INT64,
            ]:
                global_variables[param.name].value = param.init or 0
            elif param.data_type in [gt_ir.DataType.FLOAT32, gt_ir.DataType.FLOAT64]:
                global_variables[param.name].value = param.init or 0.0

    def reset(self):
        self.body_ = []
        self.op_counts_ = {}
        self.symbols_ = {}
        self.field_refs_.clear()
        self.stack_.clear()
        self.max_arg_ -= 1

    def _make_scalar_literal(self, value, data_type: gt_ir.DataType):
        assert data_type != gt_ir.DataType.INVALID
        literal_access_expr = AttrDict(
            value=float(value),
            data_type="f64", #str(data_type).replace("FLOAT", "f").replace("INT", "i"),
            node_type=gt_ir.ScalarLiteral,
        )
        id = self._add_operation(literal_access_expr)
        return literal_access_expr

    def visit_ScalarLiteral(self, node: gt_ir.ScalarLiteral, **kwargs):
        return self._make_scalar_literal(node.value, node.data_type)

    def visit_VarRef(self, node: gt_ir.VarRef, **kwargs):
        if node.name in self.global_variables_:
            # Replace globals with scalar literals...
            global_var = self.global_variables_[node.name]
            return self._make_scalar_literal(global_var.value, global_var.data_type)
        else:
            return AttrDict(name=node.name, is_external=True, type=gt_ir.VarRef)

    def visit_FieldDecl(self, node: gt_ir.FieldDecl, **kwargs):
        field = AttrDict(
            name=node.name,
            dimensions="?x" * len(node.axes),  # """.join(node.axes).lower(),
            is_temporary=(not node.is_api),
            data_type=str(node.data_type).replace("FLOAT", "f"),
            intent=Intent.IN,
            node_type=gt_ir.FieldDecl,
        )
        self.fields_.append(field)
        return field

    def visit_FieldRef(self, node: gt_ir.FieldRef, **kwargs):
        field = self._get_field(node.name)
        offset = [node.offset[ax] if ax in node.offset else 0 for ax in DOMAIN_AXES]

        field_access_expr = AttrDict(
            name=node.name, offset=offset, data_type=field.data_type, node_type=gt_ir.FieldRef
        )

        id = self._add_operation(field_access_expr)
        if node.name not in self.field_refs_:
            self.field_refs_[node.name] = self.max_arg_
            self.max_arg_ += 1

        return field_access_expr

    def visit_UnaryOpExpr(self, node: gt_ir.UnaryOpExpr, **kwargs):
        op = node.op.python_symbol
        operand = self.visit(node.arg)
        if op == "+":
            rhs = self.stack_[-1]
            return self.operations_[rhs]     # Do nothing for plus operator (leave operand untouched on the stack)
        else:
            # Pop operand off the stack...
            rhs = self.stack_.pop()
            print("pop(%s)" % rhs)
            self._emit_operation(rhs)
            data_type = self.operations_[rhs].data_type

            unary_op_expr = AttrDict(
                operand=rhs, op=op, data_type=data_type, node_type=gt_ir.UnaryOpExpr
            )
            id = self._add_operation(unary_op_expr)

            return unary_op_expr

    def visit_BinOpExpr(self, node: gt_ir.BinOpExpr, **kwargs):
        op = node.op.python_symbol
        if op == "**":
            if node.rhs.value == 2:
                op = "*"
                node.rhs = node.lhs
            else:
                raise NotImplementedError(f"Unsupported exponent value '{right.value}'")

        left = self.visit(node.lhs)
        right = self.visit(node.rhs)

        # Pop two items off the stack...
        rhs = self.stack_.pop()
        print("pop(%s)" % rhs)
        self._emit_operation(rhs)

        lhs = self.stack_.pop()
        print("pop(%s)" % lhs)
        self._emit_operation(lhs)
        data_type = self.operations_[lhs].data_type

        bin_op_expr = AttrDict(
            lhs=lhs, rhs=rhs, op=op, data_type=data_type, node_type=gt_ir.BinOpExpr
        )
        id = self._add_operation(bin_op_expr)

        return bin_op_expr

    def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr, **kwargs):
        cond = self.visit(node.condition)
        left = self.visit(node.then_expr)
        right = self.visit(node.else_expr)

        # Pop three items off the stack...
        rhs = self.stack_.pop()
        print("pop(%s)" % rhs)
        self._emit_operation(rhs)

        lhs = self.stack_.pop()
        print("pop(%s)" % lhs)
        self._emit_operation(lhs)

        cexp = self.stack_.pop()
        print("pop(%s)" % cexp)
        self._emit_operation(cexp)

        ternary_op_expr = AttrDict(
            lhs=lhs,
            rhs=rhs,
            cond=cexp,
            node_type=gt_ir.TernaryOpExpr,
            data_type=self.operations_[lhs].data_type,
        )

        id = self._add_operation(ternary_op_expr)

        return ternary_op_expr

    def visit_BlockStmt(self, node: gt_ir.BlockStmt, *, make_block=True, **kwargs):
        statements = [
            self.visit(stmt) for stmt in node.stmts
        ]  # if not isinstance(stmt, gt_ir.FieldDecl)]
        if make_block:
            stmts = AttrDict(statements=statements, node_type=gt_ir.BlockStmt)
        return statements

    def visit_Assign(self, node: gt_ir.Assign, **kwargs):
        self.reset()

        left = self.visit(node.target)
        right = self.visit(node.value)

        # At this point we expect two items on the stack, the rhs expression, and the destination store, lhs
        rhs = self.stack_.pop()
        print("pop(%s)" % rhs)

        lhs = self.stack_.pop()
        print("pop(%s)" % lhs)
        lhs_op = self.operations_[lhs]

        # rhs should be the return value for stencil.apply...
        rhs_op = self._emit_operation(rhs)
        self.body_.append(f"stencil.return %{rhs} : {rhs_op.data_type}")

        # Emit apply code...
        if self.file_:
            # %lap = stencil.apply %arg1 = %input : !stencil.temp<ijk,f64> {
            out_name = lhs_op.name
            out_field = self._get_field(out_name)

            if out_name in self.out_counts_:
                self.out_counts_[out_name] += 1
                out_name += str(self.out_counts_[out_name])
            else:
                self.out_counts_[out_name] = 0

            out_field.data_type = "f64"  # TODO: Where is AUTO data type coming from?
            temp_type = f"!stencil.temp<{out_field.dimensions}{out_field.data_type}>"
            indent = self.indent_

            line = (indent * 2) + f"%{out_name} = stencil.apply ("
            for field_name in self.field_refs_:
                if field_name != out_name:
                    line += f"%%arg%d = %%%s : {temp_type}, " % (
                        self.field_refs_[field_name],
                        field_name,
                    )

            line = line[0 : len(line) - 2] + f") -> {temp_type} " + "{\n"
            self.file_.write(line)

            for line in self.body_:
                self.file_.write((indent * 3) + line + "\n")
            self.file_.write((indent * 2) + "}\n")

        assign_expr = AttrDict(lhs=lhs, rhs=rhs, op="=", node_type=gt_ir.Assign)

        return assign_expr

    def visit_AugAssign(self, node: gt_ir.AugAssign):
        bin_op = gt_ir.BinOpExpr(lhs=node.target, op=node.op, rhs=node.value)
        assign = gt_ir.Assign(target=node.target, value=bin_op)
        return self.visit_Assign(assign)

    def visit_If(self, node: gt_ir.If, **kwargs):
        cond = sir_utils.make_expr_stmt(self.visit(node.condition))
        then_part = self.visit(node.main_body)
        else_part = self.visit(node.else_body)
        stmt = sir_utils.make_if_stmt(cond, then_part, else_part)
        return stmt

    def visit_AxisBound(self, node: gt_ir.AxisBound, **kwargs):
        assert isinstance(node.level, gt_ir.LevelMarker)
        return node.level, node.offset

    def visit_AxisInterval(self, node: gt_ir.AxisInterval, **kwargs):
        lower_level, lower_offset = self.visit(node.start)
        upper_level, upper_offset = self.visit(node.end)
        interval = AttrDict(
            lower_level=lower_level,
            upper_level=upper_level,
            lower_offset=lower_offset,
            upper_offset=upper_offset,
            node_type=gt_ir.AxisInterval,
        )
        return interval

    def visit_ComputationBlock(self, node: gt_ir.ComputationBlock, **kwargs):
        interval = self.visit(node.interval)
        body_ast = self.visit(node.body, make_block=False)
        loop_order = node.iteration_order

        vertical_region_stmt = AttrDict(
            body_ast=body_ast, interval=interval, loop_order=loop_order
        )

        return vertical_region_stmt

    def visit_StencilDefinition(self, node: gt_ir.StencilDefinition, **kwargs):
        stencils = []
        functions = []

        self._make_global_variables(node.parameters, node.externals)
        fields = [self.visit(field) for field in node.api_fields]

        stencil_name = node.name.split(".")[-1]
        file_name = os.path.join(os.getcwd(), "mlir", stencil_name + ".mlir")
        self.file_ = open(file_name, "w")

        if self.file_:
            indent = self.indent_
            # TODO: Determine whether field is output based on AST traversal...
            if 'Grad' in stencil_name:
                fields[0].intent = fields[1].intent = Intent.INOUT
            elif 'UV' in stencil_name:
                fields[-2].intent = fields[-1].intent = Intent.OUT
            else:
                fields[-1].intent = Intent.OUT

            self.file_.write("module {\n")
            self.file_.write(indent + f"func @{stencil_name}(")
            field_defs = []
            for field in fields:
                if not field.is_temporary:
                    field_defs.append(
                        f"%{field.name}_fd : !stencil.field<{field.dimensions}{field.data_type}>"
                    )
            self.file_.write(", ".join(field_defs) + ") attributes { stencil.program } {\n")

            # Assert fields...
            for field in fields:
                if not field.is_temporary:
                    # stencil.assert %input_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
                    halo_size = self.halo_size_
                    field_extent = halo_size + self.field_size_
                    self.file_.write(
                        (indent * 2)
                        + f"stencil.assert %{field.name}_fd ([-{halo_size}, -{halo_size}, -{halo_size}]:[{field_extent}, {field_extent}, {field_extent}]) : !stencil.field<{field.dimensions}{field.data_type}>\n"
                    )
            self.file_.write("\n")

            # Load input fields...
            for field in fields:
                if not field.is_temporary and field.intent != Intent.OUT:
                    # %input = stencil.load %input_fd : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
                    field_type = f"!stencil.field<{field.dimensions}{field.data_type}>"
                    temp_type = f"!stencil.temp<{field.dimensions}{field.data_type}>"
                    self.file_.write(
                        (indent * 2)
                        + f"%{field.name} = stencil.load %{field.name}_fd : ({field_type}) -> {temp_type}\n"
                    )
            self.file_.write("\n")

        ast = [self.visit(computation) for computation in node.computations]

        if self.file_:
            # Store output fields...
            origin = "0, 0, 0"
            field_size_str = str(self.field_size_)
            field_sizes = field_size_str + ", " + field_size_str + ", " + field_size_str

            for field in fields:
                if not field.is_temporary and field.intent != Intent.IN:
                    # stencil.store %output to %output_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
                    field_type = f"!stencil.field<{field.dimensions}{field.data_type}>"
                    temp_type = f"!stencil.temp<{field.dimensions}{field.data_type}>"
                    self.file_.write(
                        (indent * 2)
                        + f"stencil.store %{field.name} to %{field.name}_fd([{origin}] : [{field_sizes}]) : {temp_type} to {field_type}\n"
                    )

            self.file_.write((indent * 2) + "return\n }\n}\n")
            self.file_.close()

        stencil = AttrDict(name=stencil_name, ast=ast, fields=fields)
        stencils.append(stencil)

        mlir = AttrDict(
            file_name=file_name,
            grid_type="Cartesian",
            functions=functions,
            stencils=stencils,
            global_variables=self.global_variables_,
        )

        return mlir


@gt_backend.register
class MLIRBackend(gt_backend.BasePyExtBackend):

    MLIR_BACKEND_NS = "mlir"
    MLIR_BACKEND_NAME = "mlir"
    MLIR_BACKEND_OPTS = {
        "add_profile_info": {"versioning": True},
        "clean": {"versioning": False},
        "debug_mode": {"versioning": True},
        "verbose": {"versioning": False},
    }

    GT_BACKEND_T = "x86"
    MODULE_GENERATOR_CLASS = gt_backend.PyExtModuleGenerator

    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
    TEMPLATE_FILES = {
        "computation.hpp": "computation.hpp.in",
        "computation.src": "dawn_computation.src.in",
        "bindings.cpp": "bindings.cpp.in",
    }

    _DATA_TYPE_TO_CPP = {
        gt_ir.DataType.INT8: "int",
        gt_ir.DataType.INT16: "int",
        gt_ir.DataType.INT32: "int",
        gt_ir.DataType.INT64: "int",
        gt_ir.DataType.FLOAT32: "double",
        gt_ir.DataType.FLOAT64: "double",
    }

    name = MLIR_BACKEND_NAME
    options = MLIR_BACKEND_OPTS
    storage_info = gt_backend.GTX86Backend.storage_info

    @classmethod
    def generate(
        cls,
        stencil_id: gt_definitions.StencilID,
        definition_ir: gt_ir.StencilDefinition,
        definition_func: types.FunctionType,
        options: gt_definitions.BuildOptions,
    ):
        # TODO: move this import to the top and find a better way to avoid circular imports
        from gt4py import gt_src_manager

        cls._check_options(options)

        module_kwargs = {"implementation_ir": None}
        pyext_module_name, pyext_file_path = cls.generate_extension(
            stencil_id, definition_ir, options, module_kwargs=module_kwargs
        )

        # Generate and return the Python wrapper class
        return cls._generate_module(
            stencil_id,
            definition_ir,
            definition_func,
            options,
            extra_cache_info={"pyext_file_path": pyext_file_path},
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
            **module_kwargs,
        )

    @classmethod
    def generate_extension_sources(
        cls, stencil_id, definition_ir, options, gt_backend_t, default_opts=True
    ):
        mlir = MLIRConverter.apply(definition_ir)
        stencil_short_name = stencil_id.qualified_name.split(".")[-1]

        backend_opts = dict(**options.backend_opts)
        mlir_backend = cls.MLIR_BACKEND_NAME

        # Define tools
        tools = AttrDict(
            opt="oec-opt",
            translate="mlir-translate",
            compile="llc",
            wrapper="../open-earth-compiler/runtime/oec-runtime.cpp",
            clang="clang++-9",
        )

        # Define optimizations
        unroll_factor = 2
        unroll_index = 1
        optimizations = [
            "--canonicalize",
            "--stencil-inlining",
            "--cse",
            "--pass-pipeline=stencil-unrolling{unroll-factor=%d unroll-index=%d}"
            % (unroll_factor, unroll_index),
            "--stencil-shape-inference",
            "--cse",
            "--convert-stencil-to-std",
            "--cse",
        ]

        is_cuda = True #"cuda" in mlir_backend
        if is_cuda:
            optimizations.extend(
                ["--stencil-loop-mapping=block-sizes=128,1,1", "--convert-parallel-loops-to-gpu"]
            )
        optimizations.extend(["--lower-affine", "--convert-scf-to-std"])

        if is_cuda:
            optimizations.extend(["--gpu-kernel-outlining"])
        optimizations.extend(["--cse", "--canonicalize"])

        if is_cuda:
            optimizations.extend(
                ["--stencil-gpu-to-cubin", "--stencil-gpu-to-cuda", "--cse", "--canonicalize"]
            )
        else:
            optimizations.extend(["--convert-std-to-llvm=emit-c-wrappers"])

        # Perform stencil optimizations and lower MLIR
        mlir_in = mlir.file_name
        mlir_out = os.path.splitext(mlir_in)[0] + "_lower.mlir"
        encoding = "utf-8"
        gen_files = True # not os.path.exists(mlir_out)

        command = [tools.opt]
        command.extend(optimizations)
        command.extend([mlir_in])
        if gen_files:
            output = sub.run(command, capture_output=True)
            if len(output.stderr) > 0:  # Check for error...
                raise RuntimeError("OEC-ERROR: " + str(output.stderr, encoding))
            with open(mlir_out, "w") as out:
                out.write(str(output.stdout, encoding))

        # Translate lowered MLIR to LLVM
        mlir_llvm = os.path.splitext(mlir_in)[0] + "_llvm.mlir"
        command = [tools.translate, "--mlir-to-llvmir", mlir_out]
        if gen_files:
            output = sub.run(command, capture_output=True)
            if len(output.stderr) > 0:  # Check for error...
                raise RuntimeError("MLIR-ERROR: " + str(output.stderr, encoding))
            with open(mlir_llvm, "w") as out:
                out.write(str(output.stdout, encoding))

        # Compile LLVM to assembly...
        debug_mode = options.backend_opts.get("debug_mode", False)
        opt_level = "-O" + str(0 if debug_mode else 3)
        llvm_asm = os.path.splitext(mlir_llvm)[0] + ".s"
        if gen_files:
            command = [tools.compile, opt_level, mlir_llvm, "-o", llvm_asm]
            output = sub.run(command, capture_output=True)
            if len(output.stderr) > 0:  # Check for error...
                raise RuntimeError("LLVM-ERROR: " + str(output.stderr, encoding))

        # Now how to combine all this with gt4py/pybind11 and compile with clang...
        # input1 = output
        # input2 = abs_path(temp, experiment + ".cpp")
        # input3 = wrappers
        # replace_template_config(size, height, bound_size, type, abs_path(
        #    benchmarks, kernel + ".cpp"), input2, num_measurements)
        binary = os.path.splitext(llvm_asm)[0] + ".o"
        command = [tools.clang, "-c", opt_level, llvm_asm, "-o", binary]
        output = sub.run(command, capture_output=True)
        if len(output.stderr) > 0:  # Check for error...
            raise RuntimeError("CLANG-ERROR: " + str(output.stderr, encoding))

        command = [tools.clang, opt_level, binary, tools.wrapper]
        if is_cuda:
            if "CUDA_SRC" in os.environ:
                cuda_path = os.environ["CUDA_SRC"]
                command.extend([f"-I{cuda_path}/include -L{cuda_path}/lib"])
            command.extend(["-lcudart", "-lcuda"])
        output = sub.run(command, capture_output=True)
        if len(output.stderr) > 0:  # Check for error...
            raise RuntimeError("CLANG-ERROR: " + str(output.stderr, encoding))

        source = ""  # dawn4py.compile(mlir, **dawn_opts)
        stencil_unique_name = cls.get_pyext_class_name(stencil_id)
        module_name = cls.get_pyext_module_name(stencil_id)
        pyext_sources = {f"_dawn_{stencil_short_name}.hpp": source}

        arg_fields = [
            {"name": field.name, "dtype": cls._DATA_TYPE_TO_CPP[field.data_type], "layout_id": i}
            for i, field in enumerate(definition_ir.api_fields)
        ]
        header_file = "computation.hpp"

        parameters = []
        for parameter in definition_ir.parameters:
            if parameter.data_type in [gt_ir.DataType.BOOL]:
                dtype = "bool"
            elif parameter.data_type in [
                gt_ir.DataType.INT8,
                gt_ir.DataType.INT16,
                gt_ir.DataType.INT32,
                gt_ir.DataType.INT64,
            ]:
                dtype = "int"
            elif parameter.data_type in [gt_ir.DataType.FLOAT32, gt_ir.DataType.FLOAT64]:
                dtype = "double"
            else:
                assert False, "Wrong data_type for parameter"
            parameters.append({"name": parameter.name, "dtype": dtype})

        template_args = dict(
            arg_fields=arg_fields,
            backend=mlir_backend,
            gt_backend=gt_backend_t,
            header_file=header_file,
            module_name=module_name,
            parameters=parameters,
            stencil_short_name=stencil_short_name,
            stencil_unique_name=stencil_unique_name,
        )

        for key, file_name in cls.TEMPLATE_FILES.items():
            with open(os.path.join(cls.TEMPLATE_DIR, file_name), "r") as f:
                template = jinja2.Template(f.read())
                pyext_sources[key] = template.render(**template_args)

        return pyext_sources

    @classmethod
    def _generate_module(
        cls,
        stencil_id,
        definition_ir,
        definition_func,
        options,
        *,
        extra_cache_info=None,
        **kwargs,
    ):
        if options.dev_opts.get("code-generation", True):
            # Dawn backends do not use the internal analysis pipeline, so a custom
            # wrapper_info object should be passed to the module generator
            assert "implementation_ir" in kwargs

            info = {}
            if definition_ir.sources is not None:
                info["sources"].update(
                    {
                        key: gt_utils.text.format_source(value, line_length=100)
                        for key, value in definition_ir.sources
                    }
                )
            else:
                info["sources"] = {}

            parallel_axes = definition_ir.domain.parallel_axes or []
            sequential_axis = definition_ir.domain.sequential_axis.name
            domain_info = gt_definitions.DomainInfo(
                parallel_axes=tuple(ax.name for ax in parallel_axes),
                sequential_axis=sequential_axis,
                ndims=len(parallel_axes) + (1 if sequential_axis else 0),
            )
            info["domain_info"] = repr(domain_info)

            info["field_info"] = field_info = {}
            info["parameter_info"] = parameter_info = {}

            fields = {item.name: item for item in definition_ir.api_fields}
            parameters = {item.name: item for item in definition_ir.parameters}

            halo_size = kwargs.pop("halo_size")
            boundary = gt_definitions.Boundary(
                ([(halo_size, halo_size)] * len(domain_info.parallel_axes)) + [(0, 0)]
            )

            for arg in definition_ir.api_signature:
                if arg.name in fields:
                    field_info[arg.name] = gt_definitions.FieldInfo(
                        access=gt_definitions.AccessKind.READ_WRITE,
                        dtype=fields[arg.name].data_type.dtype,
                        boundary=boundary,
                    )
                else:
                    parameter_info[arg.name] = gt_definitions.ParameterInfo(
                        dtype=parameters[arg.name].data_type.dtype
                    )

            if definition_ir.externals:
                info["gt_constants"] = {
                    name: repr(value)
                    for name, value in definition_ir.externals.items()
                    if isinstance(value, numbers.Number)
                }
            else:
                info["gt_constants"] = {}

            info["gt_options"] = {
                key: value for key, value in options.as_dict().items() if key not in ["build_info"]
            }

            info["unreferenced"] = {}

            generator = cls.MODULE_GENERATOR_CLASS(cls)
            module_source = generator(
                stencil_id, definition_ir, options, wrapper_info=info, **kwargs
            )

            file_name = cls.get_stencil_module_path(stencil_id)
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "w") as f:
                f.write(module_source)
            extra_cache_info = extra_cache_info or {}
            cls.update_cache(stencil_id, extra_cache_info)

        return cls._load(stencil_id, definition_func)

    @classmethod
    def _generic_generate_extension(
        cls, stencil_id, definition_ir, options, *, uses_cuda=False, **kwargs
    ):
        module_kwargs = kwargs["module_kwargs"]
        dawn_src_file = f"_dawn_{stencil_id.qualified_name.split('.')[-1]}.hpp"

        # Generate source
        if options.dev_opts.get("code-generation", True):
            gt_pyext_sources = cls.generate_extension_sources(
                stencil_id, definition_ir, options, cls.GT_BACKEND_T
            )
        else:
            # Pass NOTHING to the builder means try to reuse the source code files
            gt_pyext_sources = {key: gt_utils.NOTHING for key in cls.TEMPLATE_FILES.keys()}
            gt_pyext_sources[dawn_src_file] = gt_utils.NOTHING

        final_ext = ".cu" if uses_cuda else ".cpp"
        keys = list(gt_pyext_sources.keys())
        for key in keys:
            if key.split(".")[-1] == "src":
                new_key = key.replace(".src", final_ext)
                gt_pyext_sources[new_key] = gt_pyext_sources.pop(key)

        # Build extension module
        pyext_opts = dict(
            verbose=options.backend_opts.get("verbose", False),
            clean=options.backend_opts.get("clean", False),
            debug_mode=options.backend_opts.get("debug_mode", gt_backend.DEBUG_MODE),
            add_profile_info=options.backend_opts.get("add_profile_info", False),
        )
        include_dirs = [
            "{install_dir}/_external_src".format(
                install_dir=os.path.dirname(inspect.getabsfile(dawn4py))
            )
        ]

        return cls.build_extension_module(
            stencil_id,
            gt_pyext_sources,
            pyext_opts,
            pyext_extra_include_dirs=include_dirs,
            uses_cuda=uses_cuda,
        )

    @classmethod
    def generate_extension(cls, stencil_id, definition_ir, options, **kwargs):
        return cls._generic_generate_extension(
            stencil_id, definition_ir, options, uses_cuda=False, **kwargs
        )
