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

from typing import Any, Collection, Union

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from eve.concepts import LeafNode
from gtc.common import BuiltInLiteral, DataType, LevelMarker, NativeFunction, UnaryOperator
from gtc.cuir import cuir


class CUIRCodegen(codegen.TemplatedGenerator):

    LocalScalar = as_fmt("{dtype} {name};")

    FieldDecl = as_fmt("{name}")

    Temporary = as_fmt("{name}")

    AssignStmt = as_fmt("{left} = {right};")

    FieldAccess = as_mako(
        "*${name if not offset else f'sid::multi_shifted<tag::{name}>({name}, m_strides, {offset})'}"
    )

    ScalarAccess = as_fmt("{name}")

    CartesianOffset = as_mako(
        "${'' if _this_node.i == _this_node.j == _this_node.k == 0 else f'offsets({i}_c, {j}_c, {k}_c)'}"
    )

    BinaryOp = as_fmt("({left} {op} {right})")

    UnaryOp = as_fmt("({op}{expr})")

    TernaryOp = as_fmt("({cond} ? {true_expr} : {false_expr})")

    Cast = as_fmt("static_cast<{dtype}>({expr})")

    def visit_BuiltInLiteral(self, builtin: BuiltInLiteral, **kwargs: Any) -> str:
        if builtin == BuiltInLiteral.TRUE:
            return "true"
        elif builtin == BuiltInLiteral.FALSE:
            return "false"
        raise NotImplementedError("Not implemented BuiltInLiteral encountered.")

    Literal = as_mako("static_cast<${dtype}>(${value})")

    def visit_NativeFunction(self, func: NativeFunction, **kwargs: Any) -> str:
        if func == NativeFunction.SQRT:
            return "math::sqrt"
        elif func == NativeFunction.MIN:
            return "math::min"
        elif func == NativeFunction.MAX:
            return "math::max"
        raise NotImplementedError("Not implemented NativeFunction encountered.")

    NativeFuncCall = as_mako("${func}(${','.join(args)})")

    def visit_DataType(self, dtype: DataType, **kwargs: Any) -> str:
        if dtype == DataType.INT64:
            return "long long"
        elif dtype == DataType.FLOAT64:
            return "double"
        elif dtype == DataType.FLOAT32:
            return "float"
        elif dtype == DataType.BOOL:
            return "bool"
        raise NotImplementedError("Not implemented NativeFunction encountered.")

    def visit_UnaryOperator(self, op: UnaryOperator, **kwargs: Any) -> str:
        if op == UnaryOperator.NOT:
            return "!"
        elif op == UnaryOperator.NEG:
            return "-"
        elif op == UnaryOperator.POS:
            return "+"
        raise NotImplementedError("Not implemented UnaryOperator encountered.")

    Extent = as_fmt("extent<{iminus}, {iplus}, {jminus}, {jplus}>")

    HorizontalExecution = as_mako(
        """
        // ${id_}
        if (validator(${extent}())${' && ' + mask if _this_node.mask else ''}) {
            ${'\\n'.join(declarations)}
            ${'\\n'.join(body)}
        }
        """
    )

    def visit_AxisBound(self, node: cuir.AxisBound, **kwargs: Any) -> str:
        if node.level == LevelMarker.START:
            return f"{node.offset}"
        if node.level == LevelMarker.END:
            return f"k_size + {node.offset}"
        raise ValueError("Cannot handle dynamic levels")

    VerticalLoopSection = as_mako(
        """
        // ${id_}
        if (k_block >= ${start} && k_block < ${end}) {
            ${'\\n__syncthreads();\\n'.join(horizontal_executions)}
        }
        """
    )

    def visit_VerticalLoop(
        self, node: cuir.VerticalLoop, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        return self.generic_visit(
            node,
            accesses=node.iter_tree().if_isinstance(cuir.FieldAccess).getattr("name").to_set(),
            **kwargs,
        )

    VerticalLoop = as_mako(
        """
        template <class Sid>
        struct loop_${id_}_f {
            sid::ptr_holder_type<Sid> m_ptr_holder;
            sid::strides_type<Sid> m_strides;
            int k_size;

            template <class Validator>
            GT_FUNCTION_DEVICE void operator()(const int i_block,
                                               const int j_block,
                                               Validator validator) const {
                const int k_block = blockIdx.z;
                auto ptr = m_ptr_holder();
                sid::shift(ptr,
                           sid::get_stride<sid::blocked_dim<dim::i>>(m_strides),
                           blockIdx.x);
                sid::shift(ptr,
                           sid::get_stride<sid::blocked_dim<dim::j>>(m_strides),
                           blockIdx.y);
                sid::shift(ptr,
                           sid::get_stride<dim::i>(m_strides),
                           i_block);
                sid::shift(ptr,
                           sid::get_stride<dim::j>(m_strides),
                           j_block);
                sid::shift(ptr,
                           sid::get_stride<dim::k>(m_strides),
                           k_block);

                % for acc in accesses:
                    auto &&${acc} = device::at_key<tag::${acc}>(ptr);
                % endfor

                % for section in sections:
                    ${section}
                % endfor
            }
        };
        """
    )

    Kernel = as_mako(
        """
        % for vertical_loop in vertical_loops:
            ${vertical_loop}
        % endfor

        template <${', '.join(f'class Loop{vl.id_}' for vl in _this_node.vertical_loops)}>
        struct kernel_${id_}_f {
            % for vertical_loop in _this_node.vertical_loops:
                Loop${vertical_loop.id_} m_${vertical_loop.id_};
            % endfor

            template <class Validator>
            GT_FUNCTION_DEVICE void operator()(const int i_block,
                                               const int j_block,
                                               Validator validator) const {
                % for vertical_loop in _this_node.vertical_loops:
                    m_${vertical_loop.id_}(i_block, j_block, validator);
                % endfor
            }
        };

        """
    )

    def visit_Program(self, node: cuir.Program, **kwargs: Any) -> Union[str, Collection[str]]:
        return self.generic_visit(
            node,
            declarations_dtypes=[self.visit(d.dtype, **kwargs) for d in node.declarations],
            max_extent=self.visit(cuir.Extent.union(*node.iter_tree().if_isinstance(cuir.Extent))),
        )

    Program = as_mako(
        """#include <array>
        #include <gridtools/common/cuda_util.hpp>
        #include <gridtools/common/gt_math.hpp>
        #include <gridtools/common/host_device.hpp>
        #include <gridtools/common/hymap.hpp>
        #include <gridtools/common/integral_constant.hpp>
        #include <gridtools/sid/allocator.hpp>
        #include <gridtools/sid/block.hpp>
        #include <gridtools/sid/composite.hpp>
        #include <gridtools/sid/multi_shift.hpp>
        #include <gridtools/stencil/common/dim.hpp>
        #include <gridtools/stencil/common/extent.hpp>
        #include <gridtools/stencil/gpu/launch_kernel.hpp>
        #include <gridtools/stencil/gpu/shared_allocator.hpp>
        #include <gridtools/stencil/gpu/tmp_storage_sid.hpp>

        namespace ${name}_impl_{
            using namespace gridtools;
            using namespace literals;
            using namespace stencil;

            using domain_t = std::array<unsigned, 3>;
            using i_block_size_t = integral_constant<int_t, 64>;
            using j_block_size_t = integral_constant<int_t, 8>;

            template <class Storage>
            auto block(Storage storage) {
                return sid::block(std::move(storage),
                    tuple_util::make<hymap::keys<dim::i, dim::j>::values>(
                        i_block_size_t(), j_block_size_t()));
            }

            template <class I, class J, class K>
            GT_FUNCTION_DEVICE auto offsets(I i, J j, K k) {
                return tuple_util::device::make<hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k);
            }

            namespace tag {
                % for p in params + declarations:
                    struct ${p} {};
                % endfor
            }

            % for kernel in kernels:
                ${kernel}
            % endfor

            auto ${name}(domain_t domain){
                return [domain](${','.join(f'auto&& {p}' for p in params)}){
                    auto tmp_alloc = sid::device::make_cached_allocator(&cuda_util::cuda_malloc<char[]>);
                    gpu_backend::shared_allocator shared_alloc;
                    auto i_blocks = (domain[0] + i_block_size_t() - 1) / i_block_size_t();
                    auto j_blocks = (domain[1] + j_block_size_t() - 1) / j_block_size_t();
                    % for d, dtype in zip(declarations, declarations_dtypes):
                        auto ${d} = gpu_backend::make_tmp_storage<${dtype}>(
                            1_c,
                            i_block_size_t(),
                            j_block_size_t(),
                            ${max_extent}(),
                            i_blocks,
                            j_blocks,
                            domain[2],
                            tmp_alloc);
                    % endfor

                    % for kernel in _this_node.kernels:
                        % for vertical_loop in kernel.vertical_loops:
                            auto composite_${vertical_loop.id_} = sid::composite::make<
                                    ${', '.join(f'tag::{p}' for p in params + declarations)}
                                >(
                                    ${', '.join([f'block({p})' for p in params] + list(declarations))}
                                );
                            loop_${vertical_loop.id_}_f<decltype(composite_${vertical_loop.id_})> loop_${vertical_loop.id_}{
                                sid::get_origin(composite_${vertical_loop.id_}),
                                sid::get_strides(composite_${vertical_loop.id_}),
                                domain[2]
                            };
                        % endfor
                        kernel_${kernel.id_}_f<${', '.join(f'decltype(loop_{vl.id_})' for vl in kernel.vertical_loops)}> kernel_${kernel.id_}{
                            ${', '.join(f'loop_{vl.id_}' for vl in kernel.vertical_loops)}
                        };
                        gpu_backend::launch_kernel<${max_extent},
                            i_block_size_t::value, j_block_size_t::value>(
                            domain[0], domain[1], domain[2],
                            kernel_${kernel.id_},
                            shared_alloc.size());
                    % endfor

                    GT_CUDA_CHECK(cudaDeviceSynchronize());
                };
            }
        }

        using ${name}_impl_::${name};
        """
    )

    @classmethod
    def apply(cls, root: LeafNode, **kwargs: Any) -> str:
        if not isinstance(root, cuir.Program):
            raise ValueError("apply() requires gtcpp.Progam root node")
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code