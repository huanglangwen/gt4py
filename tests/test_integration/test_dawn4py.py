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
import os
import pytest

import numpy as np

from gt4py import gtscript
from gt4py import backend as gt_backend
from gt4py import storage as gt_storage
from gt4py import config as gt_config
from gt4py import testing as gt_testing

#from ..definitions import DAWN_BACKENDS #ALL_BACKENDS, CPU_BACKENDS, GPU_BACKENDS, INTERNAL_BACKENDS
DAWN_BACKENDS = ['dawn:gtmc']

@pytest.mark.slow
def test_copy_stencil_dawn4py():
    py_version = "{major}{minor}".format(
        major=sys.version_info.major, minor=sys.version_info.minor
    )
    cpython_id = "py{version}_{api}".format(
        version=py_version, api=sys.api_version
    )
    base_path = os.path.join(
        gt_config.cache_settings["root_path"], gt_config.cache_settings["dir_name"], cpython_id
    )

    stencil_name = Dawn4PyCopy.name()
    stencil_def = Dawn4PyCopy.definition

    for backend in DAWN_BACKENDS:
        implementation = gtscript.stencil(
            backend=backend,
            definition=stencil_def,
            name=stencil_name,
            rebuild=True,
            externals=Dawn4PyCopy.externals()
        )

        fs_backend = backend.replace(':', '')
        source_path = __file__.replace(os.getcwd() + '/', '').replace('.py', '')
        stencil_path = os.path.join(base_path, fs_backend, source_path)
        stencil_stub = f"m_{stencil_name}__{fs_backend}_{implementation._gt_id_}"

        # Assert file existence...
        pybind_file_path= os.path.join(stencil_path, f"{stencil_stub}.py")
        assert os.path.exists(pybind_file_path)

        shared_object_path = os.path.join(stencil_path, f"{stencil_stub}_pyext.cpython-{py_version}m-x86_64-linux-gnu.so")
        assert os.path.exists(shared_object_path)

        gen_dir = os.path.join(stencil_path, stencil_stub + "_pyext_BUILD")
        gen_files = ['bindings.cpp', 'computation.hpp', 'computation.cpp', f"_dawn_{stencil_name}.hpp"]
        for gen_file in gen_files:
            gen_path = os.path.join(gen_dir, gen_file)
            assert os.path.exists(gen_path)

        # Verify the data
        Dawn4PyCopy.validate(implementation)


# ---- Copy stencil ----
class Dawn4PyCopy(object):
    # Define stencil
    def definition(field_a: gtscript.Field[float], field_b: gtscript.Field[float]):
        with computation(PARALLEL), interval(...):
            field_b = field_a

    @classmethod
    def name(cls):
        return 'copy_stencil'

    @classmethod
    def externals(cls):
        return {}

    @classmethod
    def storages(cls):
        shape = (10, 10, 1)
        nhalo = 3
        storage_backend = DAWN_BACKENDS[0]
        field_a = gt_storage.ones(
            shape=shape, default_origin=(nhalo, nhalo, 0), dtype=float, backend=storage_backend
        )
        field_b = gt_storage.zeros(
            shape=shape, default_origin=(nhalo, nhalo, 0), dtype=float, backend=storage_backend
        )
        return field_a, field_b

    @classmethod
    def validate(cls, implementation):
        field_a, field_b = cls.storages()
        implementation(field_a, field_b)
        assert (field_b[3, 3] == 1.0)
