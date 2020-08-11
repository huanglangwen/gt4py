from gt4py import gtscript
from gt4py import definitions as gt_defs

sd = gtscript.Field[float]
backend_t = "numpy"
#backend_t = "mlir"
#backend_t = "dawn:cxxopt"


# , splitters={gtscript.I: ['i0', 'ie'], gtscript.J: ['j0', 'je']})
@gtscript.stencil(backend=backend_t)
def divergence_corner(
    u: sd,
    v: sd,
    ua: sd,
    va: sd,
    dxc: sd,
    dyc: sd,
    cos_sg1: sd,
    cos_sg2: sd,
    cos_sg3: sd,
    cos_sg4: sd,
    sin_sg1: sd,
    sin_sg2: sd,
    sin_sg3: sd,
    sin_sg4: sd,
    rarea_c: sd,
    divg_d: sd,
):
    from __splitters__ import i0, ie, j0, je

    with computation(FORWARD), interval(...):
        uf = (
            (u - 0.25 * (va[0, -1, 0] + va) * (cos_sg4[0, -1, 0] + cos_sg2))
            * dyc
            * 0.5
            * (sin_sg4[0, -1, 0] + sin_sg2)
        )

        with parallel(region[i0 : i0 + 1, :], region[ie - 1 : ie, :]):
            uf = u * dyc * 0.5 * (sin_sg4[0, -1, 0] + sin_sg2)

        vf = (
            (v - 0.25 * (ua[-1, 0, 0] + ua) * (cos_sg3[-1, 0, 0] + cos_sg1))
            * dxc
            * 0.5
            * (sin_sg3[-1, 0, 0] + sin_sg1)
        )

        with parallel(region[:, j0 : j0 + 1], region[:, je - 1 : je]):
            vf = v * dxc * 0.5 * (sin_sg3[-1, 0, 0] + sin_sg1)

        divg_d = rarea_c * (vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf)

        with parallel(region[i0 : i0 + 1, j0 : j0 + 1], region[ie - 1 : ie, j0 : j0 + 1]):
            divg_d = rarea_c * (-vf + uf[-1, 0, 0] - uf)

        with parallel(region[i0 : i0 + 1, je - 1 : je], region[ie - 1 : ie, je - 1 : je]):
            divg_d = rarea_c * (vf[0, -1, 0] + uf[-1, 0, 0] - uf)
