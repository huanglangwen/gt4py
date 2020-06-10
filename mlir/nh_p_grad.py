import gt4py
from gt4py import gtscript

sd = gtscript.Field[float]


@gtscript.stencil(backend="mlir", rebuild=True)
def NH_P_Grad(
    u: sd,
    v: sd,
    du: sd,
    dv: sd,
    rdx: sd,
    rdy: sd,
    wk: sd,
    wk1: sd,
    gz: sd,
    pk3: sd,
    pp: sd,
    dt: float,
):
    # CalcWk
    with computation(PARALLEL), interval(...):
        wk = pk3[0, 0, 1] - pk3

    # CalcU
    with computation(PARALLEL), interval(...):
        # hydrostatic contribution
        du = (
            dt
            / (wk + wk[1, 0, 0])
            * (
                (gz[0, 0, 1] - gz[1, 0, 0]) * (pk3[1, 0, 1] - pk3)
                + (gz - gz[1, 0, 1]) * (pk3[0, 0, 1] - pk3[1, 0, 0])
            )
        )
        # nonhydrostatic contribution
        u = (
            u
            + du
            + dt
            / (wk1 + wk1[1, 0, 0])
            * (
                (gz[0, 0, 1] - gz[1, 0, 0]) * (pp[1, 0, 1] - pp)
                + (gz - gz[1, 0, 1]) * (pp[0, 0, 1] - pp[1, 0, 0])
            )
        ) * rdx

    # CalcV
    with computation(PARALLEL), interval(...):
        # hydrostatic contribution
        dv = (
            dt
            / (wk + wk[0, 1, 0])
            * (
                (gz[0, 0, 1] - gz[0, 1, 0]) * (pk3[0, 1, 1] - pk3)
                + (gz - gz[0, 1, 1]) * (pk3[0, 0, 1] - pk3[0, 1, 0])
            )
        )
        # nonhydrostatic contribution
        v = (
            v
            + dv
            + dt
            / (wk1 + wk1[0, 1, 0])
            * (
                (gz[0, 0, 1] - gz[0, 1, 0]) * (pp[0, 1, 1] - pp)
                + (gz - gz[0, 1, 1]) * (pp[0, 0, 1] - pp[0, 1, 0])
            )
        ) * rdy
