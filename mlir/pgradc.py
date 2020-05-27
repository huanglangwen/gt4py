import gt4py
from gt4py import gtscript

ftype = gtscript.Field[float]


@gtscript.stencil(backend="mlir", rebuild=True)
def PGradC(
    uc_in: ftype,
    vc_in: ftype,
    delpc: ftype,
    pkc: ftype,
    gz: ftype,
    rdxc: ftype,
    rdyc: ftype,
    *,
    hydrostatic: int,
    dt2: float,
):
    with computation(PARALLEL), interval(0, -1):
        # p_grad_c_ustencil
        wk = pkc[0, 0, 1] - pkc if hydrostatic else delpc
        uc_in = uc_in + dt2 * rdxc[0, 0, 0] / (wk[-1, 0, 0] + wk) * (
            (gz[-1, 0, 1] - gz[0, 0, 0]) * (pkc[0, 0, 1] - pkc[-1, 0, 0])
            + (gz[-1, 0, 0] - gz[0, 0, 1]) * (pkc[-1, 0, 1] - pkc[0, 0, 0])
        )
        # p_grad_c_vstencil
        vc_in = vc_in + dt2 * rdyc[0, 0, 0] / (wk[0, -1, 0] + wk) * (
            (gz[0, -1, 1] - gz) * (pkc[0, 0, 1] - pkc[0, -1, 0])
            + (gz[0, -1, 0] - gz[0, 0, 1]) * (pkc[0, -1, 1] - pkc)
        )
