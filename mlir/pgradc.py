import gt4py
from gt4py import gtscript

field = gtscript.Field[float]


@gtscript.stencil(backend="mlir", rebuild=True)
def PGradC(
    uc_in: field,
    vc_in: field,
    delpc: field,
    pkc: field,
    gz: field,
    rdxc: field,
    rdyc: field,
    *,
    hydrostatic: int,
    dt2: float,
):
    with computation(PARALLEL), interval(0, -1):
        # p_grad_c_ustencil
        wk = pkc[0, 0, 1] - pkc if hydrostatic else delpc
        uc_in = uc_in + dt2 * rdxc / (wk[-1, 0, 0] + wk) * (
            (gz[-1, 0, 1] - gz) * (pkc[0, 0, 1] - pkc[-1, 0, 0])
            + (gz[-1, 0, 0] - gz[0, 0, 1]) * (pkc[-1, 0, 1] - pkc)
        )
        # p_grad_c_vstencil
        vc_in = vc_in + dt2 * rdyc / (wk[0, -1, 0] + wk) * (
            (gz[0, -1, 1] - gz) * (pkc[0, 0, 1] - pkc[0, -1, 0])
            + (gz[0, -1, 0] - gz[0, 0, 1]) * (pkc[0, -1, 1] - pkc)
        )
