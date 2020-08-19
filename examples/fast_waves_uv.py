from gt4py import gtscript
from gt4py import definitions as gt_defs

sd = gtscript.Field[float]
backend_t = "mlir"
#backend_t = "dawn:cxxopt"


# , splitters={gtscript.I: ['i0', 'ie'], gtscript.J: ['j0', 'je']})
@gtscript.stencil(backend=backend_t)
def fast_waves_ut(
    uin: sd,
    utens: sd,
    vin: sd,
    vtens: sd,
    upos: sd,
    vpos: sd,
    wgtfac: sd,
    ppuv: sd,
    hhl: sd,
    rho: sd,
    rho0: sd,
    p0: sd,
    uout: sd,
    vout: sd,
    fx: sd,         # !stencil.field<0x?x0xf64>
    xlhsx: sd,      # !stencil.field<?x?x0xf64>
    xlhsy: sd,      # !stencil.field<?x?x0xf64>
    xdzdx: sd,      # !stencil.field<?x?x0xf64>
    xdzdy: sd,      # !stencil.field<?x?x0xf64>
    cwp: sd,        # !stencil.field<?x?x0xf64>
    wbbctens: sd,   # !stencil.field<?x?x0xf64>,
    dt: float,      # f64
    edadlat: float, # f64
):
    from __splitters__ import i0, ie, j0, je

    with computation(PARALLEL), interval(...):
        ppgk = ((wgtfac * ppuv) + (ppuv[0, 0, -1] * (1.0 - wgtfac)))
        ppgc = ppgk[0, 0, 1] - ppgk

        ppgu_terrain = (ppuv[1, 0, 0] - ppuv) + ((5.000000e-01 * (ppgc[1, 0, 0] + ppgc)) * (((hhl[0, 0, 1] + hhl) - (hhl[1, 0, 1] + hhl[1, 0, 0])) / ((hhl[0, 0, 1] - hhl) + (hhl[1, 0, 1] - hhl[1, 0, 0]))))
        ppgv_terrain = (ppuv[0, 1, 0] - ppuv) + ((5.000000e-01 * (ppgc[0, 1, 0] + ppgc)) * (((hhl[0, 0, 1] + hhl) - (hhl[0, 1, 1] + hhl[0, 1, 0])) / ((hhl[0, 0, 1] - hhl) + (hhl[0, 1, 1] - hhl[0, 1, 0]))))

        ppgu_free = ppuv[1, 0, 0] - ppuv
        ppgv_free = ppuv[0, 1, 0] - ppuv

        xrhsx =  (((-fx) / ((rho[1, 0, 0] + rho) * 0.500000e+00)) * (ppuv[1, 0, 0] - ppuv)) + utens
        xrhsy = (((-edadlat) / ((rho[0, 1, 0] + rho) * 0.5)) * (ppuv[0, 1, 0] - ppuv)) + vtens
        xrhsz = (((rho0 / rho) * 9.80665) * (1.0 - ((p0 + ppuv) * cwp))) + wbbctens

    with computation(PARALLEL):
        with interval(0, 15):
            uout = (dt * (utens - (ppgu_free * ((2.0 * fx) / (rho[1, 0, 0] + rho))))) + uin

        with interval(15, 59):
            uout = (dt * (utens - (ppgu_terrain * ((2.0 * fx) / (rho[1, 0, 0] + rho))))) + uin

        with interval(59, 60):
            uout = upos + (dt * (((((0.5 * (xrhsz[1, 0, 0] + xrhsz)) - (xdzdx * xrhsx)) -
                                ((0.5 * ((0.5 * (xdzdy[1, -1, 0] + xdzdy[1, 0, 0])) +
                                    (0.5 * (xdzdy[0, -1, 0] + xdzdy)))) * (0.5 *
                                ((0.5 * (xrhsy[1, -1, 0] + xrhsy[1, 0, 0])) +
                                    (0.5 * (xrhsy[0, -1, 0] + xrhsy)))))) * (xlhsx * xdzdx)) + xrhsx))

