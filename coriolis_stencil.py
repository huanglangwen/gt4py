import gt4py
from gt4py import gtscript

backend="dawn:gtmc"
dtype = float

@gtscript.stencil(backend=backend)
def coriolis_stencil(
    u_nnow: gtscript.Field[dtype],
    v_nnow: gtscript.Field[dtype],
    fc: gtscript.Field[dtype],
    u_tens: gtscript.Field[dtype],
    v_tens: gtscript.Field[dtype]
):
    with computation(FORWARD), interval(...):
        z_fv_north = fc * (v_nnow + v_nnow[1, 0, 0])
        z_fv_south = fc[0, -1, 0] * (v_nnow[0, -1, 0] + v_nnow[1, -1, 0])
        u_tens += 0.25 * (z_fv_north + z_fv_south)
        z_fu_east = fc * (u_nnow + u_nnow[0, 1, 0])
        z_fu_west = fc[-1, 0, 0] * (u_nnow[-1, 0, 0] + u_nnow[-1, 1, 0])
        v_tens -= 0.25 * (z_fu_east + z_fu_west)
