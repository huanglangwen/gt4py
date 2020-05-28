import gt4py
from gt4py import gtscript

sd = gtscript.Field[float]


@gtscript.function
def absolute_value(in_array):
    abs_value = in_array if in_array > 0 else -in_array
    return abs_value

@gtscript.function
def is_smt5_mord5(bl, br):
    return bl * br < 0

@gtscript.function
def is_smt5_most_mords(bl, br, b0):
    return (3.0 * absolute_value(in_array=b0)) < absolute_value(in_array=(bl - br))

@gtscript.function
def get_bl(al, q):
    bl = al - q
    return bl

@gtscript.function
def get_br(al, q):
    br = al[0, 1, 0] - q
    return br

@gtscript.function
def get_b0(bl, br):
    b0 = bl + br
    return b0

@gtscript.function
def flux_intermediates(q, al, mord):
    bl = get_bl(al=al, q=q)
    br = get_br(al=al, q=q)
    b0 = get_b0(bl=bl, br=br)
    smt5 = is_smt5_mord5(bl, br) if mord == 5 else is_smt5_most_mords(bl, br, b0)
    tmp = smt5[0, -1, 0] + smt5 * (smt5[0, -1, 0] == 0)
    return bl, br, b0, tmp

@gtscript.function
def fx1_c_positive(c, br, b0):
    return (1.0 - c) * (br[0, -1, 0] - c * b0[0, -1, 0])


@gtscript.function
def fx1_c_negative(c, bl, b0):
    return (1.0 + c) * (bl + c * b0)

@gtscript.function
def final_flux(c, q, fx1, tmp):
    return q[0, -1, 0] + fx1 * tmp if c > 0.0 else q + fx1 * tmp

@gtscript.function
def fx1_fn(c, br, b0, bl):
    return fx1_c_positive(c, br, b0) if c > 0.0 else fx1_c_negative(c, bl, b0)

@gtscript.stencil(backend="mlir", rebuild=True)
def get_flux(q: sd, c: sd, al: sd, flux: sd, mord: int):
    with computation(PARALLEL), interval(0, None):
        bl, br, b0, tmp = flux_intermediates(q, al, mord)
        fx1 = fx1_fn(c, br, b0, bl)
        flux = final_flux(c, q, fx1, tmp)
