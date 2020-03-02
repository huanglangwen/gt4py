import gt4py
from gt4py import gtscript

backend = "dawn:gtmc"
dtype = float


@gtscript.stencil(backend=backend)
def hdiff(input: gtscript.Field[dtype], coeff: gtscript.Field[dtype], output: gtscript.Field[dtype]):
    with computation(FORWARD), interval(...):
        # lap = 4.0 * input[0, 0, 0] - (input[-1, 0, 0] + input[1, 0, 0] + input[0, 1, 0] + input[0, -1, 0])
        a0 = input[0, 0, 0]
        a1 = input[-1, 0, 0]
        a2 = input[1, 0, 0]
        a3 = input[0, 1, 0]
        a4 = input[0, -1, 0]
        e0 = a1 + a2
        e1 = a3 + a4
        e2 = e0 + e1
        c0 = 4.0
        e3 = c0 * a0
        e4 = e3 - e2
        lap[0, 0, 0] = e4      # return lap
        # res = lap[1, 0, 0] - lap[0, 0, 0]
        # flx = 0 if (res * (input[1, 0, 0] - input[0, 0, 0])) > 0 else res
        a0 = lap[1, 0, 0]
        a1 = lap[0, 0, 0]
        e0 = a0 - a1
        a2 = input[1, 0, 0]
        a3 = input[0, 0, 0]
        e1 = a2 - a3
        e2 = e0 * e1
        c0 = 0.0
        e3 = e2 > c0
        s0 = c0 if e3 else e0
        flx[0, 0, 0] = s0
        # res = lap[0, 1, 0] - lap[0, 0, 0]
        # fly = 0 if (res * (input[0, 1, 0] - input[0, 0, 0])) > 0 else res
        a0 = lap[0, 1, 0]
        a1 = lap[0, 0, 0]
        e0 = a0 - a1
        a2 = input[0, 1, 0]
        a3 = input[0, 0, 0]
        e1 = a2 - a3
        e2 = e0 * e1
        c0 = 0.0
        e3 = e2 > c0
        s0 = c0 if e3 else e0
        fly[0, 0, 0] = s0
        # output = input[0, 0, 0] - coeff[0, 0, 0] * (flx[0, 0, 0] - flx[-1, 0, 0] + fly[0, 0, 0] - fly[0, -1, 0])
        a0 = flx[0, 0, 0]
        a1 = flx[-1, 0, 0]
        e0 = a0 - a1
        a2 = fly[0, 0, 0]
        a3 = fly[0, -1, 0]
        e1 = a2 - a3
        e2 = e0 + e1
        a4 = coeff[0, 0, 0]
        e3 = a4 * e2
        a5 = input[0, 0, 0]
        e4 = a5 - e3
        output[0, 0, 0] = e4

