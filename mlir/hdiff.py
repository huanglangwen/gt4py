import gt4py
from gt4py import gtscript

backend = "mlir"
dtype = float

@gtscript.stencil(backend=backend, rebuild=True) #, dump_sir=True)
def hdiff(
    input: gtscript.Field[dtype], coeff: gtscript.Field[dtype], output: gtscript.Field[dtype]
):
    with computation(FORWARD), interval(...):
        lap = 4.0 * input - (
            input[-1, 0, 0] + input[1, 0, 0] + input[0, 1, 0] + input[0, -1, 0]
        )
        res = lap[1, 0, 0] - lap
        flx = 0.0 if res * (input[1, 0, 0] - input) > 0.0 else res
        res = lap[0, 1, 0] - lap[0, 0, 0]
        fly = 0.0 if res * (input[0, 1, 0] - input) > 0.0 else res
        output = input - coeff * (
            flx - flx[-1, 0, 0] + fly - fly[0, -1, 0]
        )
