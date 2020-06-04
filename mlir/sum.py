from gt4py import gtscript
field_type = gtscript.Field[float]

@gtscript.stencil(backend="mlir")
def sum(inp: field_type, out: field_type):
    with computation(PARALLEL), interval(...):
        out = inp[1, 0, 0] + inp[-1, 0, 0]
