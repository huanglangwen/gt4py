import gt4py
from gt4py import gtscript

backend="dawn:gtmc"
dtype = float
rebuild = False

@gtscript.function
def absolute_value(phi):
    abs_phi = phi[0, 0, 0] * (phi[0, 0, 0] >= 0.) - phi[0, 0, 0] * (phi[0, 0, 0] < 0.)
    return abs_phi


@gtscript.function
def advection_x(dx, u, abs_u, phi):
    adv_phi_x = u[0, 0, 0] / (60. * dx) * (
            + 45. * (phi[1, 0, 0] - phi[-1, 0, 0])
            - 9. * (phi[2, 0, 0] - phi[-2, 0, 0])
            + (phi[3, 0, 0] - phi[-3, 0, 0])
    ) - abs_u[0, 0, 0] / (60. * dx) * ((phi[3, 0, 0] + phi[-3, 0, 0])
                        - 6. * (phi[2, 0, 0] + phi[-2, 0, 0])
                        + 15. * (phi[1, 0, 0] + phi[-1, 0, 0])
                        - 20. * phi[0, 0, 0])
    return adv_phi_x


@gtscript.function
def advection_y(dy, v, abs_v, phi):
    adv_phi_y = v[0, 0, 0] / (60. * dy) * (
            + 45. * (phi[0, 1, 0] - phi[0, -1, 0])
            - 9. * (phi[0, 2, 0] - phi[0, -2, 0])
            + (phi[0, 3, 0] - phi[0, -3, 0])
    ) - abs_v[0, 0, 0] / (60. * dy) * (
                        +       (phi[0, 3, 0] + phi[0, -3, 0])
                        - 6. * (phi[0, 2, 0] + phi[0, -2, 0])
                        + 15. * (phi[0, 1, 0] + phi[0, -1, 0])
                        - 20. * phi[0, 0, 0]
                )
    return adv_phi_y


@gtscript.function
def advection(dx, dy, u, v):
    abs_u = absolute_value(phi=u)
    abs_v = absolute_value(phi=v)

    adv_u_x = advection_x(dx=dx, u=u, abs_u=abs_u, phi=u)
    adv_u_y = advection_y(dy=dy, v=v, abs_v=abs_v, phi=u)
    adv_u = adv_u_x[0, 0, 0] + adv_u_y[0, 0, 0]

    adv_v_x = advection_x(dx=dx, u=u, abs_u=abs_u, phi=v)
    adv_v_y = advection_y(dy=dy, v=v, abs_v=abs_v, phi=v)
    adv_v = adv_v_x[0, 0, 0] + adv_v_y[0, 0, 0]

    return adv_u, adv_v


@gtscript.function
def diffusion_x(dx, phi):
    diff_phi = (-phi[-2, 0, 0]
                       + 16. * phi[-1, 0, 0]
                       - 30. * phi[0, 0, 0]
                       + 16. * phi[1, 0, 0]
                       - phi[2, 0, 0]
               ) / (12. * dx ** 2)
    return diff_phi


@gtscript.function
def diffusion_y(dy, phi):
    diff_phi = (
                       -       phi[0, -2, 0]
                       + 16. * phi[0, -1, 0]
                       - 30. * phi[0, 0, 0]
                       + 16. * phi[0, 1, 0]
                       - phi[0, 2, 0]
               ) / (12. * dy ** 2)
    return diff_phi


@gtscript.function
def diffusion(dx, dy, u, v):
    diff_u_x = diffusion_x(dx=dx, phi=u)
    diff_u_y = diffusion_y(dy=dy, phi=u)
    diff_u = diff_u_x[0, 0, 0] + diff_u_y[0, 0, 0]

    diff_v_x = diffusion_x(dx=dx, phi=v)
    diff_v_y = diffusion_y(dy=dy, phi=v)
    diff_v = diff_v_x[0, 0, 0] + diff_v_y[0, 0, 0]

    return diff_u, diff_v


#@gtscript.stencil(backend=backend, externals=externals, rebuild=rebuild, **backend_opts)
@gtscript.stencil(backend=backend, rebuild=rebuild)
def rk_stage(
        in_u_now: gtscript.Field[dtype],
        in_v_now: gtscript.Field[dtype],
        in_u_tmp: gtscript.Field[dtype],
        in_v_tmp: gtscript.Field[dtype],
        out_u: gtscript.Field[dtype],
        out_v: gtscript.Field[dtype],
        *,
        dt: float,
        dx: float,
        dy: float,
        mu: float
):
    with computation(PARALLEL), interval(...):
        adv_u, adv_v = advection(dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp)
        diff_u, diff_v = diffusion(dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp)
        out_u = in_u_now[0, 0, 0] + dt * (- adv_u[0, 0, 0] + mu * diff_u[0, 0, 0])
        out_v = in_v_now[0, 0, 0] + dt * (- adv_v[0, 0, 0] + mu * diff_v[0, 0, 0])


@gtscript.stencil(backend=backend)
def copy(in_phi: gtscript.Field[dtype], out_phi: gtscript.Field[dtype]):
    with computation(PARALLEL), interval(...):
        out_phi = in_phi[0, 0, 0]
