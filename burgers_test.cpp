#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil_function absolute_value
{
  storage phi;
  double Do() { return phi * (phi >= 0.) - phi * (phi < 0.); }
};

stencil_function advection_x
{
  storage dx, u, abs_u, phi;
  Do
  {
    return u / (60. * dx) * (45. * (phi[i + 1] - phi[i - 1]) - 9. * (phi[i + 2] - phi[i - 2]) + (phi[i + 3] - phi[i - 3])) -
                abs_u / (60. * dx) * ((phi[i + 3] + phi[i - 3]) - 6. * (phi[i + 2] + phi[i - 2]) + 15. * (phi[i + 1] + phi[i - 1]) - 20. * phi);
  }
};

stencil_function advection_y
{
  storage dy, v, abs_v, phi;
  Do
  {
    return v / (60. * dy) * (45. * (phi[j + 1] - phi[j - 1]) - 9. * (phi[j + 2] - phi[j - 2]) + (phi[j + 3] - phi[j - 3])) -
                abs_v / (60. * dy) * ((phi[j + 3] + phi[j - 3]) - 6. * (phi[j + 2] + phi[j - 2]) + 15. * (phi[j + 1] + phi[j - 1]) - 20. * phi);
  }
};

stencil_function advection
{
  storage dx, dy, u, v;
  storage abs_u, abs_v, adv_u_x, adv_u_y, adv_u, adv_v_x, adv_v_y, adv_v;
  Do
  {
    abs_u = absolute_value(u);
    abs_v = absolute_value(v);

    adv_u_x = advection_x(dx, u, abs_u, u);
    adv_u_y = advection_y(dy, v, abs_v, u);
    adv_u = adv_u_x + adv_u_y;

    adv_v_x = advection_x(dx, u, abs_u, v);
    adv_v_y = advection_y(dy, v, abs_v, v);
    adv_v = adv_v_x + adv_v_y;

    return (adv_u, adv_v);
  }
};

stencil_function diffusion_x
{
  storage dx, phi;
  Do
  {
    return (-phi[i-2]
                       + 16. * phi[i-1]
                       - 30. * phi
                       + 16. * phi[i+1]
                       - phi[i+2]
               ) / (12. * dx * dx);
  }
};

stencil_function diffusion_x
{
  storage dx, phi;
  Do
  {
    return (-phi[i-2]
                       + 16. * phi[i-1]
                       - 30. * phi
                       + 16. * phi[i+1]
                       - phi[i+2]
               ) / (12. * dx * dx);
  }
};

stencil_function diffusion_y
{
  storage dy, phi;
  Do
  {
    return (
                       -       phi[j-2]
                       + 16. * phi[j-1]
                       - 30. * phi
                       + 16. * phi[j+1]
                       - phi[j+2]
               ) / (12. * dy * dy);
  }
};

stencil_function diffusion
{
  storage dx, dy, u, v;
  storage diff_u_x, diff_u_y, diff_u, diff_v_x, diff_v_y, diff_v;
  Do
  {
    diff_u_x = diffusion_x(dx, u);
    diff_u_y = diffusion_y(dy, u);
    diff_u = diff_u_x + diff_u_y;

    diff_v_x = diffusion_x(dx, v);
    diff_v_y = diffusion_y(dy, v);
    diff_v = diff_v_x + diff_v_y;

    return (diff_u, diff_v);
  }
};

stencil rk_stage
{
  storage in_u_now, in_v_now, in_u_tmp, in_v_tmp, out_u, out_v;
  var dt, dx, dy, mu;
  vertical_region(k_start, k_end)
  {
    (adv_u, adv_v) = advection(dx, dy, in_u_tmp, in_v_tmp);
    (diff_u, diff_v) = diffusion(dx, dy, in_u_tmp, in_v_tmp);
    out_u = in_u_now + dt * (-adv_u + mu * diff_u);
    out_v = in_v_now + dt * (-adv_v + mu * diff_v);
  }
};

stencil copy
{
  storage in_phi, out_phi;
  vertical_region(k_start, k_end)
  {
    out_phi = in_phi;
  }
};
