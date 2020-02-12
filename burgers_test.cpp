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
  var adv_phi_x;
  Do
  {
    adv_phi_x = u / (60. * dx) * (45. * (phi[i + 1] - phi[i - 1]) - 9. * (phi[i + 2] - phi[i - 2]) + (phi[i + 3] - phi[i - 3])) -
                abs_u / (60. * dx) * ((phi[i + 3] + phi[i - 3]) - 6. * (phi[i + 2] + phi[i - 2]) + 15. * (phi[i + 1] + phi[i - 1]) - 20. * phi);
    return adv_phi_x;
  }
};

stencil_function advection_y
{
  storage dy, v, abs_v, phi;
  var adv_phi_y;
  Do
  {
    adv_phi_y = v / (60. * dy) * (45. * (phi[j + 1] - phi[j - 1]) - 9. * (phi[j + 2] - phi[j - 2]) + (phi[j + 3] - phi[j - 3])) -
                abs_v / (60. * dy) * ((phi[j + 3] + phi[j - 3]) - 6. * (phi[j + 2] + phi[j - 2]) + 15. * (phi[j + 1] + phi[j - 1]) - 20. * phi);
    return adv_phi_y;
  }
};

stencil_function advection
{
  storage dx, dy, u, v;
  var cabs_u, abs_v, adv_u_x, adv_u_y, adv_u;
  Do
  {
    cabs_u = absolute_value(u);
    abs_v = absolute_value(v);

    adv_u_x = advection_x(dx, u, abs_u, u);
    adv_u_y = advection_y(dy, v, abs_v, u);
    adv_u = adv_u_x + adv_u_y;

    adv_v_x = advection_x(dx = dx, u = u, abs_u = abs_u, phi = v);
    adv_v_y = advection_y(dy = dy, v = v, abs_v = abs_v, phi = v);
    adv_v = adv_v_x + adv_v_y;

    return (adv_u, adv_v);
  }
};

stencil horizontal_diffusion_type2_stencil
{
  storage out, in, crlato, crlatu, hdmask;
  var lap;

  Do
  {
    vertical_region(k_start, k_end)
    {
      lap = laplacian(in, crlato, crlatu);
      const double delta_flux_x = diffusive_flux_x(lap, in) -
                                  diffusive_flux_x(lap(i - 1), in(i - 1));
      const double delta_flux_y = diffusive_flux_y(lap, in, crlato) -
                                  diffusive_flux_y(lap(j - 1), in(j - 1), crlato(j - 1));
      out = in - hdmask * (delta_flux_x + delta_flux_y);
    }
  }
};
