module {
func @hdiff(%input_fd : !stencil.field<ijk,f64>,
            %coeff_fd : !stencil.field<ijk,f64>,
            %output_fd : !stencil.field<ijk,f64>)
  attributes { stencil.program } {
	stencil.assert %input_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %coeff_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>
  stencil.assert %output_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<ijk,f64>

  %input = stencil.load %input_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  %coeff = stencil.load %coeff_fd : (!stencil.field<ijk,f64>) -> !stencil.view<ijk,f64>
  
  // lap = 4.0 * input[0, 0, 0] - (input[-1, 0, 0] + input[1, 0, 0] + input[0, 1, 0] + input[0, -1, 0])
  %lap = stencil.apply %arg1 = %input : !stencil.view<ijk,f64> {
      %c0 = constant 4.0 : f64
      %a0 = stencil.access %arg1[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64      
      %e0 = mulf %c0, %a0 : f64
      %a1 = stencil.access %arg1[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %a2 = stencil.access %arg1[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %a3 = stencil.access %arg1[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %a4 = stencil.access %arg1[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %e1 = addf %a1, %a2 : f64
      %e2 = addf %a3, %a4 : f64
      %e3 = addf %e1, %e2 : f64            
      %e4 = subf %e0, %e3 : f64
      stencil.return %e4 : f64
	} : !stencil.view<ijk,f64>
  // flx = 0.0 if ((lap[1,0,0] - lap[0,0,0]) * (input[1,0,0] - input[0,0,0])) > 0.0 else (lap[1,0,0] - lap[0,0,0])

  // flx
  %flx = stencil.apply %arg2 = %input, %arg3 = %lap : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {
      %a0 = stencil.access %arg3[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %a1 = stencil.access %arg3[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %e0 = subf %a0, %a1 : f64
      %a2 = stencil.access %arg2[1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %a3 = stencil.access %arg2[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %e1 = subf %a2, %a3 : f64
      %e2 = mulf %e0, %e1 : f64
      %c0 = constant 0.0 : f64
      %e3 = cmpf "ogt", %e2, %c0 : f64
      %s0 = select %e3, %c0, %e0 : f64
      stencil.return %s0 : f64
	} : !stencil.view<ijk,f64>
  // fly = 0 if ((lap[0,1,0] - lap[0,0,0]) * (input[0,1,0] - input[0,0,0])) > 0 else (lap[0,1,0] - lap[0,0,0])
  %fly = stencil.apply %arg4 = %input, %arg5 = %lap : !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {      
      %a0 = stencil.access %arg5[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %a1 = stencil.access %arg5[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %e0 = subf %a0, %a1 : f64
      %a2 = stencil.access %arg4[0, 1, 0] : (!stencil.view<ijk,f64>) -> f64
      %a3 = stencil.access %arg4[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %e1 = subf %a2, %a3 : f64
      %e2 = mulf %e0, %e1 : f64
      %c0 = constant 0.0 : f64
      %e3 = cmpf "ogt", %e2, %c0 : f64
      %s0 = select %e3, %c0, %e0 : f64
      stencil.return %s0 : f64
	} : !stencil.view<ijk,f64>
  // output = input[0, 0, 0] - coeff[0, 0, 0] * (flx[0, 0, 0] - flx[-1, 0, 0] + fly[0, 0, 0] - fly[0, -1, 0])
  %output = stencil.apply %arg6 = %input, %arg7 = %flx, %arg8 = %fly, %arg9 = %coeff : !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64>, !stencil.view<ijk,f64> {  
      %a0 = stencil.access %arg7[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %a1 = stencil.access %arg7[-1, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %e0 = subf %a0, %a1 : f64
      %a2 = stencil.access %arg8[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %a3 = stencil.access %arg8[0, -1, 0] : (!stencil.view<ijk,f64>) -> f64
      %e1 = subf %a2, %a3 : f64
      %e2 = addf %e0, %e1 : f64
      %a4 = stencil.access %arg9[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %e3 = mulf %a4, %e2 : f64
      %a5 = stencil.access %arg6[0, 0, 0] : (!stencil.view<ijk,f64>) -> f64
      %e4 = subf %a5, %e3 : f64      
      stencil.return %e4 : f64
	} : !stencil.view<ijk,f64>
	stencil.store %output to %output_fd ([0, 0, 0]:[64, 64, 64]) : !stencil.view<ijk,f64> to !stencil.field<ijk,f64>
  return
 }
}
