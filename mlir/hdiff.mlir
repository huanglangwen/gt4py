module {
  func @hdiff(%input_fd : !stencil.field<?x?x?xf64>, %coeff_fd : !stencil.field<?x?x?xf64>, %output_fd : !stencil.field<?x?x?xf64>) attributes { stencil.program } {
    stencil.assert %input_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %coeff_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<?x?x?xf64>
    stencil.assert %output_fd ([-4, -4, -4]:[68, 68, 68]) : !stencil.field<?x?x?xf64>

    %input = stencil.load %input_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    %coeff = stencil.load %coeff_fd : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>

    %lap = stencil.apply (%arg0 = %input : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst0 = constant -4.000000e+00 : f64
      %acc1 = stencil.access %arg0[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %acc3 = stencil.access %arg0[1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %acc2 = stencil.access %arg0[-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %acc4 = stencil.access %arg0[0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %exp1 = addf %acc2, %acc3 : f64
      %acc5 = stencil.access %arg0[0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %exp2 = addf %exp1, %acc4 : f64
      %exp3 = addf %exp2, %acc5 : f64
      %exp0 = mulf %acc1, %cst0 : f64
      %exp4 = addf %exp0, %exp3 : f64
      stencil.return %exp4 : f64
    }
    %dx = stencil.apply (%arg1 = %lap : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %acc2 = stencil.access %arg1[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %acc1 = stencil.access %arg1[1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %exp0 = subf %acc1, %acc2 : f64
      stencil.return %exp0 : f64
    }
    %flx = stencil.apply (%arg2 = %dx : !stencil.temp<?x?x?xf64>, %arg3 = %input : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %acc3 = stencil.access %arg3[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %acc2 = stencil.access %arg3[1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %exp0 = subf %acc2, %acc3 : f64
      %acc1 = stencil.access %arg2[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %cst0 = constant 0.000000e+00 : f64
      %exp1 = mulf %acc1, %exp0 : f64
      %exp2 = cmpf "ogt", %exp1, %cst0 : f64
      %sel0 = select %exp2, %cst0, %acc1 : f64
      stencil.return %sel0 : f64
    }
    %dy = stencil.apply (%arg4 = %lap : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %acc2 = stencil.access %arg4[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %acc1 = stencil.access %arg4[0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %exp0 = subf %acc1, %acc2 : f64
      stencil.return %exp0 : f64
    }
    %fly = stencil.apply (%arg5 = %dy : !stencil.temp<?x?x?xf64>, %arg6 = %input : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %acc3 = stencil.access %arg6[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %acc2 = stencil.access %arg6[0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %exp0 = subf %acc2, %acc3 : f64
      %acc1 = stencil.access %arg5[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %cst0 = constant 0.000000e+00 : f64
      %exp1 = mulf %acc1, %exp0 : f64
      %exp2 = cmpf "ogt", %exp1, %cst0 : f64
      %sel0 = select %exp2, %cst0, %acc1 : f64
      stencil.return %sel0 : f64
    }
    %output = stencil.apply (%arg7 = %input : !stencil.temp<?x?x?xf64>, %arg8 = %coeff : !stencil.temp<?x?x?xf64>, %arg9 = %flx : !stencil.temp<?x?x?xf64>, %arg10 = %fly : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %acc4 = stencil.access %arg9[-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %acc3 = stencil.access %arg9[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %acc5 = stencil.access %arg10[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %exp0 = subf %acc3, %acc4 : f64
      %acc6 = stencil.access %arg10[0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %exp1 = addf %exp0, %acc5 : f64
      %exp2 = subf %exp1, %acc6 : f64
      %acc2 = stencil.access %arg8[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %exp3 = mulf %acc2, %exp2 : f64
      %acc1 = stencil.access %arg7[0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %exp4 = subf %acc1, %exp3 : f64
      stencil.return %exp4 : f64
    }
    stencil.store %output to %output_fd([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return
 }
}
