// ForgeIR StableHLO textual bridge
// graph_schema_version = 1.0
// graph_hash = ff1e5157e2783dcfe957062353c12af775b496948416b3e24d93b48df4654c56
module {
  func.func @main(%v0000: tensor<1x2xf32>, %v0001: tensor<2x2xf32>) -> tensor<1x2xf32> {
    // forgeir.op_id = op0000; forgeir.operation = Input
    // forgeir.op_id = op0001; forgeir.operation = Parameter
    // forgeir.op_id = op0002; forgeir.operation = Linear
    %forgeir0 = "stablehlo.transpose"(%v0001) {permutation = array<i64: 1, 0>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %v0002 = "stablehlo.dot_general"(%v0000, %forgeir0) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
    "func.return"(%v0002) : (tensor<1x2xf32>) -> ()
  }
}
