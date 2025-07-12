#ifndef SDFG_RETURN_NODE
#define SDFG_RETURN_NODE

include "mlir/Interfaces/ControlFlowInterfaces.td"

// Terminator that ends an SDFG node region, returning optional values.
def SDFG_ReturnOp : SDFG_Op<"return", [Terminator, ReturnLike]> {
  let summary = "SDFG return terminator";
  let description = [{
    Terminator for an `sdfg.sdfg` operation body.  It returns zero or more
    SSA values whose types must match the result types of the parent
    `sdfg.sdfg` operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = [{
    attr-dict ( $operands^ `:` type($operands) )?
  }];
}

#endif // SDFG_RETURN_NODE 