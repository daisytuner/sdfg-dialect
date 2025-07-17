#ifndef SDFG_SDFGLibraryNode
#define SDFG_SDFGLibraryNode

/// Defining the SDFG library call.
def SDFG_LibraryNodeOp : SDFG_Op<"library_node"> {
  let summary     = "Library Node";
  let description = [{
    SDFG library node representing an external library call.
    The node supports:
    - operands: Input operands to the library function
    - result: The single value produced by the library function
    - attributes: Additional parameters specific to the library function (e.g., kernel_shape, pads, strides)
  }];

  let arguments = (ins
    StrAttr:$code,
    Variadic<AnyType>:$operands
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    $code ` ` $operands `:` type($operands) `->` type($results) attr-dict
  }];
}

#endif // SDFG_SDFGLibraryNode
