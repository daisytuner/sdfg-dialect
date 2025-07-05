#ifndef SDFG_SDFGLibraryNode
#define SDFG_SDFGLibraryNode

/// Defining the SDFG library call.
def SDFG_LibraryNodeOp : SDFG_Op<"library_node", [AttrSizedOperandSegments]> {
  let summary     = "Library Node";
  let description = [{
    SDFG library node
  }];

  let arguments = (ins
    StrAttr:$code,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$output_operands
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    $code `ins` ` ` $inputs `:` type($inputs) `outs` $output_operands `:` type($output_operands) `->` type($results) attr-dict
  }];
}

#endif // SDFG_SDFGLibraryNode
