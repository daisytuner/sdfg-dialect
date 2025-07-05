#ifndef SDFG_SDFGLibraryNode
#define SDFG_SDFGLibraryNode

/// Defining the SDFG library call.
def SDFG_LibraryNodeOp : SDFG_Op<"library_node", [
    ParentOneOf<["BlockOp"]>,
]> {
  let summary     = "Library Node";
  let description = [{
    SDFG library node
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
    $code $operands `:` type($operands) `->` type($results) attr-dict
  }];

}

#endif // SDFG_SDFGLibraryNode
