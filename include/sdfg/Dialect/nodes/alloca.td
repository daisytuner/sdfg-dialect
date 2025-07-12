#ifndef SDFG_SDFGAllocaNode
#define SDFG_SDFGAllocaNode

// A node that allocates storage for a tensor and can optionally initialise it
// with a constant value (e.g. lowered from `torch.operator "onnx.Constant"`).
// The op produces one result representing the allocated tensor.

def SDFG_AllocaOp : SDFG_Op<"alloca"> {
  let summary = "Tensor allocation";
  let description = [{
    `sdfg.alloca` allocates a tensor within an SDFG and optionally stores a
    compile-time constant into it via the `value` attribute.
  }];

  let arguments = (ins
    OptionalAttr<AnyAttr>:$value
  );

  let results = (outs
    AnyType:$result
  );

  // Print like:  `sdfg.alloca {value = dense<...>} : !tensor<...>`
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

#endif // SDFG_SDFGAllocaNode 