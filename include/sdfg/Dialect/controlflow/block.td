#ifndef SDFG_SDFGBlock
#define SDFG_SDFGBlock

def SDFG_BlockOp : SDFG_Op<"block", [
  ParentOneOf<["SDFGNode"]>,
  NoTerminator,
]> {
  let summary = "SDFG Block";
  let description = [{
    A control-flow node holding dataflow operations.
  }];

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
  }];

}

#endif // SDFG_SDFGBlock
