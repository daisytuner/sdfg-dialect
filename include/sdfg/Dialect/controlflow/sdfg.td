#ifndef SDFG_SDFGNode
#define SDFG_SDFGNode

/// Defining the SDFG
def SDFG_SDFGNode : SDFG_Op<"sdfg", [
    ParentOneOf<["mlir::ModuleOp"]>,
    NoTerminator,
    IsolatedFromAbove,
    SymbolTable,
    Symbol,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
    let summary = "SDFG";
    let description = [{
        A stateful dataflow multigraph region.
    }];
    let hasCustomAssemblyFormat = 1;

    let regions = (region SizedRegion<1>:$body);

    let arguments = (ins 
        SymbolNameAttr:$sym_name,
        DefaultValuedAttr<I32Attr, "0">:$num_args
    );

    let assemblyFormat = [{
        $sym_name attr-dict-with-keyword $body
    }];

    let extraClassDeclaration = [{
        static SDFGNode create(PatternRewriter &rewriter, Location location, unsigned num_args, TypeRange args);
        static SDFGNode create(PatternRewriter &rewriter, Location location);
        Block::BlockArgListType getArgs();
        TypeRange getArgTypes();
    }];
}

#endif // SDFG_SDFGNode
