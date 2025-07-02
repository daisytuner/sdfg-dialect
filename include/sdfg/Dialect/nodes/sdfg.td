// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Table-driven file for SDFG nodes.

#ifndef SDFG_SDFGNode
#define SDFG_SDFGNode

/// Defining the SDFG
def SDFG_SDFGNode : SDFG_Op<"sdfg", [
    ParentOneOf<["mlir::ModuleOp"]>,
    NoTerminator,
    IsolatedFromAbove,
    SymbolTable,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
    let summary = "SDFG";
    let description = [{
        Describes an SDFG.
        For example:

        ```mlir
        sdfg.sdfg{} {
            ...
        }
        ```
    }];

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        attr-dict-with-keyword $body
    }];

    let arguments = (ins 
        I32Attr:$ID,
        I32Attr:$num_args
    );

    let extraClassDeclaration = [{
        static SDFGNode create(PatternRewriter &rewriter, Location location, unsigned num_args, TypeRange args);
        static SDFGNode create(PatternRewriter &rewriter, Location location);
        Block::BlockArgListType getArgs();
        TypeRange getArgTypes();
    }];
}

#endif // SDFG_SDFGNode
