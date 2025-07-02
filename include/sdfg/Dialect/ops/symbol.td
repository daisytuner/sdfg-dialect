// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Table-driven file for symbol operations.

#ifndef SDFG_SymbolOps
#define SDFG_SymbolOps

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

/// Defining the SDFG symbol definition operation.
def SDFG_AllocSymbolOp : SDFG_Op<"symbol", [
	ParentOneOf<["SDFGNode"]>
]> {
    let summary = "symbol creation operation";
    let description = [{
        Alloc operation to create a new symbol.
    }];

    let arguments = (ins StrAttr:$sym);

    let assemblyFormat = [{
        `(` $sym `)` attr-dict
    }];

    let extraClassDeclaration = [{
        static AllocSymbolOp create(PatternRewriter &rewriter, Location loc, StringRef sym);
        static AllocSymbolOp create(Location loc, StringRef sym);
    }];
}

//===----------------------------------------------------------------------===//
// SymOp
//===----------------------------------------------------------------------===//

/// Defining the SDFG symbolic expression.
def SDFG_SymOp : SDFG_Op<"expression",[
	ParentOneOf<["SDFGNode"]>
]>{
    let summary = "symbolic expression operation";
    let description = [{
        Describes an arithmetic symbolic expression.
    }];

    let assemblyFormat = [{
        `(` $expr `)` attr-dict `:` type($res)
    }];

    let arguments = (ins StrAttr:$expr);
    let results = (outs AnyTypeOf<[AnyInteger, Index]>:$res);
    let extraClassDeclaration = [{
        static SymOp create(PatternRewriter &rewriter, Location loc, Type t, StringRef expr);
        static SymOp create(Location loc, Type t, StringRef expr);
    }];
}

#endif // SDFG_SymbolOps
