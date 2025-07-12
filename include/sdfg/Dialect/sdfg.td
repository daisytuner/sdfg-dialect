#ifndef SDFG_SDFGNode
#define SDFG_SDFGNode

include "mlir/IR/OpBase.td"

/// Defining the SDFG
def SDFG_SDFGNode : SDFG_Op<"sdfg", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<FunctionOpInterface>
]> {
    let summary = "SDFG function (drop-in replacement for func.func)";
    let description = [{
        A stateful dataflow multigraph function that serves as a drop-in replacement for func.func.
        The SDFG function can contain library nodes and other SDFG operations while maintaining
        function-like semantics with arguments, results, and a body region. Supports all attributes and
        interfaces required by func.func, including argument/result attributes and FunctionOpInterface.
    }];
    let hasCustomAssemblyFormat = 1;

    let regions = (region SizedRegion<1>:$body);

    let arguments = (ins 
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<ArrayAttr>:$arg_attrs,
        OptionalAttr<ArrayAttr>:$res_attrs
    );

    let extraClassDeclaration = [{
        static SDFGNode create(PatternRewriter &rewriter, Location location, StringRef name, TypeRange inputs, TypeRange results);
        Block::BlockArgListType getFunctionArgs();
        mlir::Region* getCallableRegion() const;
        mlir::ArrayRef<mlir::Type> getArgumentTypes() const;
        mlir::ArrayRef<mlir::Type> getResultTypes() const;
    }];
}

#endif // SDFG_SDFGNode
