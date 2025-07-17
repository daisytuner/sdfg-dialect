# SDFG MLIR Dialect

A dialect for [Stateful Dataflow Multigraphs](https://github.com/daisytuner/sdfglib) in MLIR.
The implementation consists of the dialect and conversion passes with a focus on torch-mlir.

## Dependencies

The implementation depends on IREE, which packages main dialects and useful tools such as torch-mlir and onnx-to-torch-conversion.
Accordingly, LLVM, MLIR and other dialects are automatically build with IREE.

## Build and Test

### Step 0: venv and LIT

```bash
python3 -m venv venv
source venv/bin/activate

pip install lit
```

### Step 1: IREE Dependencies

```bash
cd 3rdParty/iree/

pip install hatch
pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx==1.17.0
```

### Step 2: SDFGLib

Build [sdfglib](https://github.com/daisytuner/sdfglib) according to its instructions and install it to some <SDFGLIB-PREFIX-PATH>.
The library exports its cmake configurations with the installation and can simply be provided to the next build command.

### Step 2: Build

```bash
mkdir build && cd build

cmake -GNinja \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_C_COMPILER=clang-19 \
 -DCMAKE_CXX_COMPILER=clang++-19 \
 -DIREE_ENABLE_ASSERTIONS=ON \
 -DIREE_ENABLE_SPLIT_DWARF=ON \
 -DIREE_ENABLE_THIN_ARCHIVES=ON \
 -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
 -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
 -DIREE_HAL_DRIVER_DEFAULTS=OFF \
 -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
 -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
 -DIREE_BUILD_PYTHON_BINDINGS=ON \
 -DPython3_EXECUTABLE="$(which python3)" \
 -DIREE_INPUT_STABLEHLO=OFF \
 -DIREE_INPUT_TOSA=OFF \
 -DIREE_BUILD_TESTS=OFF \
 -DLLVM_EXTERNAL_LIT=<path-to-external-lit> \
 -Dsdfglib_DIR=<SDFGLIB-PREFIX-PATH>/lib/cmake/sdfglib \
 -DSymEngine_DIR=<SDFGLIB-PREFIX-PATH>/lib/cmake/symengine \
 ..

cmake --build . --target check-sdfg-opt
cmake --build . --target sdfg-export
```

### Step 3: Python Bindings

From the build directory:

```bash
cd 3rdParty/iree/compiler
hatch build
pip install dist/iree_base_compiler-*.whl

cd ../runtime
hatch build
pip install dist/iree_base_runtime-*.whl
```

## Usage

The basic flow is to convert ONNX or any other format to torch-mlir and then convert it to an SDFG.

### Step 0: Import ONNX (IREE)

```bash
cd examples/
iree-import-onnx --opset-version=17 squeezenet1.0-3.onnx -o squeezenet1.0-3.torch.mlir
```

### Step 1: Torch MLIR -> SDFG Dialect

```bash
../build/bin/sdfg-opt --torch-to-sdfg --allow-unregistered-dialect squeezenet1.0-3.torch.mlir -o squeezenet1.0-3.sdfg.mlir
```

### Step 2: SDFG Dialect -> SDFG

```bash
../build/bin/sdfg-export --export-sdfg squeezenet1.0-3.sdfg.mlir
```
