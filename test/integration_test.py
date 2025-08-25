import pytest
import subprocess
from pathlib import Path
import re
from collections import Counter
import os

EXPECTED_OPS = {
    "alexnet_Opset17": {
        "ReLU": 7,
        "Conv": 5,
        "MaxPool": 3,
        "Gemm": 3,
        "AveragePool": 1,
        "Flatten": 1,
        "Metadata": 2,
    },
    "bert_Opset18": {
        "Add": 122,
        "MatMul": 96,
        "Reshape": 48,
        "Transpose": 48,
        "Mul": 26,
        "LayerNormalization": 25,
        "Div": 24,
        "Softmax": 12,
        "Erf": 12,
        "Gather": 4,
        "Unsqueeze": 2,
        "ConstantOfShape": 1,
        "Equal": 1,
        "Where": 1,
        "Expand": 1,
        "Cast": 1,
        "Sub": 1,
        "Gemm": 1,
        "Tanh": 1,
        "Metadata": 107,
    },
    "darknetaa53_Opset18": {
        "Conv": 52,
        "LeakyReLU": 52,
        "Add": 23,
        "AveragePool": 5,
        "GlobalAveragePool": 1,
        "Flatten": 1,
        "Gemm": 1,
        "Metadata": 7,
    },
    "gptneox_Opset18": {
        "Add": 55,
        "Slice": 55,
        "Identity": 38,
        "Mul": 36,
        "MatMul": 30,
        "Reshape": 26,
        "Transpose": 25,
        "Concat": 20,
        "Unsqueeze": 12,
        "Gather": 11,
        "LayerNormalization": 11,
        "Neg": 10,
        "Cast": 6,
        "Where": 5,
        "Softmax": 5,
        "Div": 5,
        "Erf": 5,
        "Sub": 1,
        "Metadata": 208,
    },
    "retinanet_resnet50_fpn_v2_Opset18": {
        "Unsqueeze": 161,
        "Reshape": 157,
        "Shape": 140,
        "Add": 123,
        "Gather": 120,
        "Conv": 111,
        "Mul": 110,
        "ReLU": 90,
        "Cast": 85,
        "Concat": 85,
        "Identity": 64,
        "Slice": 55,
        "Div": 41,
        "InstanceNormalization": 40,
        "Sub": 30,
        "ConstantOfShape": 21,
        "Transpose": 21,
        "Min": 11,
        "Squeeze": 10,
        "Range": 10,
        "Expand": 10,
        "NonZero": 10,
        "Less": 10,
        "Mod": 10,
        "Clip": 10,
        "Exp": 10,
        "Max": 10,
        "Split": 7,
        "ReduceMin": 6,
        "Equal": 6,
        "Sigmoid": 5,
        "Greater": 5,
        "GatherND": 5,
        "TopK": 5,
        "Xor": 5,
        "Not": 5,
        "And": 5,
        "Where": 5,
        "Flatten": 5,
        "ReduceMax": 4,
        "Resize": 3,
        "Reciprocal": 2,
        "Floor": 2,
        "Ceil": 2,
        "Pad": 1,
        "MaxPool": 1,
        "ReduceProd": 1,
        "If": 1,
        "Metadata": 1125,
    },
    "sageconv_Opset18": {
        "Gather": 3,
        "ScatterElements": 2,
        "Reshape": 2,
        "Shape": 2,
        "Expand": 2,
        "Clip": 1,
        "Div": 1,
        "Gemm": 1,
        "MatMul": 1,
        "Add": 1,
        "Metadata": 12,
    },
}


@pytest.mark.parametrize(
    "model_name, opset_version",
    [
        ("alexnet_Opset17", 17),
        ("bert_Opset18", 18),
        ("darknetaa53_Opset18", 18),
        ("gptneox_Opset18", 18),
        ("retinanet_resnet50_fpn_v2_Opset18", 18),
        ("sageconv_Opset18", 18),
    ],
)
def test_models(model_name, opset_version):
    model_dir = Path(__file__).parent.parent / "models"
    model_path = model_dir / f"{model_name}.onnx"

    # Step 1: Import ONNX models to Torch MLIR
    cmd = [
        "iree-import-onnx",
        f"--opset-version={opset_version}",
        str(model_path.absolute()),
        "-o",
        f"{model_name}.torch.mlir",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    # Step 2: Convert Torch MLIR to SDFG Dialect
    sdfg_opt_path = Path(__file__).parent.parent / "build" / "bin" / "sdfg-opt"
    cmd = [
        str(sdfg_opt_path.absolute()),
        "--torch-to-sdfg",
        "--allow-unregistered-dialect",
        f"{model_name}.torch.mlir",
        "-o",
        f"{model_name}.sdfg.mlir",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    # Step 3: Convert SDFG Dialect to SDFG
    sdfg_export_path = Path(__file__).parent.parent / "build" / "bin" / "sdfg-export"
    cmd = [str(sdfg_export_path.absolute()), "--export-sdfg", f"{model_name}.sdfg.mlir"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    output_sdfg = f"{model_name}.sdfg.mlir.main_graph.json"
    output_dot = f"{model_name}.sdfg.mlir.main_graph.dot"

    assert Path(output_sdfg).exists()
    assert Path(output_dot).exists()

    found_ops = Counter()
    with open(output_sdfg, "r") as handle:
        for l in handle.readlines():
            line = l.strip()
            # Match regex "code": "ml::*" and print the matched group
            # Example: "code": "ml::Gemm",
            match = re.match(r"\"code\": \"ml::(.*)\"", line)
            if match:
                op_name = match.group(1)
                found_ops.update([op_name])
                continue

            # Match metadata
            if line == '"code": "Metadata",':
                found_ops.update(["Metadata"])
                continue

    print(found_ops)
    for op_name, count in EXPECTED_OPS[model_name].items():
        assert found_ops[op_name] == count

    # Clean up
    os.remove(output_sdfg)
    os.remove(output_dot)
    os.remove(f"{model_name}.sdfg.mlir")
    os.remove(f"{model_name}.torch.mlir")
