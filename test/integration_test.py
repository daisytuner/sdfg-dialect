import pytest
import subprocess
from pathlib import Path

@pytest.mark.parametrize("model_name, opset_version", [
    ("alexnet_Opset17", 17),
    ("bert_Opset18", 18),
    ("darknetaa53_Opset18", 18),
    ("gptneox_Opset18", 18),
    ("retinanet_resnet50_fpn_v2_Opset18", 18),
    ("sageconv_Opset18", 18),
])
def test_models(model_name, opset_version):
    model_dir = Path(__file__).parent.parent / "models"
    model_path = model_dir / f"{model_name}.onnx"

    # Step 1: Import ONNX models to Torch MLIR
    cmd = ["iree-import-onnx", f"--opset-version={opset_version}", str(model_path.absolute()), "-o", f"{model_name}.torch.mlir"]
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
    cmd = [str(sdfg_opt_path.absolute()), "--torch-to-sdfg", "--allow-unregistered-dialect", f"{model_name}.torch.mlir", "-o", f"{model_name}.sdfg.mlir"]
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
