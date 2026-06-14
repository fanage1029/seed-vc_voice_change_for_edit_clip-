"""
Seed-VC 本地推理封装
调用 Seed-VC/inference.py 进行语音转换
"""

import os
import sys
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_VC_DIR = os.path.join(SCRIPT_DIR, "Seed-VC")


def convert(source_wav, target_wav, output_wav, diffusion_steps=25, device="cuda"):
    """
    调用 Seed-VC 进行语音转换
    source_wav: 源音频（要变声的）
    target_wav: 目标音色参考音频
    output_wav: 输出路径
    """
    if not os.path.exists(SEED_VC_DIR):
        raise RuntimeError(f"Seed-VC 未安装，请运行: cd {SCRIPT_DIR} && bash setup_gpu.sh")

    inference_py = os.path.join(SEED_VC_DIR, "inference.py")

    # 创建输出目录
    output_dir = tempfile.mkdtemp()

    cmd = [
        sys.executable, inference_py,
        "--source", os.path.abspath(source_wav),
        "--target", os.path.abspath(target_wav),
        "--output", output_dir,
        "--diffusion-steps", str(diffusion_steps),
        "--length-adjust", "1.0",
        "--inference-cfg-rate", "0.5",
        "--f0-condition", "false",
        "--semi-tone-shift", "0",
    ]

    result = subprocess.run(
        cmd,
        cwd=SEED_VC_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # 清理
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        raise RuntimeError(f"Seed-VC 推理失败:\n{result.stderr[-500:]}")

    # 找到输出文件
    output_files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
    if not output_files:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        raise RuntimeError("Seed-VC 未生成输出文件")

    # 复制输出文件
    import shutil
    src_file = os.path.join(output_dir, output_files[0])
    shutil.copy(src_file, output_wav)
    shutil.rmtree(output_dir, ignore_errors=True)
