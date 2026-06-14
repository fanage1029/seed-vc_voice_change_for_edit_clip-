# 视频智能剪辑 + 变声 使用指南

## 概述

两个独立脚本，可分开使用：

1. **`separate_speakers_v7.py`** — 智能剪辑（字幕识别 → AI评分 → 裁剪 → 变声 → 合成）
2. **`1_voice_change.py`** — 纯变声处理（多说话人分离 → 选择保留原音者 → 其余变声 → 合成）

## 环境准备

### 1. 系统依赖

```bash
sudo apt install ffmpeg
```

### 2. Python 依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install pyannote.audio pydub python-dotenv praat-parselmouth soundfile openai-whisper transformers torch
```

### 3. 安装 Seed-VC（变声方式3，可选）

如果需要使用 Seed-VC AI 变声（效果最好），需额外安装：

```bash
cd training/
git clone https://github.com/Plachta/Seed-VC.git
cd Seed-VC
pip install -r requirements.txt
```

或直接运行安装脚本：

```bash
bash setup_gpu.sh
```

要求：
- NVIDIA GPU（V100/P40 等），CUDA 11.8+
- Python 3.10+
- 模型首次运行时自动下载到 `~/.cache/huggingface/`

如果不安装 Seed-VC，仍可使用变声方式 1（Praat PSOLA）和方式 2（Praat 共振峰），无需 GPU。

### 3. 环境变量

创建 `.env` 文件（放在 `training/` 目录下）：

```env
HF_TOKEN=你的HuggingFace访问令牌
```

获取 HF_TOKEN：https://huggingface.co/settings/tokens

还需要同意以下模型的使用协议：
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### 4. GPU 要求

- Qwen2.5-7B float16：~14GB 显存
- Whisper base：~1GB 显存
- Pyannote：~2GB 显存
- **V100 32GB 完全够用**

## 使用方法

### separate_speakers_v7.py — 智能剪辑 + 变声

```bash
python separate_speakers_v7.py <input.mp4>
```

全自动流程：
1. Whisper 识别中文字幕
2. Qwen2.5-7B 对每条字幕评分（话题热度/信息重要性/爆点潜力）
3. 自动选择高分连贯片段，每个切片 ≤ 5分钟
4. 说话人分离（pyannote，固定2人）
5. 交互选择变声说话人和方式
6. 变声 + 合成输出视频

### 1_voice_change.py — 多说话人变声

```bash
python 1_voice_change.py <trimmed.mp4> [output_dir]
```

交互流程：
1. 输入说话人个数（支持2人及以上）
2. 说话人分离
3. 试听各说话人示例音频
4. **选择一人保持原音**（其余所有人变声）
5. 为每个变声说话人配置：
   - 性别（m/f）
   - 参考音频 wav 路径（可选，回车用默认）
6. 变声 + 合成输出视频

交互示例：
```
请输入说话人个数: 3
...
输入保持原音的说话人编号 (00/01/02): 01
选择变声方式 (1/2/3, 默认3): 3
配置 SPEAKER_00:
  SPEAKER_00 的性别 (m/f): m
  参考音频wav路径 (直接回车用默认): /path/to/voice1.wav
配置 SPEAKER_02:
  SPEAKER_02 的性别 (m/f): f
  参考音频wav路径 (直接回车用默认):
```

## 变声方式

| 方式 | 说明 | 速度 | 效果 |
|------|------|------|------|
| 1 | Praat PSOLA 音高变换 | 快 | 一般 |
| 2 | Praat 共振峰+音高联合变换 | 快 | 较好 |
| 3 | Seed-VC AI变声（需参考音频） | 慢(GPU) | 最好 |

方式3的参考音频：
- 默认使用 `ref_voices/male_ref.wav` 或 `ref_voices/female_ref.wav`
- 可在交互时输入自定义 wav 文件路径

## 制作参考音频

如果你有一段 mp4 视频，想提取其中的声音作为变声参考音频：

```bash
# 从 mp4 提取为 wav（单声道，22050Hz，适合 Seed-VC）
ffmpeg -i input.mp4 -vn -ac 1 -ar 22050 -acodec pcm_s16le ref_voice.wav
```

建议：
- 参考音频 5~30 秒即可，选一段目标声音清晰、无背景音乐的片段
- 如果只想截取某一段（比如第10秒到第25秒）：

```bash
ffmpeg -i input.mp4 -ss 10 -to 25 -vn -ac 1 -ar 22050 -acodec pcm_s16le ref_voice.wav
```

制作好后，运行 `1_voice_change.py` 时在提示 `参考音频wav路径` 处输入该文件路径即可。

也可以放到 `ref_voices/` 目录下作为默认参考音频：
```bash
cp ref_voice.wav ref_voices/male_ref.wav   # 或 female_ref.wav
```

## 模型说明

| 模型 | 用途 | 首次自动下载 | 大小 |
|------|------|-------------|------|
| Qwen2.5-7B-Instruct | 内容评分（v7） | ✔ | ~14GB |
| whisper base | 字幕识别 | ✔ | ~140MB |
| pyannote/speaker-diarization-3.1 | 说话人分离 | ✔ | ~200MB |

所有模型缓存在 `~/.cache/huggingface/hub/`，无需手动放置。

## 预估耗时（V100 32GB，30分钟视频）

| 步骤 | 耗时 |
|------|------|
| 字幕识别 | 3-5 分钟 |
| AI 评分（v7） | 1-2 分钟 |
| 视频裁剪 | 2-5 分钟 |
| 说话人分离 | 3-5 分钟 |
| 变声（Seed-VC） | 5-15 分钟 |
| **总计** | **15-30 分钟** |
