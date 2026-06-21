"""Step 1: ASR转录 + 说话人分离"""

import os
import json
from pathlib import Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline
import torch
import config


def extract_audio(video_path: str, audio_path: str):
    """从视频提取音频"""
    import ffmpeg
    if os.path.exists(audio_path):
        return
    ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).overwrite_output().run(quiet=True)


def transcribe(audio_path: str) -> list[dict]:
    """ASR转录，返回带时间戳的segments"""
    import subprocess
    from tqdm import tqdm

    # 获取音频时长用于进度条
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip()) if result.stdout.strip() else None

    model = WhisperModel(config.ASR_MODEL, device=config.ASR_DEVICE, compute_type="float16")
    segments, _ = model.transcribe(audio_path, language=config.ASR_LANGUAGE, beam_size=5)

    results = []
    with tqdm(total=duration, unit="s", desc="    ASR进度") as pbar:
        last_end = 0
        for s in segments:
            results.append({"start": s.start, "end": s.end, "text": s.text})
            pbar.update(s.end - last_end)
            last_end = s.end
    return results


def diarize(audio_path: str) -> list[dict]:
    """说话人分离"""
    hf_token = os.environ.get("HF_TOKEN")
    pipeline = DiarizationPipeline.from_pretrained(
        config.DIARIZATION_MODEL, use_auth_token=hf_token
    )
    pipeline.to(torch.device(config.ASR_DEVICE))
    diarization = pipeline(
        audio_path,
        min_speakers=config.DIARIZATION_MIN_SPEAKERS,
        max_speakers=config.DIARIZATION_MAX_SPEAKERS,
    )
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return results


def align_transcript_with_speakers(transcript: list[dict], diarization: list[dict]) -> list[dict]:
    """将ASR结果与说话人对齐"""
    aligned = []
    for seg in transcript:
        mid = (seg["start"] + seg["end"]) / 2
        speaker = "UNKNOWN"
        for d in diarization:
            if d["start"] <= mid <= d["end"]:
                speaker = d["speaker"]
                break
        aligned.append({**seg, "speaker": speaker})
    return aligned


def process_video(video_path: str, output_dir: str) -> str:
    """处理单个视频，返回转录JSON路径"""
    video_name = Path(video_path).stem
    audio_path = os.path.join(output_dir, f"{video_name}.wav")
    transcript_path = os.path.join(output_dir, f"{video_name}_transcript.json")

    if os.path.exists(transcript_path):
        print(f"  跳过已处理: {video_name}")
        return transcript_path

    print(f"  提取音频: {video_name}")
    extract_audio(video_path, audio_path)

    print(f"  ASR转录中...")
    transcript = transcribe(audio_path)

    print(f"  说话人分离中...")
    diarization_result = diarize(audio_path)

    print(f"  对齐说话人...")
    aligned = align_transcript_with_speakers(transcript, diarization_result)

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    # 清理音频文件节省空间
    os.remove(audio_path)
    print(f"  完成: {len(aligned)} segments")
    return transcript_path
