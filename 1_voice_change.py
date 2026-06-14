"""
对已剪辑的视频进行说话人分离 + 变声 + 合成
支持多说话人，选择一人保持原音，其余变声，每人可指定参考音频。

用法: python 1_voice_change.py <trimmed.mp4> [output_dir]
流程: 输入说话人数 -> 说话人分离 -> 选择保留原音者 -> 其余变声 -> 合成输出

依赖: pip install pyannote.audio pydub python-dotenv praat-parselmouth soundfile torch
系统依赖: ffmpeg
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime
import soundfile as sf
import torch
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})
import parselmouth
from parselmouth.praat import call
from dotenv import load_dotenv
from pydub import AudioSegment
from pyannote.audio import Pipeline

try:
    import bigvgan
    _orig_from_pretrained = bigvgan.BigVGAN._from_pretrained.__func__
    @classmethod
    def _patched_from_pretrained(cls, *, proxies=None, resume_download=None, **kwargs):
        return _orig_from_pretrained(cls, proxies=proxies, resume_download=resume_download, **kwargs)
    bigvgan.BigVGAN._from_pretrained = _patched_from_pretrained
except ImportError:
    pass

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_VOICES_DIR = os.path.join(SCRIPT_DIR, "ref_voices")


# ==================== 变声方法 ====================

def transform_praat_pitch(input_wav, output_wav, gender):
    sound = parselmouth.Sound(input_wav)
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()
    if gender == "male":
        factor, floor, ceiling = 0.8, 50, 200
    else:
        factor, floor, ceiling = 1.25, 100, 400
    manipulation = call(sound, "To Manipulation", 0.01, floor, ceiling)
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
    call([pitch_tier, manipulation], "Replace pitch tier")
    new_sound = call(manipulation, "Get resynthesis (overlap-add)")
    sf.write(output_wav, new_sound.values[0], int(new_sound.sampling_frequency))


def transform_praat_formant(input_wav, output_wav, gender):
    sound = parselmouth.Sound(input_wav)
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()
    if gender == "male":
        formant_ratio, new_pitch_median, pitch_range_factor = 0.82, 100.0, 0.6
        floor, ceiling = 50, 300
    else:
        formant_ratio, new_pitch_median, pitch_range_factor = 1.35, 310.0, 1.5
        floor, ceiling = 50, 500
    new_sound = call(sound, "Change gender", floor, ceiling, formant_ratio, new_pitch_median, pitch_range_factor, 1.0)
    sf.write(output_wav, new_sound.values[0], int(new_sound.sampling_frequency))


def transform_seed_vc(input_wav, output_wav, gender, ref_audio=None):
    import seed_vc_local
    if ref_audio is None:
        ref_audio = os.path.join(REF_VOICES_DIR, f"{gender}_ref.wav")
    if not os.path.exists(ref_audio):
        sys.exit(f"参考音频不存在: {ref_audio}")
    seed_vc_local.convert(source_wav=input_wav, target_wav=ref_audio, output_wav=output_wav, diffusion_steps=25)


METHODS = {
    "1": ("Praat PSOLA (本地，快速)", transform_praat_pitch),
    "2": ("Praat 共振峰+音高 (本地，效果较好)", transform_praat_formant),
    "3": ("Seed-VC AI变声 (本地GPU，效果最好)", transform_seed_vc),
}


# ==================== 主流程 ====================

def main(input_mp4, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # 输入说话人个数
    num_speakers = int(input("请输入说话人个数: ").strip())
    if num_speakers < 2:
        sys.exit("说话人数至少为2")

    audio_work = AudioSegment.from_file(input_mp4)
    total_ms = len(audio_work)

    # Step 1: 说话人分离
    audio_16k = audio_work.set_frame_rate(16000).set_channels(1)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_16k.export(tmp_wav.name, format="wav")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("请设置环境变量 HF_TOKEN")

    print(f"正在进行说话人分离 (num_speakers={num_speakers})...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    diarization = pipeline(tmp_wav.name, num_speakers=num_speakers)
    os.unlink(tmp_wav.name)
    print("说话人分离完成")

    annotation = diarization.speaker_diarization if hasattr(diarization, 'speaker_diarization') else diarization
    mapped_segments = []
    for seg, _, speaker in annotation.itertracks(yield_label=True):
        mapped_segments.append((int(seg.start * 1000), int(seg.end * 1000), speaker))

    speaker_labels = sorted(set(sp for _, _, sp in mapped_segments))
    print(f"\n检测到 {len(speaker_labels)} 个说话人: {speaker_labels}")

    # Step 2: 导出示例音频
    print("\n正在导出说话人示例音频...")
    for label in speaker_labels:
        clips = [(s, e) for s, e, sp in mapped_segments if sp == label]
        if clips:
            longest = max(clips, key=lambda x: x[1] - x[0])
            sample = audio_work[longest[0]:longest[1]]
            sample_path = os.path.join(output_dir, f"sample_{label}.wav")
            sample.export(sample_path, format="wav")
            print(f"  {label}: {sample_path} ({(longest[1]-longest[0])/1000:.1f}s)")

    # Step 3: 选择保持原音的说话人
    print("\n请试听示例音频后，选择保持原音（不变声）的说话人。")
    valid_ids = [label.replace("SPEAKER_", "") for label in speaker_labels]
    keep_id = input(f"输入保持原音的说话人编号 ({'/'.join(valid_ids)}): ").strip()
    keep_label = f"SPEAKER_{keep_id.zfill(2)}"
    if keep_label not in speaker_labels:
        sys.exit(f"无效编号: {keep_id}")
    print(f"  {keep_label} 保持原音")

    # Step 4: 为需要变声的说话人配置参数
    voices_to_change = [l for l in speaker_labels if l != keep_label]

    print("\n变声方式:")
    for k, (desc, _) in METHODS.items():
        print(f"  {k}. {desc}")
    method_choice = input(f"选择变声方式 (1/2/3, 默认3): ").strip() or "3"
    if method_choice not in METHODS:
        sys.exit("无效选择")

    # 为每个变声说话人配置
    speaker_configs = {}  # {label: {"gender": ..., "ref_audio": ...}}
    for label in voices_to_change:
        print(f"\n配置 {label}:")
        gender = input(f"  {label} 的性别 (m/f): ").strip().lower()
        gender = "male" if gender == "m" else "female"
        ref_audio = None
        if method_choice == "3":
            ref_path = input(f"  参考音频wav路径 (直接回车用默认): ").strip()
            if ref_path and os.path.exists(ref_path):
                ref_audio = ref_path
            elif ref_path:
                print(f"  文件不存在，使用默认")
        speaker_configs[label] = {"gender": gender, "ref_audio": ref_audio}

    # Step 5: 变声
    method_name, method_fn = METHODS[method_choice]
    all_transformed = {}  # {label: [(start_ms, end_ms, clip), ...]}

    for label, config in speaker_configs.items():
        target_clips = [(s, e) for s, e, sp in mapped_segments if sp == label]
        if not target_clips:
            continue

        print(f"\n正在对 {label} 变声 [{method_name}]... ({len(target_clips)} 片段)")
        transformed = []
        for idx, (start_ms, end_ms) in enumerate(target_clips):
            clip = audio_work[start_ms:end_ms]
            print(f"  {idx+1}/{len(target_clips)}", end="", flush=True)

            if len(clip) < 200 or clip.dBFS < -45:
                transformed.append((start_ms, end_ms, clip))
                print(" skip", end="", flush=True)
                continue

            tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            clip.export(tmp_in.name, format="wav")

            try:
                if method_choice == "3":
                    method_fn(tmp_in.name, tmp_out.name, config["gender"], config["ref_audio"])
                else:
                    method_fn(tmp_in.name, tmp_out.name, config["gender"])
                result_clip = AudioSegment.from_file(tmp_out.name)
                if len(result_clip) > len(clip):
                    result_clip = result_clip[:len(clip)]
                elif len(result_clip) < len(clip):
                    result_clip += AudioSegment.silent(duration=len(clip) - len(result_clip))
                if result_clip.dBFS < -45:
                    raise Exception("静音")
                if clip.dBFS > -45 and result_clip.dBFS > -45:
                    result_clip = result_clip.apply_gain(clip.dBFS - result_clip.dBFS)
                transformed.append((start_ms, end_ms, result_clip))
                print(" ✔", end="", flush=True)
            except Exception as e:
                if method_choice == "2":
                    try:
                        transform_praat_pitch(tmp_in.name, tmp_out.name, config["gender"])
                        result_clip = AudioSegment.from_file(tmp_out.name)
                        if len(result_clip) > len(clip):
                            result_clip = result_clip[:len(clip)]
                        elif len(result_clip) < len(clip):
                            result_clip += AudioSegment.silent(duration=len(clip) - len(result_clip))
                        transformed.append((start_ms, end_ms, result_clip))
                        print(" ✔(备选)", end="", flush=True)
                    except:
                        transformed.append((start_ms, end_ms, clip))
                        print(" ✘", end="", flush=True)
                else:
                    transformed.append((start_ms, end_ms, clip))
                    print(f" ✘", end="", flush=True)
            finally:
                os.unlink(tmp_in.name)
                os.unlink(tmp_out.name)

        all_transformed[label] = transformed
        print()

    # Step 6: 混合音轨
    mixed = audio_work[:]
    for label, transformed in all_transformed.items():
        for start_ms, end_ms, clip in transformed:
            mixed = mixed[:start_ms] + clip + mixed[end_ms:]

    mixed_wav = os.path.join(output_dir, "mixed_audio.wav")
    mixed.export(mixed_wav, format="wav")

    # Step 7: 合成输出
    mp4_basename = os.path.splitext(os.path.basename(input_mp4))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_mp4 = os.path.join(output_dir, f"{mp4_basename}_{timestamp}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_mp4,
        "-i", mixed_wav,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_mp4
    ]
    print("\n正在合成视频...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"完成! 输出: {output_mp4}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法: python 1_voice_change.py <trimmed.mp4> [output_dir]")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    main(sys.argv[1], output_dir)
