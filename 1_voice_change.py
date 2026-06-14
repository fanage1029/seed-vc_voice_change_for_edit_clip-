"""
对已剪辑的视频进行说话人分离 + 变声 + 合成

用法: python voice_change.py <trimmed.mp4> [output_dir]
流程: 说话人分离 -> 选择说话人 -> 变声 -> 合成输出

依赖: pip install pyannote.audio pydub python-dotenv praat-parselmouth soundfile torch
系统依赖: ffmpeg
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime
import numpy as np
import soundfile as sf
import torch
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})
import parselmouth
from parselmouth.praat import call
from dotenv import load_dotenv
from pydub import AudioSegment
from pyannote.audio import Pipeline

# Patch BigVGAN 兼容新版 huggingface_hub
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
    """Praat PSOLA 音高变换"""
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
    """Praat 共振峰+音高联合变换"""
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
    """Seed-VC AI变声"""
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


def prompt_choice(msg, valid, retries=5, default=None):
    for _ in range(retries):
        val = input(msg).strip().lower()
        if val == "" and default:
            return default
        if val in valid:
            return val
        print(f"无效输入，请输入: {'/'.join(v for v in valid if v)}")
    sys.exit("连续5次输入错误，退出")


# ==================== 主流程 ====================

def main(input_mp4, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    audio_work = AudioSegment.from_file(input_mp4)
    total_ms = len(audio_work)

    # Step 1: 说话人分离（直接对输入视频做）
    audio_16k = audio_work.set_frame_rate(16000).set_channels(1)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_16k.export(tmp_wav.name, format="wav")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("请设置环境变量 HF_TOKEN")

    print("正在进行说话人分离...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    diarization = pipeline(tmp_wav.name, num_speakers=2)
    os.unlink(tmp_wav.name)
    print("说话人分离完成")

    # 提取分离结果
    mapped_segments = []
    annotation = diarization.speaker_diarization if hasattr(diarization, 'speaker_diarization') else diarization
    for seg, _, speaker in annotation.itertracks(yield_label=True):
        mapped_segments.append((int(seg.start * 1000), int(seg.end * 1000), speaker))

    # Step 2: 导出说话人示例音频供试听
    print("\n正在导出说话人示例音频...")
    for label in ["SPEAKER_00", "SPEAKER_01"]:
        clips = [(s, e) for s, e, sp in mapped_segments if sp == label]
        if clips:
            clips_sorted = sorted(clips, key=lambda x: x[1] - x[0], reverse=True)
            sample = None
            for s, e in clips_sorted:
                candidate = audio_work[s:e]
                if candidate.dBFS > -40:
                    sample = candidate
                    longest = (s, e)
                    break
            if sample is None:
                longest = clips_sorted[0]
                sample = audio_work[longest[0]:longest[1]]
            sample_path = os.path.join(output_dir, f"sample_{label}.wav")
            sample.export(sample_path, format="wav")
            print(f"  {label} 示例: {sample_path} (时长 {(longest[1]-longest[0])/1000:.1f}s)")

    # Step 3: 用户选择
    print("\n请试听上面的示例音频后选择要变声的说话人。")
    speaker_choice = prompt_choice("选择要变声的说话人 (00/01): ", ("00", "01"))
    gender = prompt_choice("该说话人的性别 (m/f): ", ("m", "f"))
    gender = "male" if gender == "m" else "female"

    print("\n变声方式:")
    for k, (desc, _) in METHODS.items():
        print(f"  {k}. {desc}")
    method_choice = prompt_choice("选择变声方式 (1/2/3, 默认3): ", ("1", "2", "3", ""), default="3")

    ref_audio = None
    if method_choice == "3":
        use_custom = input("是否使用自定义参考音频? (y/n, 默认n): ").strip().lower()
        if use_custom == "y":
            ref_audio = input("请输入参考音频路径: ").strip()
            if not os.path.exists(ref_audio):
                print("文件不存在，将使用默认参考音频")
                ref_audio = None

    target_label = f"SPEAKER_{speaker_choice}"

    # Step 4: 变声
    target_clips = [(s, e) for s, e, sp in mapped_segments if sp == target_label]

    if not target_clips:
        sys.exit(f"{target_label} 没有语音片段，退出")

    method_name, method_fn = METHODS[method_choice]
    print(f"\n正在对 {target_label} 进行变声 [{method_name}]...")

    transformed_clips = []
    for idx, (start_ms, end_ms) in enumerate(target_clips):
        clip = audio_work[start_ms:end_ms]
        s_min, s_sec = divmod(start_ms // 1000, 60)
        e_min, e_sec = divmod(end_ms // 1000, 60)
        print(f"  片段 {idx+1}/{len(target_clips)}: {s_min:02d}:{s_sec:02d}~{e_min:02d}:{e_sec:02d}", end="")

        if len(clip) < 200:
            transformed_clips.append((start_ms, end_ms, clip))
            print(" (跳过，太短)")
            continue

        if clip.dBFS < -35:
            # 尝试提高音量后再判断是否真的是静音
            boosted = clip.apply_gain(-16 - clip.dBFS)  # 归一化到 -16 dBFS
            if boosted.max_dBFS < -30:
                transformed_clips.append((start_ms, end_ms, clip))
                print(" (跳过，静音)")
                continue
            # 不是真静音，用提高音量后的版本继续处理
            clip = boosted

        tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        clip.export(tmp_in.name, format="wav")

        try:
            if method_choice == "3":
                method_fn(tmp_in.name, tmp_out.name, gender, ref_audio)
            else:
                method_fn(tmp_in.name, tmp_out.name, gender)
            result_clip = AudioSegment.from_file(tmp_out.name)
            if len(result_clip) > len(clip):
                result_clip = result_clip[:len(clip)]
            elif len(result_clip) < len(clip):
                result_clip += AudioSegment.silent(duration=len(clip) - len(result_clip))
            if result_clip.dBFS < -45:
                raise Exception("变声结果为静音")
            if clip.dBFS > -45 and result_clip.dBFS > -45:
                volume_diff = clip.dBFS - result_clip.dBFS
                result_clip = result_clip.apply_gain(volume_diff)
            transformed_clips.append((start_ms, end_ms, result_clip))
            print(" ✔")
        except Exception as e:
            if method_choice == "2":
                try:
                    transform_praat_pitch(tmp_in.name, tmp_out.name, gender)
                    result_clip = AudioSegment.from_file(tmp_out.name)
                    if len(result_clip) > len(clip):
                        result_clip = result_clip[:len(clip)]
                    elif len(result_clip) < len(clip):
                        result_clip += AudioSegment.silent(duration=len(clip) - len(result_clip))
                    transformed_clips.append((start_ms, end_ms, result_clip))
                    print(" ✔ (PSOLA备选)")
                except:
                    transformed_clips.append((start_ms, end_ms, clip))
                    print(" ✘ 使用原音")
            else:
                transformed_clips.append((start_ms, end_ms, clip))
                print(f" ✘ 使用原音 ({e})")
        finally:
            os.unlink(tmp_in.name)
            os.unlink(tmp_out.name)

    # Step 5: 混合音轨（以原始音频为底，仅替换变声片段）
    # 先统一所有变声片段的音量到目标水平
    target_dBFS = -16
    mixed = audio_work[:]
    for start_ms, end_ms, clip in transformed_clips:
        if clip.dBFS < -45:
            # 真静音，不调整
            mixed = mixed[:start_ms] + clip + mixed[end_ms:]
        else:
            gain = target_dBFS - clip.dBFS
            normalized_clip = clip.apply_gain(gain)
            mixed = mixed[:start_ms] + normalized_clip + mixed[end_ms:]

    mixed_wav = os.path.join(output_dir, "mixed_audio.wav")
    mixed.export(mixed_wav, format="wav")

    # Step 6: 合成输出视频（带音量归一化）
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
    print("正在合成视频...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"\n完成! 输出: {output_mp4}")
    print(f"  时长: {total_ms/1000:.0f}s ({total_ms/60000:.1f}分钟)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法: python voice_change.py <trimmed.mp4> [output_dir]")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    main(sys.argv[1], output_dir)
