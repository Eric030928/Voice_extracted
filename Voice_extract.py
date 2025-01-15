import os
from ffmpy3 import FFmpeg
import pydub
import numpy as np
import torch
import wave
import contextlib
import subprocess
import librosa
import soundfile as sf
import noisereduce as nr

# 初始化 Silero VAD 模型，用于语音活动检测
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github')
(sample_rate, min_buffer_duration) = (16000, 0.6)

def mkdir_output(output_dir):
    """
    创建输出目录，如果目录不存在则创建。
    """
    if not os.path.exists(output_dir):
        print('创建音频存放目录')
        os.makedirs(output_dir)
    else:
        print('目录已存在,即将保存！')

def int2float(sound):
    """
    将整数格式的音频数据转换为浮点数格式。
    """
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    return sound.squeeze()

def audio_to_wave(audio_path, target_path="temp.wav"):
    """
    将音频文件转换为WAV格式，并设置为单声道和16000 Hz采样率。
    """
    audio = pydub.AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(target_path, format="wav")

def frame_generator(audio, sample_rate):
    """
    将音频数据分帧，生成每帧的音频数据。
    """
    n = 512  # 每帧的样本数
    offset = 0
    timestamp = 0.0
    duration = (n / sample_rate) * 1000.0  # 单位为毫秒
    while offset + n <= len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

class Frame:
    """
    表示音频帧的类，包含帧数据、时间戳和持续时间。
    """
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def vad_collector(frames, sample_rate):
    """
    使用VAD模型检测语音活动，返回语音片段的起始和结束时间。
    """
    voiced_frames = []
    for frame in frames:
        audio_frame_np = np.frombuffer(frame.bytes, dtype=np.int16)
        audio_float32 = int2float(audio_frame_np)
        with torch.no_grad():
            new_confidence = model(torch.from_numpy(audio_float32), sample_rate).item()
        if new_confidence > 0.63:
            voiced_frames.append(frame)
        elif voiced_frames:
            start, end = voiced_frames[0].timestamp, voiced_frames[-1].timestamp + voiced_frames[-1].duration
            voiced_frames = []
            yield start, end
    if voiced_frames:
        start, end = voiced_frames[0].timestamp, voiced_frames[-1].timestamp + voiced_frames[-1].duration
        yield start, end

def merge_segments(segments, merge_distance=3500):
    """
    合并相邻的语音片段，如果它们之间的间隔小于给定的距离。
    """
    merged_segments = []
    for start, end in segments:
        if merged_segments and start - merged_segments[-1][1] <= merge_distance:
            merged_segments[-1] = (merged_segments[-1][0], end)
        else:
            merged_segments.append((start, end))
    return merged_segments

def read_wave(path):
    """
    读取WAV文件，返回PCM数据和采样率。
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def write_wave(path, audio: np.ndarray, sample_rate):
    """
    将音频数据写入WAV文件。
    """
    audio = audio.astype(np.int16)
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

def detect_speech_segments(audio_path, output_path):
    """
    检测音频文件中的语音片段并保存到新的音频文件中。
    """
    audio_to_wave(audio_path)
    pcm_data, sample_rate = read_wave("temp.wav")
    audio_np = np.frombuffer(pcm_data, dtype=np.int16)
    
    # 添加静音帧以避免第一个字被裁切
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.int16)  # 1秒的静音
    audio_np = np.concatenate((silence, audio_np))
    
    frames = frame_generator(audio_np, sample_rate)
    segments = list(vad_collector(list(frames), sample_rate))
    merged_segments = merge_segments(segments)

    # 在片段之间插入0.5秒的空白
    silence = np.zeros(int(sample_rate * 0.4))  # 0.5秒的空白
    all_voiced_audio = []
    for start, end in merged_segments:
        all_voiced_audio.append(audio_np[int(start * sample_rate / 1000):int(end * sample_rate / 1000)])
        all_voiced_audio.append(silence)

    all_voiced_audio = np.concatenate(all_voiced_audio)
    write_wave(output_path, all_voiced_audio, sample_rate)
    print(f"Speech segment saved: {output_path}")

def extract_audio_from_video(video_path, audio_path):
    """
    从视频文件中提取音频，并保存为WAV格式。
    """
    ff = FFmpeg(
        inputs={video_path: None},
        outputs={audio_path: '-vn -ar 16000 -ac 1 -f wav -y'}
    )
    ff.run()

def separate_audio_with_demucs(audio_path, output_dir):
    """
    使用Demucs算法分离音频中的人声和背景音。
    """
    command = [
        'demucs',
        '--two-stems=vocals',  # 指定只提取人声
        '-d', 'cpu',  # 使用 CPU 进行处理，如果有 GPU 可用，替换为 'cuda'
        '-n', 'htdemucs_ft',
        audio_path,
        '-o', output_dir
    ]
    subprocess.run(command, check=True)

def reduce_noise_with_librosa(input_path, output_path):
    """
    使用Librosa库通过频谱减法进行降噪处理。
    """
    y, sr = librosa.load(input_path, sr=None)
    
    # 使用 librosa 的 STFT 进行频域分析
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(stft)

    # 估计噪声谱：使用更复杂的策略，例如自适应阈值
    noise_estimation = np.median(magnitude, axis=1, keepdims=True)
    noise_reduction_factor = 1.5  # 增加降噪强度
    magnitude_denoised = np.maximum(magnitude - noise_reduction_factor * noise_estimation, 0)

    # 反向变换
    y_denoised = librosa.istft(magnitude_denoised * phase, hop_length=512)

    # 使用 noisereduce 进一步降噪
    y_denoised = nr.reduce_noise(y=y_denoised, sr=sr, n_std_thresh=1.0, prop_decrease=0.95)

    sf.write(output_path, y_denoised, sr)

if __name__ == '__main__':
    """
    主程序流程：从输入目录中读取视频文件，提取音频，检测语音片段，分离人声，进行降噪处理，最终输出处理后的人声音频。
    """
    input_dir = r"/Volumes/文枢工作空间/input"
    output_dir = r"/Users/zhanshiquan/Downloads/输出/人物"
    mkdir_output(output_dir)

    file_counter = 1

for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.mov', '.mp4', '.m4a', '.wav')):
            continue

        video_path = os.path.join(input_dir, filename)
        temp_audio_path = os.path.join(output_dir, f"temp{file_counter}.wav")
        voiced_audio_path = os.path.join(output_dir, f"voiced{file_counter}.wav")

        extract_audio_from_video(video_path, temp_audio_path)
        detect_speech_segments(temp_audio_path, voiced_audio_path)

        separate_output_dir = os.path.join(output_dir, f"output{file_counter}")
        mkdir_output(separate_output_dir)
        separate_audio_with_demucs(voiced_audio_path, separate_output_dir)

        vocals_path = os.path.join(separate_output_dir, 'vocals.wav')
        output_vocals_path = os.path.join(output_dir, f"{file_counter}_vocals.wav")
        if os.path.exists(vocals_path):
            os.rename(vocals_path, output_vocals_path)

            # 使用谱减法进行降噪
            noise_reduced_path = os.path.join(output_dir, f"{file_counter}_vocals_reduced.wav")
            reduce_noise_with_librosa(output_vocals_path, noise_reduced_path)

        os.remove(temp_audio_path)
        os.remove(voiced_audio_path)

        file_counter += 1