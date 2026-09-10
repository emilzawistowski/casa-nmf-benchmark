import os
import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 44100
TARGET_DURATION = 25.0     
TARGET_RMS_DBFS = -20.0    
TRIM_START_SEC = 1.0         

def rms_dbfs(signal):
    rms = np.sqrt(np.mean(signal**2) + 1e-12)
    return 20 * np.log10(rms + 1e-12)

def normalize_to_dbfs(signal, target_dbfs):
    current_dbfs = rms_dbfs(signal)
    gain_db = target_dbfs - current_dbfs
    gain_lin = 10 ** (gain_db / 20.0)
    out = signal * gain_lin
    max_abs = np.max(np.abs(out)) + 1e-12
    if max_abs > 0.999:
        out = out / max_abs * 0.999
    return out

def trim_segment(signal, sr, start_sec, target_duration):
    start_sample = int(start_sec * sr)
    target_samples = int(target_duration * sr)
    if start_sample >= len(signal):
        return np.zeros(target_samples, dtype=np.float32)
    segment = signal[start_sample:start_sample + target_samples]
    if len(segment) < target_samples:
        pad_len = target_samples - len(segment)
        segment = np.concatenate([segment, np.zeros(pad_len, dtype=segment.dtype)])
    return segment

def load_mono(path, sr_target=TARGET_SR):
    audio, sr = librosa.load(path, sr=sr_target, mono=True)
    return audio, sr_target

def process_single_file(in_path, out_path):
    print(f"Processing single file: {in_path} -> {out_path}")
    audio, sr = load_mono(in_path, TARGET_SR)
    audio = trim_segment(audio, sr, TRIM_START_SEC, TARGET_DURATION)
    audio = normalize_to_dbfs(audio, TARGET_RMS_DBFS)
    sf.write(out_path, audio, sr, subtype='PCM_16')
    print(f"Done: {out_path}")

def process_choir(sop_path, alt_path, ten_path, bas_path, out_path):
    print(f"Processing choir: {out_path}")
    voices = []
    for path in [sop_path, alt_path, ten_path, bas_path]:
        if not os.path.exists(path):
            print(f"WARNING: {path} doesnt exist.")
            continue
        audio, sr = load_mono(path, TARGET_SR)
        seg = trim_segment(audio, sr, TRIM_START_SEC, TARGET_DURATION)
        voices.append(seg)

    if len(voices) == 0:
        print("No speech.")
        return

    min_len = min(len(v) for v in voices)
    voices = [v[:min_len] for v in voices]

    mix = np.sum(np.stack(voices, axis=0), axis=0)

    mix = normalize_to_dbfs(mix, TARGET_RMS_DBFS)

    sf.write(out_path, mix, TARGET_SR, subtype='PCM_16')
    print(f"Done choir mix: {out_path}")

if __name__ == "__main__":

    SOPRANO_IN = "audio_data/soprano.wav"
    ALTO_IN    = "audio_data/alto.wav"
    TENOR_IN   = "audio_data/tenor.wav"
    BASS_IN    = "audio_data/bass.wav"
    CHOIR_OUT  = "audio_data/audio_data_ready_to_go/choir.wav"

    RAINFOREST_IN     = "audio_data/rainforest_long.wav"
    REVERB_SPEECH_IN  = "audio_data/reverb_speech_long.wav"
    RAINFOREST_OUT    = "audio_data/audio_data_ready_to_go/rainforest.wav"
    REVERB_SPEECH_OUT = "audio_data/audio_data_ready_to_go/reverb_speech.wav"

    process_choir(SOPRANO_IN, ALTO_IN, TENOR_IN, BASS_IN, CHOIR_OUT)

 
    if os.path.exists(RAINFOREST_IN):
        process_single_file(RAINFOREST_IN, RAINFOREST_OUT)
    else:
        print(f"WARNING: {RAINFOREST_IN} not exist.")

    if os.path.exists(REVERB_SPEECH_IN):
        process_single_file(REVERB_SPEECH_IN, REVERB_SPEECH_OUT)
    else:
        print(f"WARNING: {REVERB_SPEECH_IN} not exist.")
