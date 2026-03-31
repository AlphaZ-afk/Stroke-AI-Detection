import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def record_audio(filename="recorded.wav", duration=5, fs=16000):
    print("[DEBUG SPEECH] Recording... 🎤")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("[DEBUG SPEECH] Recording complete ✅")
    return filename

def analyze_speech_live(duration=5):
    try:
        path = record_audio(duration=duration)
        y, sr = librosa.load(path, duration=duration)
        
        if len(y) == 0 or np.max(np.abs(y)) < 0.001:
            print("[DEBUG SPEECH] Audio array is empty or purely silent.")
            return 0.0
            
        # 1. Clarity (MFCC Variance)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.var(mfcc) # Low variance = monotone
        
        # 2. Pauses (Silence Ratio)
        non_silent_intervals = librosa.effects.split(y, top_db=20)
        non_silent_samples = sum([end - start for start, end in non_silent_intervals]) if len(non_silent_intervals) > 0 else 0
        silence_ratio = 1.0 - (non_silent_samples / len(y))
        
        # 3. Speech Rate (Onset Detection)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames', wait=10, pre_avg=3, post_avg=3, pre_max=3, post_max=3)
        speech_rate = len(onset_frames) / float(duration)
        
        # Scale to strictly 0-100 Ranges. 
        # Add a baseline of 5.0 risk for healthy patients to avoid "always 0.0" concerns.
        risk_mfcc = max(5.0, min((500.0 - mfcc_var) / 5.0, 100.0)) if mfcc_var < 500 else 5.0
        risk_pauses = max(5.0, min((silence_ratio - 0.2) * 125.0, 100.0)) if silence_ratio > 0.2 else 5.0
        risk_rate = max(5.0, min((1.5 - speech_rate) * 66.6, 100.0)) if speech_rate < 1.5 else 5.0
            
        combined_risk = (risk_mfcc * 0.4) + (risk_pauses * 0.3) + (risk_rate * 0.3)
        combined_risk = max(5.0, min(combined_risk, 100.0))
        
        print(f"\n[DEBUG SPEECH] MFCC Var: {mfcc_var:.1f} (Risk Component: {risk_mfcc:.1f}/100)")
        print(f"[DEBUG SPEECH] Pause Ratio: {silence_ratio:.3f} (Risk Component: {risk_pauses:.1f}/100)")
        print(f"[DEBUG SPEECH] Speech Rate: {speech_rate:.2f}/s (Risk Component: {risk_rate:.1f}/100)")
        print(f"[DEBUG SPEECH] Final Combined Speech Risk: {combined_risk:.1f}/100\n")
        
        return float(combined_risk)
        
    except Exception as e:
        print(f"[DEBUG SPEECH] Processing failed: {e}")
        return 0.0