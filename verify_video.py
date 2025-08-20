# verify_video.py
# - 視覺：MediaPipe 抽樣臉偵測，近似單人判定
# - 語音：ffmpeg 抽音 -> VAD(webrtcvad) 篩出有聲片段 -> Vosk
#   * 中文通道：完整轉寫 + 中文數字正規化
#   * 英文通道：Grammar 模式僅擷取關鍵字 ["sugarbee"]（更快更穩）
# - 比對：sugarbee 由任一通道聽到 + 數字取中文通道（失敗再用英文通道備援）
# - 所有輸出為 Python 原生型別

import os, sys, json, re, tempfile, subprocess, collections
import numpy as np
import cv2
import mediapipe as mp
from unidecode import unidecode

# ======================
# ffmpeg: 自動偵測路徑
# ======================
def find_ffmpeg():
    for path in os.environ.get("PATH", "").split(os.pathsep):
        for name in ("ffmpeg.exe", "ffmpeg"):
            exe = os.path.join(path, name)
            if os.path.exists(exe):
                return exe
    fallback = r"C:\ffmpeg\bin\ffmpeg.exe"
    if os.path.exists(fallback):
        return fallback
    raise RuntimeError("找不到 ffmpeg，請安裝並加入 PATH，或把它放在 C:\\ffmpeg\\bin。")

FFMPEG_BIN = find_ffmpeg()

# ==================================
# 視覺：以臉數近似「是否只有一個人」
# ==================================
def check_single_person(video_path, sample_fps=2, min_coverage=0.80):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))

    mp_fd = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

    idx, has_face, per_frame_counts = 0, 0, []
    ret, frame = cap.read()
    while ret:
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_fd.process(rgb)
            n = len(res.detections) if res.detections else 0
            per_frame_counts.append(n)
            if n >= 1:
                has_face += 1
        idx += 1
        ret, frame = cap.read()

    cap.release()
    total = len(per_frame_counts) or 1
    coverage = has_face / total
    max_faces = max(per_frame_counts) if per_frame_counts else 0
    ok = (coverage >= min_coverage) and (max_faces <= 1)
    return bool(ok), float(coverage), [int(x) for x in per_frame_counts]

# ======================
# 音訊：抽音（ffmpeg）
# ======================
def extract_audio_to_wav_ffmpeg(video_path):
    """
    以 ffmpeg 抽音成 16kHz、單聲道、PCM s16le WAV。
    回傳臨時 wav 路徑（呼叫端負責刪除）。
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav = tmp.name
    tmp.close()

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        "-acodec", "pcm_s16le",
        tmp_wav
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("找不到 ffmpeg。請確認已安裝並設定 PATH，或把 ffmpeg.exe 放在 C:\\ffmpeg\\bin。")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 抽音失敗：{e.stderr.decode('utf-8', errors='ignore')[:400]}")
    return tmp_wav

# ======================
# VAD（webrtcvad）：擷取有聲 PCM bytes
# ======================
def read_wav_bytes(wav_path):
    import wave
    wf = wave.open(wav_path, "rb")
    sr = wf.getframerate()
    ch = wf.getnchannels()
    width = wf.getsampwidth()
    data = wf.readframes(wf.getnframes())
    wf.close()
    return sr, ch, width, data

def frame_generator(frame_ms, audio_bytes, sample_rate):
    """將整段 PCM bytes 切成固定長度 frame（bytes）。"""
    n_bytes_per_sample = 2  # 16-bit
    bytes_per_frame = int(sample_rate * (frame_ms / 1000.0)) * n_bytes_per_sample
    offset = 0
    while offset + bytes_per_frame <= len(audio_bytes):
        yield audio_bytes[offset:offset + bytes_per_frame]
        offset += bytes_per_frame

def vad_collect(audio_bytes, sample_rate=16000, frame_ms=30, padding_ms=300, aggressiveness=2):
    """
    使用 webrtcvad 擷取有聲片段，輸出為「合併後的有聲 PCM bytes」。
    """
    import webrtcvad
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frame_generator(frame_ms, audio_bytes, sample_rate))

    num_padding_frames = int(padding_ms / frame_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_bytes = bytearray()
    collected = []

    for f in frames:
        is_speech = vad.is_speech(f, sample_rate)
        if not triggered:
            ring_buffer.append((f, is_speech))
            num_voiced = len([1 for _, s in ring_buffer if s])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # 包含 ring_buffer 中的所有樣本
                for fr, _ in ring_buffer:
                    voiced_bytes.extend(fr)
                ring_buffer.clear()
        else:
            voiced_bytes.extend(f)
            ring_buffer.append((f, is_speech))
            num_unvoiced = len([1 for _, s in ring_buffer if not s])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # 一段語音結束
                collected.append(bytes(voiced_bytes))
                voiced_bytes = bytearray()
                ring_buffer.clear()
                triggered = False

    # 尾端若仍在 triggered 狀態，補上一段
    if voiced_bytes:
        collected.append(bytes(voiced_bytes))

    # 合併所有有聲段（對短句最簡潔）
    return b"".join(collected) if collected else b""

# ======================
# Vosk 模型路徑解析
# ======================
def resolve_vosk_model_dir(path):
    need = ["conf", "am"]
    try:
        if all(os.path.isdir(os.path.join(path, d)) for d in need):
            return path
        entries = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if len(entries) == 1:
            sub = os.path.join(path, entries[0])
            if all(os.path.isdir(os.path.join(sub, d)) for d in need):
                return sub
    except Exception:
        pass
    raise RuntimeError(f"Vosk model folder invalid: {path}（請指定含 conf/am 的那一層資料夾）")

# =========================================
# 文字正規化（英文 + 中文數字 → 阿拉伯數字）
# =========================================
_ZH_PINYIN_TO_HANZI = {
    "ling":"零","yi":"一","er":"二","liang":"兩","san":"三","si":"四","wu":"五",
    "liu":"六","qi":"七","ba":"八","jiu":"九","shi":"十","bai":"百",
}
_ZH_DIGIT = {"零":0,"〇":0,"一":1,"二":2,"兩":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}

def _pinyin_to_hanzi(text: str) -> str:
    def repl(m): return _ZH_PINYIN_TO_HANZI.get(m.group(0), m.group(0))
    return re.sub(r"\b(ling|yi|er|liang|san|si|wu|liu|qi|ba|jiu|shi|bai)\b", repl, text)

def _zh_numeral_to_int(s: str):
    if not s: return None
    if all(ch in _ZH_DIGIT for ch in s):  # 逐字
        val = 0
        for ch in s: val = val*10 + _ZH_DIGIT[ch]
        return val
    s = s.replace("兩","二")
    if "百" in s:
        parts = s.split("百", 1)
        h = _ZH_DIGIT.get(parts[0], 1) if parts[0] else 1
        rest = parts[1]
        tens = ones = 0
        if "十" in rest:
            p2 = rest.split("十", 1)
            tens = 1 if (p2[0]=="" or p2[0]=="零") else _ZH_DIGIT.get(p2[0], 0)
            ones = _ZH_DIGIT.get(p2[1], 0) if p2[1] else 0
        else:
            if rest.startswith("零"):
                ones = _ZH_DIGIT.get(rest[1:2], 0)
            elif rest:
                if len(rest)==1 and rest in _ZH_DIGIT:
                    tens = _ZH_DIGIT[rest]; ones = 0   # 二百三 → 230
                else:
                    ones = _ZH_DIGIT.get(rest, 0)
        return h*100 + tens*10 + ones
    if "十" in s:
        p = s.split("十", 1)
        tens = 1 if p[0]=="" else _ZH_DIGIT.get(p[0], 0)
        ones = _ZH_DIGIT.get(p[1], 0) if p[1] else 0
        return tens*10 + ones
    if s in _ZH_DIGIT: return _ZH_DIGIT[s]
    return None

def _zh_numbers_to_arabic(text: str) -> str:
    text = text.replace("〇","零")
    zh = "零一二兩三四五六七八九十百"
    out = []; i = 0
    while i < len(text):
        if text[i] in zh:
            j = i
            while j < len(text) and text[j] in zh: j += 1
            seg = text[i:j]; val = _zh_numeral_to_int(seg)
            out.append(str(val) if val is not None and 0 <= val <= 1000 else seg); i = j
        else:
            out.append(text[i]); i += 1
    return "".join(out)

def normalize_en(text: str) -> str:
    t = unidecode(text).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    vocab = {
        "zero":"0","oh":"0","o":"0","one":"1","two":"2","three":"3","four":"4","for":"4",
        "five":"5","six":"6","seven":"7","eight":"8","ate":"8","nine":"9","ten":"10",
    }
    toks = [vocab.get(tok, tok) for tok in t.split()]
    s = "".join(toks)
    s = s.replace("sugar bee","sugarbee").replace("suger","sugar")
    return s

def normalize_zh(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff\s〇零一二兩三四五六七八九十百]", " ", t)
    t = t.replace("sugar bee","sugarbee").replace("suger","sugar")
    t = _pinyin_to_hanzi(t)
    t = _zh_numbers_to_arabic(t)
    t = re.sub(r"\s+", "", t)
    return t

# ======================
# Vosk ASR：以 bytes 餵入（支援 Grammar）
# ======================
def vosk_recognize_bytes(pcm_bytes, model_dir, sample_rate=16000, grammar=None):
    from vosk import Model, KaldiRecognizer
    import json as _json
    model = Model(resolve_vosk_model_dir(model_dir))
    if grammar:
        rec = KaldiRecognizer(model, sample_rate, json.dumps(grammar))
    else:
        rec = KaldiRecognizer(model, sample_rate)
    rec.SetWords(True)

    # 分塊餵入
    chunk = 4000 * 2  # 4000 samples * 2 bytes
    pos = 0; texts = []
    while pos < len(pcm_bytes):
        data = pcm_bytes[pos:pos+chunk]
        pos += chunk
        if rec.AcceptWaveform(data):
            res = _json.loads(rec.Result())
            texts.append(res.get("text",""))
    res = _json.loads(rec.FinalResult())
    texts.append(res.get("text",""))
    return " ".join([t for t in texts if t]).strip()

# ======================
# 混合比對流程：VAD + 中英雙通道
# ======================
def mixed_speech_check(video_path, expected_phrase, zh_model=None, en_model=None):
    # 期望值切出數字
    exp_norm_en = normalize_en(expected_phrase)
    m = re.findall(r"(\d+)", exp_norm_en)
    expected_digits = m[-1] if m else ""
    expected_keyword = "sugarbee"

    wav = extract_audio_to_wav_ffmpeg(video_path)
    try:
        sr, ch, width, pcm = read_wav_bytes(wav)
        if not (sr == 16000 and ch == 1 and width == 2):
            raise RuntimeError("WAV 格式非 16k 單聲道 16-bit")
        voiced = vad_collect(pcm, sample_rate=sr, frame_ms=30, padding_ms=300, aggressiveness=2)
        if not voiced:
            # 若未偵測到有聲片段，退而求其次用全段
            voiced = pcm

        raw_zh = norm_zh = raw_en = norm_en = ""

        if zh_model:
            raw_zh = vosk_recognize_bytes(voiced, zh_model, sample_rate=sr, grammar=None)
            norm_zh = normalize_zh(raw_zh)

        if en_model:
            # 只聽 "sugarbee" 關鍵字，提升速度與準確度
            raw_en = vosk_recognize_bytes(voiced, en_model, sample_rate=sr, grammar=["sugarbee"])
            # 有些時候會輸出空字串或重複字，照樣正規化
            norm_en = normalize_en(raw_en)

    finally:
        try: os.remove(wav)
        except: pass

    sugarbee_ok = ("sugarbee" in norm_en) or ("sugarbee" in norm_zh)

    # 數字：優先中文通道，其次英文通道
    def last_digits(s):
        nums = re.findall(r"\d+", s)
        return nums[-1] if nums else ""
    spoken_digits = last_digits(norm_zh) or last_digits(norm_en)
    number_ok = (spoken_digits == expected_digits)

    final_ok = bool(sugarbee_ok and number_ok)
    info = {
        "expected_keyword": expected_keyword,
        "expected_digits": expected_digits,
        "raw_zh": str(raw_zh),
        "norm_zh": str(norm_zh),
        "raw_en": str(raw_en),
        "norm_en": str(norm_en),
        "sugarbee_ok": bool(sugarbee_ok),
        "spoken_digits": str(spoken_digits),
        "number_ok": bool(number_ok),
        "used_vad": True
    }
    return final_ok, info

# ======================
# 主流程
# ======================
def verify_video(video_path, expected_phrase, vosk_model_zh=None, vosk_model_en=None):
    # 1) 單人
    single_ok, coverage, counts = check_single_person(video_path, sample_fps=2, min_coverage=0.80)

    # 2) 語音（VAD + 中英雙通道）
    phrase_ok, speech_info = mixed_speech_check(
        video_path,
        expected_phrase,
        zh_model=vosk_model_zh,
        en_model=vosk_model_en,
    )

    return {
        "single_person_ok": bool(single_ok),
        "face_coverage": float(round(coverage, 3)),
        "max_faces_in_any_sample": int(max(counts) if counts else 0),
        "speech_match_ok": bool(phrase_ok),
        "speech_detail": speech_info,
        "expected_normalized_en": normalize_en(expected_phrase),
        "final_ok": bool(single_ok and phrase_ok)
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to selfie video (mp4/mov/etc.)")
    ap.add_argument("expected", help="e.g. sugarbee100（英文 sugarbee + 中文數字也 OK）")
    ap.add_argument("--vosk_model_zh", required=True, help="中文 Vosk 模型資料夾（含 conf/am 那層）")
    ap.add_argument("--vosk_model_en", required=True, help="英文 Vosk 模型資料夾（含 conf/am 那層）")
    args = ap.parse_args()

    res = verify_video(args.video, args.expected, vosk_model_zh=args.vosk_model_zh, vosk_model_en=args.vosk_model_en)
    print(json.dumps(res, ensure_ascii=False, indent=2))
