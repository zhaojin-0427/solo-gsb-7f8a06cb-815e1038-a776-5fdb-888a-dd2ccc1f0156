"""本地房间脉冲响应 RT60（混响时间）测量 API。

POST /rt60 上传单声道 16 位 PCM WAV（16 或 48 kHz，时长 1~10 秒），
基于 Schroeder 反向积分衰减曲线，由 -5 dB / -25 dB 越线时间外推 RT60。
"""

import io
import math
import wave
from typing import Literal, Optional, Union

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# ---------------------------------------------------------------- 常量

ALLOWED_SAMPLE_RATES = {16000, 48000}   # 允许的采样率 (Hz)
MIN_DURATION_S = 1.0                    # 最短时长 (s)
MAX_DURATION_S = 10.0                   # 最长时长 (s)

NOISE_TAIL_FRACTION = 0.2               # 用末尾 20% 样本估计噪声
FRAME_S = 0.020                         # 尾端检测帧长 20 ms
FRAME_NOISE_RATIO = 2.0                 # 帧 RMS 不高于噪声 RMS 的 2 倍视为噪声帧
CONSEC_NOISE_FRAMES = 3                 # 需连续 3 帧
PEAK_NOISE_RATIO = 4.0                  # 峰值须高于 4 倍噪声 RMS
MIN_DECAY_S = 0.100                     # 有效区间至少 100 ms
TAIL_MARGIN_S = 0.020                   # -25 dB 越线点距尾端至少 20 ms

LEVEL_NEAR_DB = -5.0                    # 近端越线电平
LEVEL_FAR_DB = -25.0                    # 远端越线电平
DB_FLOOR = -120.0                       # 衰减曲线分贝下限，避免 log(0)

app = FastAPI(
    title="RT60 Measurement API",
    version="1.0.0",
    description="上传单声道 16 位 PCM WAV 脉冲响应，测量混响时间 RT60。",
)


# ---------------------------------------------------------------- 响应模型

class RT60Success(BaseModel):
    """测量成功：各时间点均为相对录音起点的秒数。"""

    measurable: Literal[True] = True
    peak_time: float      # 绝对峰值位置 (s)
    end_time: float       # 有效衰减区间尾端 (s)
    t5: float             # 首次越过 -5 dB 的时间 (s)
    t25: float            # 首次越过 -25 dB 的时间 (s)
    rt60: float           # 3 * (t25 - t5) (s)


class RT60Failure(BaseModel):
    """无法测量：给出原因，不给出 RT60。"""

    measurable: Literal[False] = False
    reason: str


# ---------------------------------------------------------------- WAV 读取与校验

def _unprocessable(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


def _load_wav(data: bytes) -> tuple[np.ndarray, int]:
    """解析并校验 WAV，返回 (float64 单声道样本, 采样率)。格式或范围错误抛 422。"""
    if not data:
        raise _unprocessable("上传文件为空")
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            nframes = w.getnframes()
            raw = w.readframes(nframes)
    except HTTPException:
        raise
    except Exception as exc:  # wave.Error / EOFError / struct.error 等
        raise _unprocessable(f"无法解析 WAV 文件（需为 PCM WAV）: {exc}")

    if channels != 1:
        raise _unprocessable(f"仅支持单声道 WAV，当前声道数: {channels}")
    if sampwidth != 2:
        raise _unprocessable(f"仅支持 16 位 PCM，当前样本宽度: {sampwidth * 8} 位")
    if framerate not in ALLOWED_SAMPLE_RATES:
        raise _unprocessable(f"采样率须为 16000 或 48000 Hz，当前: {framerate} Hz")
    duration = nframes / framerate
    if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        raise _unprocessable(
            f"时长须在 {MIN_DURATION_S:.0f}~{MAX_DURATION_S:.0f} 秒之间，当前: {duration:.3f} 秒"
        )

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if samples.size != nframes:
        raise _unprocessable("WAV 数据损坏：样本数与帧数不一致")
    return samples, framerate


# ---------------------------------------------------------------- 核心算法

def _find_effective_end(x: np.ndarray, peak_idx: int, noise_rms: float, fs: int) -> int:
    """从峰值起按 20 ms 帧计算原始 RMS，首次连续 3 帧均不高于 2 倍噪声 RMS 时，
    以首帧起点为有效尾端；否则取录音终点。"""
    n = x.size
    frame = max(1, int(round(FRAME_S * fs)))
    threshold = FRAME_NOISE_RATIO * noise_rms
    run = 0
    start = peak_idx
    while start < n:
        seg = x[start:min(start + frame, n)]
        rms = math.sqrt(float(np.mean(seg * seg)))
        if rms <= threshold:
            run += 1
            if run == CONSEC_NOISE_FRAMES:
                return start - (CONSEC_NOISE_FRAMES - 1) * frame
        else:
            run = 0
        start += frame
    return n


def _first_crossing_time(db: np.ndarray, level_db: float, peak_idx: int, fs: int) -> Optional[float]:
    """衰减曲线上首次越过 level_db 的时间（相邻样本线性插值），缺失返回 None。"""
    below = np.nonzero(db <= level_db)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:  # 起点即越线（实际不会发生，db[0] = 0）
        return peak_idx / fs
    d0 = float(db[i - 1])  # > level_db
    d1 = float(db[i])      # <= level_db
    frac = (d0 - level_db) / (d0 - d1)
    return (peak_idx + (i - 1) + frac) / fs


def _analyze(x: np.ndarray, fs: int) -> Union[RT60Success, RT60Failure]:
    n = x.size

    # 绝对峰值
    abs_x = np.abs(x)
    peak_idx = int(np.argmax(abs_x))
    peak_value = float(abs_x[peak_idx])

    # 末尾 20% 样本估计噪声均方功率
    tail_len = max(1, int(round(NOISE_TAIL_FRACTION * n)))
    tail = x[n - tail_len:]
    noise_power = float(np.mean(tail * tail))
    noise_rms = math.sqrt(noise_power)

    # 峰值须高于 4 倍噪声 RMS
    if peak_value <= PEAK_NOISE_RATIO * noise_rms:
        return RT60Failure(reason="峰值不高于 4 倍噪声 RMS，信噪比不足，无法测量")

    # 有效尾端：噪声为零时直接取录音终点，否则按帧检测
    if noise_power == 0.0:
        end_idx = n
    else:
        end_idx = _find_effective_end(x, peak_idx, noise_rms, fs)

    # 有效区间须不短于 100 ms
    if (end_idx - peak_idx) / fs < MIN_DECAY_S:
        return RT60Failure(reason="有效衰减区间短于 100 ms，无法测量")

    # 逐样本扣除噪声功率后做 Schroeder 反向积分，以起点能量归一化为分贝
    seg = x[peak_idx:end_idx]
    e = np.maximum(seg * seg - noise_power, 0.0)
    energy = np.cumsum(e[::-1])[::-1]
    total = float(energy[0])
    if total <= 0.0:
        return RT60Failure(reason="有效区间能量为零，无法测量")
    floor = total * 10.0 ** (DB_FLOOR / 10.0)
    db = 10.0 * np.log10(np.maximum(energy, floor) / total)

    # -5 dB / -25 dB 越线时间（线性插值）
    t5 = _first_crossing_time(db, LEVEL_NEAR_DB, peak_idx, fs)
    t25 = _first_crossing_time(db, LEVEL_FAR_DB, peak_idx, fs)
    if t5 is None or t25 is None:
        missing = []
        if t5 is None:
            missing.append("-5 dB")
        if t25 is None:
            missing.append("-25 dB")
        return RT60Failure(reason=f"衰减曲线未越过 {' 和 '.join(missing)}，无法测量")

    # -25 dB 越线点距尾端须不小于 20 ms
    end_time = end_idx / fs
    if end_time - t25 < TAIL_MARGIN_S:
        return RT60Failure(reason="-25 dB 越线点距有效尾端不足 20 ms，无法测量")

    rt60 = 3.0 * (t25 - t5)
    return RT60Success(
        peak_time=round(peak_idx / fs, 6),
        end_time=round(end_time, 6),
        t5=round(t5, 6),
        t25=round(t25, 6),
        rt60=round(rt60, 6),
    )


# ---------------------------------------------------------------- 路由

@app.get("/")
def index() -> dict:
    return {
        "service": "RT60 Measurement API",
        "usage": "POST /rt60 上传单声道 16 位 PCM WAV（16/48 kHz，1~10 s）",
        "docs": "/docs",
    }


@app.post("/rt60", response_model=Union[RT60Success, RT60Failure])
async def measure_rt60(file: UploadFile = File(...)) -> Union[RT60Success, RT60Failure]:
    data = await file.read()
    samples, fs = _load_wav(data)
    return _analyze(samples, fs)
