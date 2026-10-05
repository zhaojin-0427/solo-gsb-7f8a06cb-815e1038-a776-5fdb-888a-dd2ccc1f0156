# RT60 混响时间测量 API

本地服务：上传单声道房间脉冲响应 WAV，基于 Schroeder 反向积分衰减曲线，由 -5 dB / -25 dB 越线时间线性插值并外推得到 RT60（`RT60 = 3 × (t25 − t5)`）。

## 一条命令启动

```bash
docker compose up --build
```

启动后本地访问：

- API 地址：http://localhost:8000
- 交互式文档（Swagger UI）：http://localhost:8000/docs

（也可使用 `docker-compose up --build`，端口同为 **8000**。）

## 接口

### `POST /rt60`

以 `multipart/form-data` 上传字段名为 `file` 的 WAV 文件。

**输入限制**（违反任一条件返回 `422`）：

- 单声道、16 位 PCM WAV
- 采样率仅支持 16000 或 48000 Hz
- 时长 1 ~ 10 秒

**算法概要**：

1. 以绝对峰值位置为起点；
2. 用末尾 20% 样本估计噪声均方功率；
3. 从峰值起按 20 ms 帧计算原始 RMS，首次连续 3 帧不高于 2 倍噪声 RMS 时以首帧起点为有效尾端，否则取录音终点（噪声为零时直接取录音终点）；
4. 有效区间内逐样本取 `max(样本² − 噪声功率, 0)`，反向累积并以起点能量归一化为分贝衰减曲线；
5. 对首次越过 -5 dB、-25 dB 的相邻样本做线性插值，`RT60 = 3 × (t25 − t5)`。

**测量成功**（`200`）：

```json
{
  "measurable": true,
  "peak_time": 0.5,
  "end_time": 2.34,
  "t5": 0.541667,
  "t25": 0.708333,
  "rt60": 0.5
}
```

各时间均为相对录音起点的秒数。

**无法测量**（`200`，不给出 RT60）：峰值不高于 4 倍噪声 RMS、有效区间短于 100 ms、能量为零、-5/-25 dB 越线点缺失、或 -25 dB 越线点距尾端不足 20 ms。

```json
{ "measurable": false, "reason": "有效衰减区间短于 100 ms，无法测量" }
```

**格式或范围错误**（`422`）：

```json
{ "detail": "采样率须为 16000 或 48000 Hz，当前: 44100 Hz" }
```

### 调用示例

```bash
curl -X POST http://localhost:8000/rt60 -F "file=@impulse.wav"
```

## 本地开发（不用 Docker）

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
