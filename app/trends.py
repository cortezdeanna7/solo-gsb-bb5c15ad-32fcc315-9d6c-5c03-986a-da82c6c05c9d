"""滑动窗口趋势扫描：Mann-Kendall 检验 + Theil-Sen 斜率。

思路：
1. 以 ``window_size`` / ``step_size``（均为点数）在已排序、已去重的序列上
   滑动完整窗口（不足一个窗口的尾部丢弃）；
2. 每个窗口执行 Mann-Kendall 单调趋势检验：
   - S 统计量：所有后-前点对符号差之和；
   - 方差带**重复值（tie）修正**：
     var(S) = [n(n−1)(2n+5) − Σ t(t−1)(2t+5)] / 18，t 为各等值组大小；
   - z 分数带**连续性校正**（S>0 减 1、S<0 加 1、S=0 为 0）；
   - 双侧 p 值取标准正态近似 2·(1−Φ(|z|))；
3. 同一窗口以**实际时间差**（秒）计算 Theil-Sen 稳健斜率——所有点对
   (v_j−v_i)/(t_j−t_i) 的中位数，单位「值/秒」，非等距采样下依然成立；
4. 仅当 ``p_value <= alpha`` 且 ``|slope| >= min_slope`` 时，窗口被标记为
   上升 / 下降趋势（方向取斜率符号；斜率恰为 0 时取 S 的符号）；
5. 方向相同且时间范围重叠或首尾相接（后一窗口起点索引 ≤ 当前区间终点
   索引 + 1）的显著窗口合并为趋势区间，报告区间范围、成员窗口、
   最小 p 值与成员斜率中位数。

复杂度：单窗口 O(window_size²)，整体 O(窗口数 · window_size²)，
适合窗口数百点、序列万级点数以内的离线/准实时扫描。
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any

from .models import parse_timestamp

# Mann-Kendall 检验所需的最小窗口点数（n<3 时正态近似无意义）
MIN_WINDOW_SIZE = 3


def _normal_cdf(x: float) -> float:
    """标准正态分布 CDF（基于 math.erf，无需 scipy）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_kendall(values: list[float]) -> dict[str, float]:
    """Mann-Kendall 检验：返回 S、tie 修正方差、连续性校正 z 与双侧 p 值。"""
    n = len(values)
    s = 0
    for i in range(n - 1):
        xi = values[i]
        for j in range(i + 1, n):
            diff = values[j] - xi
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # 重复值（tie）修正：每个等值组 t 从方差中扣除 t(t−1)(2t+5)
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts.values() if t > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_s <= 0.0:
        # 窗口内值全部相同：S=0 且方差为 0，无趋势可言
        return {"s": s, "var_s": 0.0, "z_score": 0.0, "p_value": 1.0}

    # 连续性校正
    if s > 0:
        z = (s - 1.0) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / math.sqrt(var_s)
    else:
        z = 0.0
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return {"s": s, "var_s": var_s, "z_score": z, "p_value": min(1.0, p_value)}


def theil_sen_slope(times: list[float], values: list[float]) -> float:
    """Theil-Sen 斜率：所有点对 (v_j−v_i)/(t_j−t_i) 的中位数（值/秒）。

    分母取实际时间差，非等距采样下结果不受采样间隔影响。
    时间戳已去重排序，恒有 t_j > t_i。
    """
    slopes: list[float] = []
    n = len(values)
    for i in range(n - 1):
        vi, ti = values[i], times[i]
        for j in range(i + 1, n):
            dt = times[j] - ti
            if dt > 0:
                slopes.append((values[j] - vi) / dt)
    return median(slopes)


def scan_trends(
    timestamps: list[str],
    values: list[float],
    window_size: int = 10,
    step_size: int = 1,
    alpha: float = 0.05,
    min_slope: float = 0.0,
) -> dict[str, Any]:
    """执行滑窗趋势扫描，返回可 JSON 序列化的完整结果。

    :param timestamps: 已排序、已去重的 ISO-8601 时间戳字符串
    :param values: 与 timestamps 对齐的数值
    :param window_size: 滑窗大小（点数，≥3）
    :param step_size: 滑窗步长（点数，≥1）
    :param alpha: Mann-Kendall 显著性水平（双侧）
    :param min_slope: 趋势显著所需的最小绝对斜率（值/秒）
    """
    n = len(values)
    if window_size < MIN_WINDOW_SIZE:
        raise ValueError(f"window_size 必须 >= {MIN_WINDOW_SIZE}")
    if step_size < 1:
        raise ValueError("step_size 必须 >= 1")
    if n < window_size:
        raise ValueError(f"点数（{n}）必须不少于 window_size（{window_size}）")

    epoch = [parse_timestamp(ts).timestamp() for ts in timestamps]

    windows: list[dict[str, Any]] = []
    start = 0
    while start + window_size <= n:
        end = start + window_size
        mk = mann_kendall(values[start:end])
        slope = theil_sen_slope(epoch[start:end], values[start:end])
        significant = mk["p_value"] <= alpha and abs(slope) >= min_slope
        if significant:
            if slope > 0:
                direction = "上升"
            elif slope < 0:
                direction = "下降"
            else:  # 斜率恰为 0 但检验显著：以 S 的符号定方向
                direction = "上升" if mk["s"] > 0 else "下降"
        else:
            direction = "无显著趋势"
        windows.append(
            {
                "window_index": len(windows),
                "start_index": start,
                "end_index": end - 1,
                "start_time": timestamps[start],
                "end_time": timestamps[end - 1],
                "point_count": window_size,
                "s": mk["s"],
                "var_s": mk["var_s"],
                "z_score": mk["z_score"],
                "p_value": mk["p_value"],
                "slope": slope,
                "significant": significant,
                "direction": direction,
            }
        )
        start += step_size

    segments = _merge_significant_windows(windows, timestamps)
    return {
        "point_count": n,
        "window_size": window_size,
        "step_size": step_size,
        "alpha": alpha,
        "min_slope": min_slope,
        "window_count": len(windows),
        "significant_window_count": sum(1 for w in windows if w["significant"]),
        "trend_segment_count": len(segments),
        "windows": windows,
        "segments": segments,
    }


def _merge_significant_windows(
    windows: list[dict[str, Any]],
    timestamps: list[str],
) -> list[dict[str, Any]]:
    """把方向相同、时间范围重叠或首尾相接的显著窗口合并为趋势区间。

    「重叠或相接」按离散索引判定：后一窗口起点索引 ≤ 当前区间终点索引 + 1
    （step_size ≤ window_size 时窗口天然重叠或相接；step_size 更大时窗口间
    存在未覆盖的空档，不合并）。不显著的窗口不参与合并，但也不阻断两侧
    显著窗口按自身时间范围判断重叠。
    """
    segments: list[dict[str, Any]] = []
    cur_direction = ""
    cur_start = 0
    cur_end = -1
    cur_members: list[dict[str, Any]] = []

    def flush() -> None:
        if not cur_members:
            return
        segments.append(
            {
                "direction": cur_direction,
                "start_index": cur_start,
                "end_index": cur_end,
                "start_time": timestamps[cur_start],
                "end_time": timestamps[cur_end],
                "window_indices": [w["window_index"] for w in cur_members],
                "window_count": len(cur_members),
                "min_p_value": min(w["p_value"] for w in cur_members),
                "median_slope": median(w["slope"] for w in cur_members),
            }
        )

    for w in windows:
        if not w["significant"]:
            continue
        if cur_members and w["direction"] == cur_direction and w["start_index"] <= cur_end + 1:
            cur_end = max(cur_end, w["end_index"])
            cur_members.append(w)
        else:
            flush()
            cur_direction = w["direction"]
            cur_start, cur_end = w["start_index"], w["end_index"]
            cur_members = [w]
    flush()
    return segments
