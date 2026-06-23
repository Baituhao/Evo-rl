#!/usr/bin/env python

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import savgol_filter
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames


def _smooth_1d(arr: np.ndarray, window: int = 1) -> np.ndarray:
    """Savitzky-Golay smoothing for 1-D array. Use window=1 to disable."""
    if window <= 1 or arr.shape[0] < 5:
        return arr.copy()

    w = min(int(window), int(arr.shape[0]))
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return arr.copy()

    return savgol_filter(arr, window_length=w, polyorder=3).astype(arr.dtype)


def _select_video_key(camera_keys: list[str], requested_video_key: str | None) -> str:
    if len(camera_keys) == 0:
        raise ValueError("No camera key found in dataset.")
    if requested_video_key is not None:
        if requested_video_key not in camera_keys:
            raise ValueError(
                f"Unknown video_key '{requested_video_key}'. Available camera keys: {camera_keys}"
            )
        return requested_video_key

    for key in camera_keys:
        lower = key.lower()
        if ".front" in lower or "_front" in lower or lower.endswith("front"):
            return key
    return camera_keys[0]


def _select_video_keys(
    camera_keys: list[str],
    requested_video_keys: str | None,
    requested_video_key: str | None,
) -> list[str]:
    if len(camera_keys) == 0:
        raise ValueError("No camera key found in dataset.")

    if requested_video_keys is None or not requested_video_keys.strip():
        return [_select_video_key(camera_keys, requested_video_key)]

    resolved: list[str] = []
    for raw_key in requested_video_keys.split(","):
        key = raw_key.strip()
        if not key:
            continue
        if key not in camera_keys:
            raise ValueError(f"Unknown video key '{key}'. Available camera keys: {camera_keys}")
        if key not in resolved:
            resolved.append(key)

    if not resolved:
        raise ValueError("'viz.video_keys' is empty after parsing. Provide comma-separated camera keys.")

    return resolved


def _parse_episodes_arg(episodes_arg: str | None, total_episodes: int) -> list[int]:
    # `draccus`/json may serialize `null` as None or the string "None".
    if episodes_arg is None:
        return list(range(total_episodes))
    value = str(episodes_arg).strip().lower()
    if value in {"none", "null", ""}:
        return list(range(total_episodes))
    if value == "all":
        return list(range(total_episodes))

    # Strip surrounding brackets if present (e.g. "[790, 791]" -> "790, 791")
    episodes_arg = episodes_arg.strip().lstrip("[").rstrip("]")

    parsed: set[int] = set()
    for token in episodes_arg.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid episode range '{part}'.")
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(part))

    episodes = sorted(parsed)
    for ep in episodes:
        if ep < 0 or ep >= total_episodes:
            raise ValueError(f"Episode index out of range: {ep}, total_episodes={total_episodes}.")
    return episodes


def _to_1d_float(values: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr.reshape(-1)


def _to_1d_int(values: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr.reshape(-1)


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _curve_points(
    values: np.ndarray,
    current_step: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    n = len(values)
    if n == 0:
        return []

    denom_x = max(1, n - 1)
    denom_y = max(1e-6, y_max - y_min)

    points = []
    last_step = min(current_step, n - 1)
    for i in range(last_step + 1):
        x = int(round(x0 + width * (i / denom_x)))
        y_norm = np.clip((float(values[i]) - y_min) / denom_y, 0.0, 1.0)
        y = int(round(y0 + (1.0 - y_norm) * height))
        points.append((x, y))
    return points


def _series_styles() -> dict[str, tuple[int, int, int, int]]:
    return {
        "target": (80, 220, 120, 220),
        "reward": (255, 170, 60, 220),
        "value": (100, 200, 255, 220),
        "advantage": (255, 110, 110, 220),
    }


def _draw_series_legend(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    current_step: int,
    series_values: dict[str, np.ndarray],
    chart_x0: int,
    chart_y0: int,
) -> None:
    styles = _series_styles()
    font = _load_font(max(12, height // 34))
    labels = ["target", "reward", "value", "advantage"]
    legend_rows: list[tuple[str, tuple[int, int, int, int], str]] = []
    for label in labels:
        values = series_values.get(label)
        if values is None or values.shape[0] == 0:
            continue
        idx = min(current_step, values.shape[0] - 1)
        legend_rows.append((label, styles[label], f"{label}: {float(values[idx]):.4f}"))

    if not legend_rows:
        return

    row_gap = max(4, height // 160)
    line_len = max(18, width // 36)
    swatch_gap = max(8, width // 160)
    padding_x = max(8, width // 120)
    padding_y = max(6, height // 140)

    max_text_w = 0
    row_heights: list[int] = []
    for _, _, text in legend_rows:
        bbox = draw.textbbox((0, 0), text, font=font)
        max_text_w = max(max_text_w, bbox[2] - bbox[0])
        row_heights.append(bbox[3] - bbox[1])

    box_w = padding_x * 2 + line_len + swatch_gap + max_text_w
    content_h = sum(row_heights) + row_gap * max(0, len(row_heights) - 1)
    box_h = padding_y * 2 + content_h
    box_x0 = chart_x0 + max(4, width // 240)
    box_y0 = chart_y0 + max(4, height // 240)
    box_x1 = min(width - 4, box_x0 + box_w)
    box_y1 = min(height - 4, box_y0 + box_h)
    draw.rounded_rectangle((box_x0, box_y0, box_x1, box_y1), radius=8, fill=(0, 0, 0, 150))

    y = box_y0 + padding_y
    for (label, color, text), row_h in zip(legend_rows, row_heights, strict=True):
        cy = y + row_h // 2
        draw.line(
            [(box_x0 + padding_x, cy), (box_x0 + padding_x + line_len, cy)],
            fill=color,
            width=max(2, width // 640),
        )
        dot_r = max(2, width // 320)
        dot_x = box_x0 + padding_x + line_len
        draw.ellipse((dot_x - dot_r, cy - dot_r, dot_x + dot_r, cy + dot_r), fill=color)
        draw.text(
            (box_x0 + padding_x + line_len + swatch_gap, y),
            text,
            fill=color,
            font=font,
        )
        y += row_h + row_gap


def _draw_overlay(
    frame: Image.Image,
    values: np.ndarray,
    current_step: int,
    advantage_t: float,
    acp_t: int,
    highlight_current_point: bool,
    y_min: float,
    y_max: float,
    advantages: np.ndarray | None = None,
    indicators: np.ndarray | None = None,
    draw_advantage_threshold_marker: bool = False,
    advantage_threshold: float | None = None,
    advantage_percent: float | None = None,
    targets: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    draw_indicator_persistent_marker: bool = False,
    draw_indicator_status_text: bool = False,
) -> Image.Image:
    _ = (advantage_t, highlight_current_point)

    rgba = frame.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = rgba.size
    n = len(values)
    if n == 0:
        return Image.alpha_composite(rgba, overlay).convert("RGB")

    margin_x = max(12, width // 96)
    margin_top = max(8, height // 72)
    margin_bottom = max(24, height // 36)
    chart_x0 = margin_x
    chart_x1 = width - margin_x
    chart_y0 = margin_top
    chart_y1 = height - margin_bottom
    chart_w = max(1, chart_x1 - chart_x0)
    chart_h = max(1, chart_y1 - chart_y0)

    denom_y = max(1e-6, y_max - y_min)
    padding = denom_y * 0.05
    val_min = y_min - padding
    val_max = y_max + padding

    last = min(current_step, n - 1)
    denom_x = max(1, n - 1)

    # Light grid + current-frame vertical cursor.
    grid_color = (255, 255, 255, 30)
    for frac in (0.25, 0.5, 0.75):
        gy = int(round(chart_y0 + frac * chart_h))
        draw.line([(chart_x0, gy), (chart_x1, gy)], fill=grid_color, width=1)
    cx = int(round(chart_x0 + chart_w * (last / denom_x)))
    draw.line([(cx, chart_y0), (cx, chart_y1)], fill=(255, 255, 255, 40), width=1)

    curve_width = max(2, width // 600)
    threshold_marker_color = (214, 132, 132, 72)
    styles = _series_styles()
    series_values: dict[str, np.ndarray] = {"value": values}
    if targets is not None:
        series_values["target"] = targets
    if rewards is not None:
        series_values["reward"] = rewards
    if advantages is not None:
        series_values["advantage"] = advantages

    series_points: dict[str, list[tuple[int, int]]] = {}
    for label in ["target", "reward", "value", "advantage"]:
        series = series_values.get(label)
        if series is None or series.shape[0] == 0:
            continue
        points = _curve_points(
            values=series,
            current_step=last,
            x0=chart_x0,
            y0=chart_y0,
            width=chart_w,
            height=chart_h,
            y_min=val_min,
            y_max=val_max,
        )
        series_points[label] = points
        if len(points) >= 2:
            draw.line(points, fill=styles[label], width=curve_width)

    if (
        draw_advantage_threshold_marker
        and advantage_threshold is not None
        and advantages is not None
        and len(series_points.get("advantage", [])) >= 1
    ):
        marker_segments_by_x: dict[int, int] = {}
        advantage_points = series_points["advantage"]
        last_advantage_idx = min(last, len(advantages) - 1, len(advantage_points) - 1)
        for step_idx in range(last_advantage_idx + 1):
            if float(advantages[step_idx]) <= float(advantage_threshold):
                continue
            px, py = advantage_points[step_idx]
            if px not in marker_segments_by_x or py > marker_segments_by_x[px]:
                marker_segments_by_x[px] = py

        marker_width = max(1, curve_width - 1)
        for px, py in marker_segments_by_x.items():
            draw.line([(px, py), (px, height - 1)], fill=threshold_marker_color, width=marker_width)

    if draw_indicator_persistent_marker and indicators is not None and len(series_points.get("advantage", [])) >= 1:
        marker_segments_by_x: dict[int, int] = {}
        advantage_points = series_points["advantage"]
        last_advantage_idx = min(last, len(indicators) - 1, len(advantage_points) - 1)
        for step_idx in range(last_advantage_idx + 1):
            if int(indicators[step_idx]) != 1:
                continue
            px, py = advantage_points[step_idx]
            if px not in marker_segments_by_x or py > marker_segments_by_x[px]:
                marker_segments_by_x[px] = py

        marker_width = max(1, curve_width - 1)
        indicator_marker_color = (214, 60, 60, 72)
        for px, py in marker_segments_by_x.items():
            draw.line([(px, py), (px, height - 1)], fill=indicator_marker_color, width=marker_width)

    for label, points in series_points.items():
        if len(points) == 0:
            continue
        px, py = points[-1]
        color = styles[label]
        radius = max(3, curve_width + (1 if label == "value" else 0))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
        draw.ellipse(
            (px - radius - 1, py - radius - 1, px + radius + 1, py + radius + 1),
            outline=(255, 255, 255, 180 if label == "value" else 110),
            width=1,
        )

    if len(series_points) == 1 and len(series_points.get("value", [])) >= 1:
        px, py = series_points["value"][-1]
        value_font = _load_font(max(16, height // 22))
        value_text = f"{float(values[last]):.4f}"
        value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
        text_w = value_bbox[2] - value_bbox[0]
        text_h = value_bbox[3] - value_bbox[1]
        tx = min(px + 10, width - text_w - 10)
        ty = max(py - text_h - 12, chart_y0 + 2)
        draw.rectangle((tx - 4, ty - 2, tx + text_w + 4, ty + text_h + 2), fill=(0, 0, 0, 160))
        draw.text((tx, ty), value_text, fill=styles["value"], font=value_font)

    _draw_series_legend(
        draw=draw,
        width=width,
        height=height,
        current_step=last,
        series_values=series_values,
        chart_x0=chart_x0,
        chart_y0=chart_y0,
    )

    small_font = _load_font(max(13, height // 30))
    frame_text = f"frame {last}/{n - 1}"
    frame_bbox = draw.textbbox((0, 0), frame_text, font=small_font)
    frame_w = frame_bbox[2] - frame_bbox[0]
    frame_h = frame_bbox[3] - frame_bbox[1]
    fx = width - frame_w - margin_x - 4
    fy = height - frame_h - 6
    draw.rectangle((fx - 3, fy - 1, fx + frame_w + 3, fy + frame_h + 1), fill=(0, 0, 0, 140))
    draw.text((fx, fy), frame_text, fill=(200, 200, 200, 220), font=small_font)

    if draw_advantage_threshold_marker and advantage_percent is not None:
        adv_text = f"adv_percent={advantage_percent:.2f}%"
        adv_bbox = draw.textbbox((0, 0), adv_text, font=small_font)
        adv_w = adv_bbox[2] - adv_bbox[0]
        adv_h = adv_bbox[3] - adv_bbox[1]
        ax = width - adv_w - margin_x - 4
        ay = fy - adv_h - 8
        draw.rectangle((ax - 3, ay - 1, ax + adv_w + 3, ay + adv_h + 1), fill=(0, 0, 0, 140))
        draw.text((ax, ay), adv_text, fill=(255, 120, 120, 235), font=small_font)

    if draw_indicator_status_text:
        indicator_text = "positive" if int(acp_t) == 1 else "negative"
        indicator_fill = (220, 30, 30, 235) if int(acp_t) == 1 else (0, 0, 0, 235)
        indicator_bbox = draw.textbbox((0, 0), indicator_text, font=small_font)
        indicator_w = indicator_bbox[2] - indicator_bbox[0]
        indicator_h = indicator_bbox[3] - indicator_bbox[1]
        ix = width - indicator_w - margin_x - 4
        iy = chart_y0 + 6
        draw.rectangle((ix - 4, iy - 2, ix + indicator_w + 4, iy + indicator_h + 2), fill=(255, 255, 255, 170))
        draw.text((ix, iy), indicator_text, fill=indicator_fill, font=small_font)

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _build_output_video_path(output_dir: Path, repo_id: str, video_key: str, episode_index: int) -> Path:
    repo_tag = repo_id.replace("/", "_")
    key_tag = video_key.replace(".", "_")
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{key_tag}.mp4"


def _build_output_video_path_multiview(
    output_dir: Path,
    repo_id: str,
    video_keys: list[str],
    episode_index: int,
) -> Path:
    repo_tag = repo_id.replace("/", "_")
    keys_tag = "__".join(key.replace(".", "_") for key in video_keys)
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{keys_tag}_multiview.mp4"


# Decode at most this many frames per call to ``decode_video_frames``. A full
# 1080p episode can be several thousand frames; decoding them all at once
# produces a single float32 tensor of 1920*1080*3*4 bytes/frame (~24MB/frame,
# i.e. >100GB for a 4k-frame episode) before any downsample can run, which OOMs.
# Decoding in bounded chunks caps peak RAM to roughly chunk_size full frames.
_DECODE_CHUNK_FRAMES = 256


def _frames_tensor_to_pil(
    frames: "torch.Tensor",
    max_frame_size: int | None,
) -> list[Image.Image]:
    # Downsample while still a tensor, before the .numpy() copy and the float
    # *255 conversion below. Source frames may be 1080p; shrinking here caps
    # every downstream copy.
    if max_frame_size is not None and frames.ndim == 4:
        h, w = int(frames.shape[-2]), int(frames.shape[-1])
        longest = max(h, w)
        if longest > max_frame_size:
            scale = max_frame_size / longest
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            frames = torch.nn.functional.interpolate(
                frames, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
    np_frames = frames.detach().cpu().numpy()
    if np_frames.ndim != 4:
        raise ValueError(f"Unexpected decoded frame tensor shape: {np_frames.shape}")
    if np_frames.shape[1] in (1, 3):
        np_frames = np.transpose(np_frames, (0, 2, 3, 1))
    if np_frames.dtype != np.uint8:
        if np.issubdtype(np_frames.dtype, np.floating):
            max_val = float(np.max(np_frames)) if np_frames.size > 0 else 1.0
            if max_val <= 1.0 + 1e-6:
                np_frames = np.clip(np_frames, 0.0, 1.0) * 255.0
            else:
                np_frames = np.clip(np_frames, 0.0, 255.0)
        else:
            np_frames = np.clip(np_frames, 0, 255)
        np_frames = np_frames.astype(np.uint8)
    return [Image.fromarray(np_frames[i]) for i in range(np_frames.shape[0])]


def _decode_frames_at_timestamps(
    video_file: Path,
    timestamps_s: np.ndarray,
    tolerance_s: float,
    backend: str | None,
    max_frame_size: int | None = None,
) -> list[Image.Image]:
    if timestamps_s.size == 0:
        return []
    # Decode in bounded chunks so peak memory stays proportional to the chunk
    # size, not the whole episode. Each chunk is downsampled and converted to
    # uint8 PIL images before the next chunk is decoded, so the large float32
    # tensor is released between chunks.
    ts_list = timestamps_s.tolist()
    pil_frames: list[Image.Image] = []
    for start in range(0, len(ts_list), _DECODE_CHUNK_FRAMES):
        chunk = ts_list[start : start + _DECODE_CHUNK_FRAMES]
        frames = decode_video_frames(
            video_path=video_file,
            timestamps=chunk,
            tolerance_s=tolerance_s,
            backend=backend,
        )
        pil_frames.extend(_frames_tensor_to_pil(frames, max_frame_size))
        del frames
    return pil_frames


def _is_image_dtype(dataset: LeRobotDataset, cam_key: str) -> bool:
    """Return True if cam_key is stored as image (PNG bytes in parquet), not as a video file."""
    features = dataset.meta.features
    return features.get(cam_key, {}).get("dtype") == "image"


def _load_image_frames_from_parquet(
    raw_dataset,
    cam_key: str,
    ep_positions: np.ndarray,
) -> list[Image.Image]:
    """Load PIL frames for a given episode directly from parquet image columns."""
    import io

    col_values = raw_dataset[cam_key]
    frames: list[Image.Image] = []
    for idx in ep_positions:
        entry = col_values[int(idx)]
        # `datasets` may either return raw bytes/dicts (parquet-encoded) or already-decoded PIL images
        # depending on the dataset feature configuration and formatting.
        if isinstance(entry, Image.Image):
            frames.append(entry.convert("RGB"))
        elif isinstance(entry, dict):
            img_bytes = entry.get("bytes")
            if img_bytes is None:
                raise ValueError(f"Missing 'bytes' in dict image entry for key={cam_key}")
        elif isinstance(entry, bytes):
            img_bytes = entry
        else:
            raise TypeError(f"Unexpected image column type: {type(entry)}")
        if not isinstance(entry, Image.Image):
            frames.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    return frames


def _get_episode_video_time_bounds(
    dataset: LeRobotDataset,
    episode_index: int,
    video_key: str,
) -> tuple[float, float | None]:
    episodes = getattr(dataset.meta, "episodes", None)
    if episodes is None:
        return 0.0, None
    episodes_ds = episodes.with_format(None)
    if "episode_index" not in episodes_ds.column_names:
        return 0.0, None

    episode_indices = np.asarray(episodes_ds["episode_index"], dtype=np.int64).reshape(-1)
    matched = np.flatnonzero(episode_indices == episode_index)
    if matched.size == 0:
        return 0.0, None
    row = int(matched[0])

    from_col = f"videos/{video_key}/from_timestamp"
    to_col = f"videos/{video_key}/to_timestamp"
    from_ts = 0.0
    to_ts: float | None = None
    if from_col in episodes_ds.column_names:
        from_ts = float(episodes_ds[from_col][row])
    if to_col in episodes_ds.column_names:
        to_ts = float(episodes_ds[to_col][row])
    return from_ts, to_ts


def _get_video_encode_options(vcodec: str) -> tuple[dict[str, str], str]:
    if vcodec == "libsvtav1":
        return {"g": "2", "crf": "30", "preset": "12"}, "yuv420p"
    if vcodec == "h264_nvenc":
        return {
            "preset": "p4",
            "rc": "vbr",
            "cq": "28",
            "b": "0",
            "g": "60",
        }, "yuv420p"
    return {"g": "2", "crf": "30"}, "yuv420p"


def _get_episode_value_bounds(*series_arrays: np.ndarray | None) -> tuple[float, float]:
    valid_arrays = []
    for arr in series_arrays:
        if arr is None or arr.shape[0] == 0:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.shape[0] > 0:
            valid_arrays.append(finite)

    if not valid_arrays:
        raise ValueError("Cannot determine value bounds from empty series.")

    merged = np.concatenate(valid_arrays, axis=0)
    y_min = float(np.min(merged))
    y_max = float(np.max(merged))
    if abs(y_max - y_min) < 1e-6:
        pad = max(1e-3, abs(y_max) * 0.05, 0.01)
        y_min -= pad
        y_max += pad
    return y_min, y_max


def _encode_pil_to_video(
    frames: list[Image.Image],
    video_path: Path,
    fps: int,
    vcodec: str,
) -> None:
    """Encode PIL frames directly to video without writing intermediate PNG files."""
    import av as _av

    video_options, pix_fmt = _get_video_encode_options(vcodec)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with _av.open(str(video_path), "w") as output:
        out_stream = output.add_stream(vcodec, fps, options=video_options)
        out_stream.pix_fmt = pix_fmt
        out_stream.width = frames[0].width
        out_stream.height = frames[0].height
        for pil_img in frames:
            av_frame = _av.VideoFrame.from_image(pil_img.convert("RGB"))
            for packet in out_stream.encode(av_frame):
                output.mux(packet)
        for packet in out_stream.encode():
            output.mux(packet)


def _export_single_episode(
    src_video_path: Path | None,
    dst_video_path: Path,
    ep_targets: np.ndarray | None,
    ep_rewards: np.ndarray | None,
    ep_values: np.ndarray,
    ep_advantages: np.ndarray,
    ep_indicators: np.ndarray,
    episode_timestamps_s: np.ndarray,
    fps: int,
    vcodec: str,
    tolerance_s: float,
    video_backend: str | None,
    frame_storage_mode: str = "memory",
    temp_dir_root: Path | None = None,
    smooth_window: int = 1,
    preloaded_frames: list[Image.Image] | None = None,
    draw_advantage_threshold_marker: bool = False,
    advantage_threshold: float | None = None,
    draw_indicator_persistent_marker: bool = False,
    draw_indicator_status_text: bool = False,
    max_frame_size: int | None = None,
) -> Path:
    if ep_targets is not None:
        ep_targets = _smooth_1d(ep_targets, smooth_window)
    if ep_rewards is not None:
        ep_rewards = _smooth_1d(ep_rewards, smooth_window)
    ep_values = _smooth_1d(ep_values, smooth_window)
    ep_advantages = _smooth_1d(ep_advantages, smooth_window)
    advantage_percent: float | None = None
    if draw_advantage_threshold_marker and advantage_threshold is not None:
        advantage_percent = float(np.mean(ep_advantages > float(advantage_threshold)) * 100.0)
    y_min, y_max = _get_episode_value_bounds(ep_targets, ep_rewards, ep_values, ep_advantages)
    if preloaded_frames is not None:
        decoded_frames = preloaded_frames
    else:
        decoded_frames = _decode_frames_at_timestamps(
            video_file=src_video_path,
            timestamps_s=episode_timestamps_s,
            tolerance_s=tolerance_s,
            backend=video_backend,
            max_frame_size=max_frame_size,
        )
    n_frames = min(len(decoded_frames), len(ep_values))
    if n_frames == 0:
        raise ValueError(f"No decoded frames for episode: {dst_video_path.stem}")

    if frame_storage_mode == "disk":
        with tempfile.TemporaryDirectory(
            dir=str(temp_dir_root) if temp_dir_root is not None else None,
            prefix=f"{dst_video_path.stem}-frames-",
        ) as temp_dir:
            temp_path = Path(temp_dir)
            for i in range(n_frames):
                frame = decoded_frames[i]
                composed = _draw_overlay(
                    frame=frame,
                    values=ep_values,
                    current_step=i,
                    advantage_t=float(ep_advantages[i]),
                    acp_t=int(ep_indicators[i]),
                    highlight_current_point=(int(ep_indicators[i]) == 1),
                    y_min=y_min,
                    y_max=y_max,
                    advantages=ep_advantages,
                    indicators=ep_indicators,
                    draw_advantage_threshold_marker=draw_advantage_threshold_marker,
                    advantage_threshold=advantage_threshold,
                    advantage_percent=advantage_percent,
                    targets=ep_targets,
                    rewards=ep_rewards,
                    draw_indicator_persistent_marker=draw_indicator_persistent_marker,
                    draw_indicator_status_text=draw_indicator_status_text,
                )
                composed.save(temp_path / f"frame-{i:06d}.png")

            encode_video_frames(
                imgs_dir=temp_path,
                video_path=dst_video_path,
                fps=fps,
                vcodec=vcodec,
                overwrite=True,
            )
        return dst_video_path

    composed_frames: list[Image.Image] = []
    for i in range(n_frames):
        composed = _draw_overlay(
            frame=decoded_frames[i],
            values=ep_values,
            current_step=i,
            advantage_t=float(ep_advantages[i]),
            acp_t=int(ep_indicators[i]),
            highlight_current_point=(int(ep_indicators[i]) == 1),
            y_min=y_min,
            y_max=y_max,
            advantages=ep_advantages,
            indicators=ep_indicators,
            draw_advantage_threshold_marker=draw_advantage_threshold_marker,
            advantage_threshold=advantage_threshold,
            advantage_percent=advantage_percent,
            targets=ep_targets,
            rewards=ep_rewards,
            draw_indicator_persistent_marker=draw_indicator_persistent_marker,
            draw_indicator_status_text=draw_indicator_status_text,
        )
        composed_frames.append(composed)

    _encode_pil_to_video(composed_frames, dst_video_path, fps, vcodec)
    return dst_video_path


def _export_single_episode_multiview(
    src_video_paths: list[Path | None],
    camera_labels: list[str],
    dst_video_path: Path,
    ep_targets: np.ndarray | None,
    ep_rewards: np.ndarray | None,
    ep_values: np.ndarray,
    ep_advantages: np.ndarray,
    ep_indicators: np.ndarray,
    episode_timestamps_per_cam: list[np.ndarray],
    fps: int,
    vcodec: str,
    tolerance_s: float,
    video_backend: str | None,
    frame_storage_mode: str = "memory",
    temp_dir_root: Path | None = None,
    smooth_window: int = 1,
    preloaded_frames_per_cam: list[list[Image.Image]] | None = None,
    draw_advantage_threshold_marker: bool = False,
    advantage_threshold: float | None = None,
    draw_indicator_persistent_marker: bool = False,
    draw_indicator_status_text: bool = False,
    max_frame_size: int | None = None,
) -> Path:
    if ep_targets is not None:
        ep_targets = _smooth_1d(ep_targets, smooth_window)
    if ep_rewards is not None:
        ep_rewards = _smooth_1d(ep_rewards, smooth_window)
    ep_values = _smooth_1d(ep_values, smooth_window)
    ep_advantages = _smooth_1d(ep_advantages, smooth_window)
    advantage_percent: float | None = None
    if draw_advantage_threshold_marker and advantage_threshold is not None:
        advantage_percent = float(np.mean(ep_advantages > float(advantage_threshold)) * 100.0)
    y_min, y_max = _get_episode_value_bounds(ep_targets, ep_rewards, ep_values, ep_advantages)

    all_cam_frames: list[list[Image.Image]] = []
    if preloaded_frames_per_cam is not None:
        all_cam_frames = preloaded_frames_per_cam
    else:
        for src_path, ts in zip(src_video_paths, episode_timestamps_per_cam, strict=True):
            frames = _decode_frames_at_timestamps(
                video_file=src_path,
                timestamps_s=ts,
                tolerance_s=tolerance_s,
                backend=video_backend,
                max_frame_size=max_frame_size,
            )
            all_cam_frames.append(frames)

    n_frames = min(len(f) for f in all_cam_frames)
    n_frames = min(n_frames, len(ep_values))
    if n_frames == 0:
        raise ValueError(f"No decoded frames for multiview video: {dst_video_path}")

    single_w = all_cam_frames[0][0].width
    single_h = all_cam_frames[0][0].height
    n_cams = len(all_cam_frames)
    total_w = single_w * n_cams

    label_font_size = max(14, single_h // 30)
    label_font = _load_font(label_font_size)

    def _compose_frame(i: int) -> Image.Image:
        wide_frame = Image.new("RGB", (total_w, single_h))
        for cam_idx, cam_frames in enumerate(all_cam_frames):
            wide_frame.paste(cam_frames[i].resize((single_w, single_h)), (cam_idx * single_w, 0))

        label_draw = ImageDraw.Draw(wide_frame)
        for cam_idx, label in enumerate(camera_labels):
            short_label = label.split(".")[-1]
            lx = cam_idx * single_w + 8
            ly = 4
            bbox = label_draw.textbbox((lx, ly), short_label, font=label_font)
            label_draw.rectangle(
                (bbox[0] - 2, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
                fill=(0, 0, 0, 180),
            )
            label_draw.text((lx, ly), short_label, fill=(255, 255, 200), font=label_font)

        return _draw_overlay(
            frame=wide_frame,
            values=ep_values,
            current_step=i,
            advantage_t=float(ep_advantages[i]) if i < len(ep_advantages) else 0.0,
            acp_t=int(ep_indicators[i]) if i < len(ep_indicators) else 0,
            highlight_current_point=False,
            y_min=y_min,
            y_max=y_max,
            advantages=ep_advantages,
            indicators=ep_indicators,
            draw_advantage_threshold_marker=draw_advantage_threshold_marker,
            advantage_threshold=advantage_threshold,
            advantage_percent=advantage_percent,
            targets=ep_targets,
            rewards=ep_rewards,
            draw_indicator_persistent_marker=draw_indicator_persistent_marker,
            draw_indicator_status_text=draw_indicator_status_text,
        )

    if frame_storage_mode == "disk":
        with tempfile.TemporaryDirectory(
            dir=str(temp_dir_root) if temp_dir_root is not None else None,
            prefix=f"{dst_video_path.stem}-frames-",
        ) as temp_dir:
            temp_path = Path(temp_dir)
            for i in range(n_frames):
                _compose_frame(i).save(temp_path / f"frame-{i:06d}.png")

            encode_video_frames(
                imgs_dir=temp_path,
                video_path=dst_video_path,
                fps=fps,
                vcodec=vcodec,
                overwrite=True,
            )
        return dst_video_path

    composed_frames: list[Image.Image] = []
    for i in range(n_frames):
        composed_frames.append(_compose_frame(i))

    _encode_pil_to_video(composed_frames, dst_video_path, fps, vcodec)
    return dst_video_path


def _export_overlay_videos(
    dataset: LeRobotDataset,
    value_field: str,
    advantage_field: str,
    indicator_field: str,
    viz_episodes: str,
    video_key: str | None,
    video_keys: str | None,
    output_dir: Path,
    overwrite: bool,
    vcodec: str,
    frame_storage_mode: str = "memory",
    smooth_window: int = 1,
    draw_advantage_threshold_marker: bool = False,
    advantage_threshold: float | None = None,
    draw_indicator_persistent_marker: bool = False,
    draw_indicator_status_text: bool = False,
    target_field: str | None = None,
    reward_field: str | None = None,
    max_frame_size: int | None = None,
) -> list[Path]:
    selected_video_keys = _select_video_keys(
        camera_keys=list(dataset.meta.camera_keys),
        requested_video_keys=video_keys,
        requested_video_key=video_key,
    )
    multiview_mode = len(selected_video_keys) > 1

    raw_dataset = dataset.hf_dataset.with_format(None)
    column_names = set(raw_dataset.column_names)

    if value_field not in column_names:
        raise KeyError(f"Missing value field '{value_field}' in dataset.")

    if draw_advantage_threshold_marker and advantage_threshold is None:
        raise ValueError(
            "advantage_threshold must be provided when draw_advantage_threshold_marker is enabled."
        )

    values_all = _to_1d_float(raw_dataset[value_field])
    if target_field is not None:
        if target_field not in column_names:
            raise KeyError(f"Missing target field '{target_field}' in dataset.")
        targets_all: np.ndarray | None = _to_1d_float(raw_dataset[target_field])
    else:
        targets_all = None
    if reward_field is not None:
        if reward_field not in column_names:
            raise KeyError(f"Missing reward field '{reward_field}' in dataset.")
        rewards_all: np.ndarray | None = _to_1d_float(raw_dataset[reward_field])
    else:
        rewards_all = None
    if advantage_field in column_names:
        advantages_all = _to_1d_float(raw_dataset[advantage_field])
    else:
        if draw_advantage_threshold_marker:
            raise KeyError(f"Missing advantage field '{advantage_field}' in dataset.")
        advantages_all = np.zeros_like(values_all, dtype=np.float32)

    if indicator_field in column_names:
        indicators_all = _to_1d_int(raw_dataset[indicator_field])
    else:
        indicators_all = np.zeros(values_all.shape[0], dtype=np.int64)

    episode_indices_all = np.asarray(raw_dataset["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices_all = np.asarray(raw_dataset["frame_index"], dtype=np.int64).reshape(-1)
    if "timestamp" in column_names:
        timestamps_all = np.asarray(raw_dataset["timestamp"], dtype=np.float64).reshape(-1)
    else:
        timestamps_all = frame_indices_all.astype(np.float64) / float(dataset.fps)

    if dataset.episodes is not None:
        available_episodes = sorted(dataset.episodes)
    else:
        available_episodes = list(range(dataset.meta.total_episodes))

    if viz_episodes.strip().lower() == "all":
        episodes = available_episodes
    else:
        requested = _parse_episodes_arg(viz_episodes, dataset.meta.total_episodes)
        episodes = [ep for ep in requested if ep in set(available_episodes)]

    output_dir.mkdir(parents=True, exist_ok=True)

    fps = int(dataset.fps)
    tolerance_s = float(getattr(dataset, "tolerance_s", 1e-4))
    video_backend = getattr(dataset, "video_backend", None)
    written_paths: list[Path] = []

    if multiview_mode:
        tasks = []
        for ep in episodes:
            ep_positions = np.flatnonzero(episode_indices_all == ep)
            if ep_positions.shape[0] == 0:
                continue

            ep_frame_indices = frame_indices_all[ep_positions]
            if bool(np.any(np.diff(ep_frame_indices) < 0)):
                ep_positions = ep_positions[np.argsort(ep_frame_indices, kind="stable")]

            ep_values = values_all[ep_positions]
            if ep_values.shape[0] == 0:
                continue

            ep_timestamps = timestamps_all[ep_positions]
            src_paths: list[Path | None] = []
            ts_per_cam: list[np.ndarray] = []
            preloaded_per_cam: list[list[Image.Image] | None] = []
            for cam_key in selected_video_keys:
                if _is_image_dtype(dataset, cam_key):
                    src_paths.append(None)
                    ts_per_cam.append(ep_timestamps)
                    preloaded_per_cam.append(
                        _load_image_frames_from_parquet(raw_dataset, cam_key, ep_positions)
                    )
                else:
                    src_path = Path(dataset.root) / dataset.meta.get_video_file_path(ep, cam_key)
                    src_paths.append(src_path)
                    from_ts, to_ts = _get_episode_video_time_bounds(dataset, ep, cam_key)
                    cam_ts = from_ts + ep_timestamps
                    if to_ts is not None:
                        cam_ts = np.minimum(cam_ts, to_ts)
                    ts_per_cam.append(cam_ts)
                    preloaded_per_cam.append(None)

            has_any_preloaded = any(f is not None for f in preloaded_per_cam)
            resolved_preloaded: list[list[Image.Image]] | None = None
            if has_any_preloaded:
                resolved_preloaded = []
                for frames, src_path, ts in zip(preloaded_per_cam, src_paths, ts_per_cam):
                    if frames is not None:
                        resolved_preloaded.append(frames)
                    else:
                        resolved_preloaded.append(
                            _decode_frames_at_timestamps(
                                src_path, ts, tolerance_s, video_backend, max_frame_size=max_frame_size
                            )
                        )

            dst_path = _build_output_video_path_multiview(
                output_dir=output_dir,
                repo_id=dataset.repo_id,
                video_keys=selected_video_keys,
                episode_index=ep,
            )
            if dst_path.exists() and not overwrite:
                continue

            tasks.append(
                (
                    src_paths,
                    dst_path,
                    None if targets_all is None else targets_all[ep_positions],
                    None if rewards_all is None else rewards_all[ep_positions],
                    ep_values,
                    advantages_all[ep_positions],
                    indicators_all[ep_positions],
                    ts_per_cam,
                    resolved_preloaded,
                )
            )

        desc_keys = ",".join(key.split(".")[-1] for key in selected_video_keys)
        for srcs, dst, tgts, rwds, vals, advs, inds, ts_per_cam, preloaded in tqdm(
            tasks,
            total=len(tasks),
            desc=f"Export overlay multiview [{desc_keys}]",
            leave=False,
        ):
            written_paths.append(
                _export_single_episode_multiview(
                    src_video_paths=srcs,
                    camera_labels=selected_video_keys,
                    dst_video_path=dst,
                    ep_targets=tgts,
                    ep_rewards=rwds,
                    ep_values=vals,
                    ep_advantages=advs,
                    ep_indicators=inds,
                    episode_timestamps_per_cam=ts_per_cam,
                    fps=fps,
                    vcodec=vcodec,
                    tolerance_s=tolerance_s,
                    video_backend=video_backend,
                    frame_storage_mode=frame_storage_mode,
                    temp_dir_root=output_dir,
                    smooth_window=smooth_window,
                    preloaded_frames_per_cam=preloaded,
                    draw_advantage_threshold_marker=draw_advantage_threshold_marker,
                    advantage_threshold=advantage_threshold,
                    draw_indicator_persistent_marker=draw_indicator_persistent_marker,
                    draw_indicator_status_text=draw_indicator_status_text,
                    max_frame_size=max_frame_size,
                )
            )
    else:
        selected_video_key = selected_video_keys[0]
        use_image_dtype = _is_image_dtype(dataset, selected_video_key)
        tasks = []
        for ep in episodes:
            ep_positions = np.flatnonzero(episode_indices_all == ep)
            if ep_positions.shape[0] > 1:
                ep_frame_indices = frame_indices_all[ep_positions]
                if bool(np.any(np.diff(ep_frame_indices) < 0)):
                    ep_positions = ep_positions[np.argsort(ep_frame_indices, kind="stable")]

            ep_values = values_all[ep_positions]
            if ep_values.shape[0] == 0:
                continue

            ep_timestamps = timestamps_all[ep_positions]

            dst_path = _build_output_video_path(output_dir, dataset.repo_id, selected_video_key, ep)
            if dst_path.exists() and not overwrite:
                continue

            if use_image_dtype:
                preloaded_frames = _load_image_frames_from_parquet(raw_dataset, selected_video_key, ep_positions)
                tasks.append(
                    (
                        None,
                        dst_path,
                        None if targets_all is None else targets_all[ep_positions],
                        None if rewards_all is None else rewards_all[ep_positions],
                        ep_values,
                        advantages_all[ep_positions],
                        indicators_all[ep_positions],
                        ep_timestamps,
                        preloaded_frames,
                    )
                )
            else:
                from_ts, to_ts = _get_episode_video_time_bounds(dataset, ep, selected_video_key)
                ep_video_timestamps = from_ts + ep_timestamps
                if to_ts is not None:
                    ep_video_timestamps = np.minimum(ep_video_timestamps, to_ts)
                src_path = Path(dataset.root) / dataset.meta.get_video_file_path(ep, selected_video_key)
                tasks.append(
                    (
                        src_path,
                        dst_path,
                        None if targets_all is None else targets_all[ep_positions],
                        None if rewards_all is None else rewards_all[ep_positions],
                        ep_values,
                        advantages_all[ep_positions],
                        indicators_all[ep_positions],
                        ep_video_timestamps,
                        None,
                    )
                )

        for src, dst, tgts, rwds, vals, advs, inds, ts, preloaded in tqdm(
            tasks,
            total=len(tasks),
            desc=f"Export overlay [{selected_video_key}]",
            leave=False,
        ):
            written_paths.append(
                _export_single_episode(
                    src,
                    dst,
                    tgts,
                    rwds,
                    vals,
                    advs,
                    inds,
                    ts,
                    fps,
                    vcodec,
                    tolerance_s,
                    video_backend,
                    frame_storage_mode,
                    output_dir,
                    smooth_window=smooth_window,
                    preloaded_frames=preloaded,
                    draw_advantage_threshold_marker=draw_advantage_threshold_marker,
                    advantage_threshold=advantage_threshold,
                    draw_indicator_persistent_marker=draw_indicator_persistent_marker,
                    draw_indicator_status_text=draw_indicator_status_text,
                    max_frame_size=max_frame_size,
                )
            )

    return written_paths
