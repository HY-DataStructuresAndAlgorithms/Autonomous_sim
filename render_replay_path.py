"""Render a saved parking replay trajectory to a PNG image.

Usage:
    python render_replay_path.py
    python render_replay_path.py --replay student_replays/20260612_session.json
    python render_replay_path.py --out path_result.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


WIDTH = 1400
HEIGHT = 850
PADDING = 60
SCRIPT_DIR = Path(__file__).resolve().parent


def load_replay(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    meta = data.get("meta", {})
    frames = data.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Replay has no frames: {path}")
    return meta if isinstance(meta, dict) else {}, frames


def latest_replay(replay_dir: Path) -> Path:
    replay_dir = replay_dir.resolve()
    files = sorted(replay_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No replay JSON files found in {replay_dir}")
    return files[0]


def resolve_input_path(path: Path | None, default_dir: Path) -> Path:
    if path is None:
        return latest_replay(default_dir)
    if path.is_absolute() or path.exists():
        return path.resolve()
    script_relative = SCRIPT_DIR / path
    if script_relative.exists():
        return script_relative.resolve()
    return path.resolve()


def resolve_output_path(path: Path | None, replay_path: Path) -> Path:
    if path is None:
        return (SCRIPT_DIR / "replay_images" / f"{replay_path.stem}.png").resolve()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def frame_state(frame: dict[str, Any]) -> tuple[float, float, float] | None:
    obs = frame.get("obs", {})
    state = obs.get("state", {}) if isinstance(obs, dict) else {}
    try:
        return float(state["x"]), float(state["y"]), float(state.get("yaw", 0.0))
    except (KeyError, TypeError, ValueError):
        return None


def target_slot(frames: Iterable[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    for frame in frames:
        obs = frame.get("obs", {})
        slot = obs.get("target_slot") if isinstance(obs, dict) else None
        if isinstance(slot, list) and len(slot) == 4:
            try:
                return tuple(float(v) for v in slot)
            except (TypeError, ValueError):
                return None
    return None


def bounds(
    points: list[tuple[float, float]],
    slot: tuple[float, float, float, float] | None,
    meta: dict[str, Any],
) -> tuple[float, float, float, float]:
    extent = meta.get("map_extent")
    if isinstance(extent, list) and len(extent) == 4:
        try:
            return tuple(float(v) for v in extent)
        except (TypeError, ValueError):
            pass

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if slot is not None:
        xs.extend([slot[0], slot[1]])
        ys.extend([slot[2], slot[3]])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = max(3.0, 0.08 * max(xmax - xmin, ymax - ymin, 1.0))
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def result_label(meta: dict[str, Any], frames: list[dict[str, Any]]) -> str:
    result = meta.get("result") or meta.get("why") or meta.get("fail_reason")
    if result:
        return str(result)
    if frames:
        return "session ended"
    return "unknown"


def fmt_number(value: Any, suffix: str = "", digits: int = 1) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}{suffix}"
    return "N/A"


def final_speed(frames: list[dict[str, Any]]) -> float | None:
    for frame in reversed(frames):
        obs = frame.get("obs", {})
        state = obs.get("state", {}) if isinstance(obs, dict) else {}
        try:
            return abs(float(state["v"]))
        except (KeyError, TypeError, ValueError):
            continue
    return None


def info_lines(meta: dict[str, Any], frames: list[dict[str, Any]]) -> list[str]:
    stats = meta.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
    score = meta.get("score")
    iou = stats.get("parking_iou")
    iou_text = f"{float(iou) * 100:.1f}%" if isinstance(iou, (int, float)) else "N/A"
    return [
        f"Final result: {result_label(meta, frames)}",
        f"Final score: {fmt_number(score, ' / 100', 1)}",
        f"Map: {meta.get('map_name') or meta.get('map_key') or 'N/A'}",
        f"Stage: {meta.get('stage_label') or meta.get('stage') or 'N/A'}",
        f"Elapsed: {fmt_number(stats.get('elapsed'), 's', 1)}",
        f"Distance: {fmt_number(stats.get('distance'), 'm', 1)}",
        f"Average speed: {fmt_number(stats.get('avg_speed'), 'm/s', 2)}",
        f"Final speed: {fmt_number(final_speed(frames), 'm/s', 2)}",
        f"Parking IoU: {iou_text}",
        f"Orientation: {stats.get('parking_orientation', 'N/A')}",
        f"Required: {stats.get('expected_orientation', 'N/A')}",
        f"Frames: {len(frames)}",
    ]


def draw_info_panel(surface: Any, pygame: Any, font: Any, lines: list[str]) -> None:
    line_h = 23
    panel_w = 360
    panel_h = 28 + line_h * len(lines)
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((255, 255, 255, 232))
    pygame.draw.rect(panel, (65, 74, 92), panel.get_rect(), 2)
    title = font.render("Simulation Summary", True, (25, 30, 42))
    panel.blit(title, (14, 10))
    for idx, line in enumerate(lines):
        text = font.render(line, True, (42, 50, 65))
        panel.blit(text, (14, 36 + idx * line_h))
    surface.blit(panel, (WIDTH - panel_w - 24, HEIGHT - panel_h - 24))


def render(
    replay_path: Path,
    output_path: Path,
    meta: dict[str, Any],
    frames: list[dict[str, Any]],
) -> None:
    import os

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame

    states = [state for frame in frames if (state := frame_state(frame)) is not None]
    if not states:
        raise ValueError("Replay contains no valid vehicle states.")

    path_points = [(x, y) for x, y, _ in states]
    slot = target_slot(frames)
    xmin, xmax, ymin, ymax = bounds(path_points, slot, meta)
    world_w = max(xmax - xmin, 1e-6)
    world_h = max(ymax - ymin, 1e-6)
    scale = min((WIDTH - 2 * PADDING) / world_w, (HEIGHT - 2 * PADDING) / world_h)

    def to_screen(x: float, y: float) -> tuple[int, int]:
        sx = PADDING + int((x - xmin) * scale)
        sy = HEIGHT - PADDING - int((y - ymin) * scale)
        return sx, sy

    pygame.init()
    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill((246, 247, 249))

    font = pygame.font.Font(None, 30)
    small = pygame.font.Font(None, 24)
    panel_font = pygame.font.Font(None, 23)

    pygame.draw.rect(surface, (220, 224, 232), (PADDING, PADDING, WIDTH - 2 * PADDING, HEIGHT - 2 * PADDING), 1)

    if slot is not None:
        x0, y0 = to_screen(slot[0], slot[2])
        x1, y1 = to_screen(slot[1], slot[3])
        rect = pygame.Rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        pygame.draw.rect(surface, (190, 245, 196), rect)
        pygame.draw.rect(surface, (40, 150, 60), rect, 3)
        cx = 0.5 * (slot[0] + slot[1])
        cy = 0.5 * (slot[2] + slot[3])
        pygame.draw.circle(surface, (30, 130, 50), to_screen(cx, cy), 6)

    if len(path_points) >= 2:
        pygame.draw.lines(surface, (35, 80, 230), False, [to_screen(x, y) for x, y in path_points], 3)

    start = states[0]
    end = states[-1]
    pygame.draw.circle(surface, (30, 150, 70), to_screen(start[0], start[1]), 8)
    pygame.draw.circle(surface, (220, 70, 70), to_screen(end[0], end[1]), 8)

    heading_len = 1.5
    hx = end[0] + math.cos(end[2]) * heading_len
    hy = end[1] + math.sin(end[2]) * heading_len
    pygame.draw.line(surface, (220, 70, 70), to_screen(end[0], end[1]), to_screen(hx, hy), 3)

    title = f"Replay path: {replay_path.name}"
    subtitle = f"result={result_label(meta, frames)}  frames={len(frames)}"
    surface.blit(font.render(title, True, (25, 30, 40)), (24, 18))
    surface.blit(small.render(subtitle, True, (70, 78, 92)), (24, 48))
    surface.blit(small.render("green=start  red=end  blue=trajectory", True, (70, 78, 92)), (24, HEIGHT - 36))
    draw_info_panel(surface, pygame, panel_font, info_lines(meta, frames))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(output_path))
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the latest parking replay path to a PNG.")
    parser.add_argument("--replay", type=Path, help="Replay JSON path. Defaults to latest student_replays/*.json.")
    parser.add_argument("--out", type=Path, help="Output PNG path. Defaults to replay_images/<replay-name>.png.")
    args = parser.parse_args()

    replay_path = resolve_input_path(args.replay, SCRIPT_DIR / "student_replays")
    meta, frames = load_replay(replay_path)
    output_path = resolve_output_path(args.out, replay_path)
    render(replay_path, output_path, meta, frames)
    print(f"[render] saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
