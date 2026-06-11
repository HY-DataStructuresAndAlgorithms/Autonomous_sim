"""Render a saved parking replay trajectory to a PNG image.

Usage:
    python render_replay_path.py
    python render_replay_path.py --replay student_replays/20260612_session.json
    python render_replay_path.py --out path_result.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable


WIDTH = 1400
HEIGHT = 850
PADDING = 60
SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent / "self-parking-sim"


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


def load_sim_context(meta: dict[str, Any]) -> dict[str, Any] | None:
    sim_path = SIM_DIR / "demo_self_parking_sim.py"
    if not sim_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("parking_sim_for_replay_render", sim_path)
    if spec is None or spec.loader is None:
        return None

    old_cwd = Path.cwd()
    try:
        sys.path.insert(0, str(SIM_DIR))
        os.chdir(SIM_DIR)
        sim = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = sim
        spec.loader.exec_module(sim)
        map_key = meta.get("map_key")
        map_cfg = next(
            (cfg for cfg in sim.AVAILABLE_MAPS if cfg.get("key") == map_key),
            sim.AVAILABLE_MAPS[0],
        )
        bundle = sim.ensure_map_loaded(map_cfg, {}, seed=meta.get("map_seed"))
        stage_idx, stage_profile = sim.get_stage_profile(map_cfg)
        return {
            "assets": bundle["assets"],
            "stage_idx": stage_idx,
            "stage_profile": stage_profile,
        }
    except Exception:
        return None
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(SIM_DIR))
        except ValueError:
            pass


def load_sim_map(meta: dict[str, Any]) -> Any | None:
    context = load_sim_context(meta)
    if context is None:
        return None
    return context.get("assets")


def bounds(
    points: list[tuple[float, float]],
    slot: tuple[float, float, float, float] | None,
    meta: dict[str, Any],
    map_assets: Any | None = None,
) -> tuple[float, float, float, float]:
    if map_assets is not None:
        try:
            return tuple(float(v) for v in map_assets.extent)
        except (TypeError, ValueError):
            pass

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


def fmt_percent(value: Any, digits: int = 1) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.{digits}f}%"
    return "N/A"


def fmt_score(value: Any, weight: Any) -> str:
    if isinstance(value, (int, float)) and isinstance(weight, (int, float)):
        return f"{float(value):.1f} / {float(weight):.0f}"
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
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


def detail_metric_lines(meta: dict[str, Any], stats: dict[str, Any]) -> list[str]:
    details = meta.get("details", {})
    if not isinstance(details, dict):
        details = {}
    component_scores = details.get("component_scores", {})
    if isinstance(component_scores, dict) and component_scores:
        weights = {}
        stage_profile = details.get("stage_profile", {})
        if isinstance(stage_profile, dict):
            weights = stage_profile.get("weights", {})
        if not isinstance(weights, dict):
            weights = {}
        component_labels = {
            "time": "Time",
            "distance": "Distance",
            "speed": "Average speed",
            "steer_flip": "Steering reversals",
            "parking_iou": "Slot IoU",
            "parking_orientation": "Orientation match",
            "parking_stop": "Stopped in slot",
        }
        lines = []
        for key, value in component_scores.items():
            label = component_labels.get(str(key), str(key))
            lines.append(f"- {label}: {fmt_score(value, weights.get(key))}")
        return lines

    expected = stats.get("expected_orientation", "N/A")
    observed = stats.get("parking_orientation", "N/A")
    stop_ok = final_stop_text(stats)
    return [
        f"- Slot IoU: {fmt_percent(stats.get('parking_iou'))} / {fmt_percent(stats.get('parking_iou_threshold'))}",
        f"- Orientation: {observed} / {expected}",
        f"- Stopped in slot: {stop_ok}",
        f"- Gear switches: {stats.get('gear_switches', 'N/A')}",
    ]


def final_stop_text(stats: dict[str, Any]) -> str:
    speed = stats.get("final_speed")
    if isinstance(speed, (int, float)):
        return "yes" if abs(float(speed)) <= 0.2 else "no"
    return "N/A"


def vs_target_lines(
    stats: dict[str, Any],
    stage_profile: dict[str, Any] | None,
    world_extent: tuple[float, float, float, float] | None,
) -> list[str]:
    if stage_profile is None:
        return [
            "- Time: N/A",
            "- Distance: N/A",
            "- Average speed: N/A",
            "- Steering reversals: N/A",
        ]

    xmin, xmax, ymin, ymax = world_extent if world_extent is not None else (0.0, 0.0, 0.0, 0.0)
    diag = math.hypot(xmax - xmin, ymax - ymin)
    distance_target = diag * float(stage_profile.get("distance_factor", 1.0)) if diag > 0 else None
    steer_flips = stats.get("direction_flips", stats.get("steering_reversals"))
    return [
        f"- Time: {fmt_number(stats.get('elapsed'), 's', 1)} / {fmt_number(stage_profile.get('time_target'), 's', 1)}",
        f"- Distance: {fmt_number(stats.get('distance'), 'm', 1)} / {fmt_number(distance_target, 'm', 1)}",
        f"- Average speed: {fmt_number(stats.get('avg_speed'), 'm/s', 2)} / {fmt_number(stage_profile.get('speed_target'), 'm/s', 2)}",
        f"- Steering reversals: {steer_flips if isinstance(steer_flips, int) else 'N/A'} / {stage_profile.get('steer_flip_target', 'N/A')}",
    ]


def info_lines(
    meta: dict[str, Any],
    frames: list[dict[str, Any]],
    stage_profile: dict[str, Any] | None = None,
    world_extent: tuple[float, float, float, float] | None = None,
) -> list[str]:
    stats = meta.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
    score = meta.get("score")
    iou = stats.get("parking_iou")
    iou_text = f"{float(iou) * 100:.1f}%" if isinstance(iou, (int, float)) else "N/A"
    final_v = final_speed(frames)
    stats_with_final = dict(stats)
    stats_with_final.setdefault("final_speed", final_v)
    lines = [
        f"Final result: {result_label(meta, frames)}",
        f"Final score: {fmt_number(score, ' / 100', 1)}",
        f"Map: {meta.get('map_name') or meta.get('map_key') or 'N/A'}",
        f"Stage: {meta.get('stage_label') or meta.get('stage') or 'N/A'}",
        f"Elapsed: {fmt_number(stats.get('elapsed'), 's', 1)}",
        f"Distance: {fmt_number(stats.get('distance'), 'm', 1)}",
        f"Average speed: {fmt_number(stats.get('avg_speed'), 'm/s', 2)}",
        f"Final speed: {fmt_number(final_v, 'm/s', 2)}",
        f"Parking IoU: {iou_text}",
        f"Orientation: {stats.get('parking_orientation', 'N/A')}",
        f"Required: {stats.get('expected_orientation', 'N/A')}",
        f"Frames: {len(frames)}",
    ]
    lines.extend(["", "Detailed metrics"])
    lines.extend(detail_metric_lines(meta, stats_with_final))
    lines.extend(["", "Vs targets"])
    lines.extend(vs_target_lines(stats_with_final, stage_profile, world_extent))
    return lines


def draw_info_panel(surface: Any, pygame: Any, font: Any, lines: list[str]) -> None:
    line_h = 20
    panel_w = 430
    panel_h = 34 + line_h * len(lines)
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((255, 255, 255, 232))
    pygame.draw.rect(panel, (65, 74, 92), panel.get_rect(), 2)
    title = font.render("Simulation Summary", True, (25, 30, 42))
    panel.blit(title, (14, 10))
    y = 36
    for idx, line in enumerate(lines):
        if not line:
            y += line_h // 2
            continue
        is_heading = line in {"Detailed metrics", "Vs targets"}
        color = (20, 28, 42) if is_heading else (42, 50, 65)
        text = font.render(line, True, color)
        panel.blit(text, (14, y))
        y += line_h
    surface.blit(panel, (WIDTH - panel_w - 24, HEIGHT - panel_h - 24))


def screen_rect(
    rect: tuple[float, float, float, float],
    to_screen: Any,
    pygame: Any,
) -> Any:
    x0, y0 = to_screen(rect[0], rect[2])
    x1, y1 = to_screen(rect[1], rect[3])
    return pygame.Rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))


def draw_slot_rect(surface: Any, pygame: Any, rect: tuple[float, float, float, float], to_screen: Any, color: tuple[int, int, int], width: int = 0) -> None:
    pygame.draw.rect(surface, color, screen_rect(rect, to_screen, pygame), width)


def draw_map(
    surface: Any,
    pygame: Any,
    map_assets: Any | None,
    slot: tuple[float, float, float, float] | None,
    to_screen: Any,
) -> None:
    if map_assets is None:
        return

    try:
        for rect in map_assets.walls_rects:
            draw_slot_rect(surface, pygame, tuple(float(v) for v in rect), to_screen, (0, 0, 0), 0)
        for x1, y1, x2, y2 in map_assets.lines:
            pygame.draw.line(
                surface,
                (0, 0, 0),
                to_screen(float(x1), float(y1)),
                to_screen(float(x2), float(y2)),
                3,
            )
        for idx, rect in enumerate(map_assets.slots):
            r = tuple(float(v) for v in rect)
            if idx < len(map_assets.occupied_idx) and bool(map_assets.occupied_idx[idx]):
                draw_slot_rect(surface, pygame, r, to_screen, (0, 0, 0), 0)
        for rect in map_assets.slots:
            draw_slot_rect(surface, pygame, tuple(float(v) for v in rect), to_screen, (0, 0, 0), 2)
        if hasattr(map_assets, "border"):
            draw_slot_rect(surface, pygame, tuple(float(v) for v in map_assets.border), to_screen, (0, 0, 0), 4)
    except Exception:
        return

    if slot is not None:
        draw_slot_rect(surface, pygame, slot, to_screen, (180, 255, 180), 0)
        draw_slot_rect(surface, pygame, slot, to_screen, (50, 140, 50), 2)


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
    sim_context = load_sim_context(meta)
    map_assets = sim_context.get("assets") if sim_context else None
    stage_profile = sim_context.get("stage_profile") if sim_context else None
    if map_assets is not None:
        target_idx = meta.get("target_idx")
        try:
            if isinstance(target_idx, int) and 0 <= target_idx < len(map_assets.slots):
                slot = tuple(float(v) for v in map_assets.slots[target_idx])
        except Exception:
            pass
    xmin, xmax, ymin, ymax = bounds(path_points, slot, meta, map_assets)
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
    draw_map(surface, pygame, map_assets, slot, to_screen)

    if slot is not None and map_assets is None:
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
    draw_info_panel(surface, pygame, panel_font, info_lines(meta, frames, stage_profile, (xmin, xmax, ymin, ymax)))

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
