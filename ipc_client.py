"""시뮬레이터와 통신을 담당하는 모듈.

이 파일은 가능한 한 수정하지 않고, 알고리즘 변경은 `student_planner.py`
내 `PlannerSkeleton` 및 `planner_step` 구현만 손보면 됩니다.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from student_planner import handle_map_payload, planner_step  # 학생 구현 모듈


STUDENT_REPLAY_DIR = "student_replays"
AUTO_RENDER_REPLAY = os.getenv("PARKING_AUTO_RENDER_REPLAY", "1").lower() not in {
    "0",
    "false",
    "off",
    "no",
}


def resolve_sim_replay_dir() -> Optional[Path]:
    candidates: List[Path] = []
    env_path = os.getenv("PARKING_SIM_DIR")
    if env_path:
        candidates.append(Path(env_path))
    script_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            script_dir.parent / "self-parking-sim" ,
            script_dir.parent / "self-parking-sim-main",
            Path.home() / "Downloads" / "self-parking-sim-main",
        ]
    )
    for candidate in candidates:
        replay_dir = candidate / "replays"
        if replay_dir.exists():
            return replay_dir
    return None


def _slugify(text: Any) -> str:
    slug = "".join(ch.lower() if str(ch).isalnum() else "_" for ch in str(text))
    slug = slug.strip("_")
    return slug or "session"


def save_student_replay(frames: List[Dict[str, Any]], meta: Dict[str, Any]) -> Optional[str]:
    if not frames:
        return None
    try:
        os.makedirs(STUDENT_REPLAY_DIR, exist_ok=True)
    except Exception as exc:
        print(f"[algo] replay dir error: {exc}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_key = meta.get("map_key") or meta.get("map_name") or "session"
    filename = f"{timestamp}_{_slugify(map_key)}.json"
    path = os.path.join(STUDENT_REPLAY_DIR, filename)
    payload = {
        "meta": meta,
        "frames": frames,
    }
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"[algo] replay saved: {path}")
        return path
    except Exception as exc:
        print(f"[algo] replay save failed: {exc}")
        return None


def compact_map_payload(map_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only lightweight drawing/identification fields for replay PNG rendering."""

    keep_keys = (
        "key",
        "map_key",
        "name",
        "map_name",
        "seed",
        "map_seed",
        "target_idx",
        "stage_label",
        "expected_orientation",
        "extent",
        "slots",
        "occupied_idx",
        "walls_rects",
        "lines",
        "border",
    )
    return {key: map_payload[key] for key in keep_keys if key in map_payload}


def render_student_replay_png(replay_path: Optional[str]) -> Optional[str]:
    if not replay_path or not AUTO_RENDER_REPLAY:
        return None

    script_dir = Path(__file__).resolve().parent
    renderer = script_dir / "render_replay_path.py"
    if not renderer.exists():
        print("[algo] replay image skipped: render_replay_path.py not found")
        return None

    replay_abs = (script_dir / replay_path).resolve() if not Path(replay_path).is_absolute() else Path(replay_path)
    output_path = script_dir / "replay_images" / f"{replay_abs.stem}.png"
    env = os.environ.copy()
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(renderer),
                "--replay",
                str(replay_abs),
                "--out",
                str(output_path),
            ],
            cwd=str(script_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=25,
        )
    except Exception as exc:
        print(f"[algo] replay image failed: {exc}")
        return None

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        print(f"[algo] replay image failed{suffix}")
        return None

    print(f"[algo] replay image saved: {output_path}")
    return str(output_path)


def render_latest_sim_replay_png(reference_time: float) -> Optional[str]:
    replay_dir = resolve_sim_replay_dir()
    if replay_dir is None:
        return None
    recent = sorted(replay_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not recent:
        return None
    replay_path = recent[0]
    if replay_path.stat().st_mtime < reference_time - 20.0:
        return None
    return render_student_replay_png(str(replay_path))


def run_session(sock: socket.socket, peer: Tuple[str, int]) -> None:
    """시뮬레이터와의 단일 TCP 세션을 처리합니다."""

    print(f"[algo] connected to simulator at {peer}")
    buffer = b""
    frames: List[Dict[str, Any]] = []
    session_meta: Dict[str, Any] = {
        "peer": {"host": peer[0], "port": peer[1]},
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "map_key": None,
        "map_name": None,
    }

    try:
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                print("[algo] simulator closed the connection")
                break

            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue

                try:
                    packet = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    print(f"[algo] bad JSON from simulator: {exc}")
                    continue

                if isinstance(packet, dict) and "map" in packet:
                    map_payload = packet["map"]
                    handle_map_payload(map_payload)
                    print("[algo] received static map payload")
                    session_meta["map_payload"] = compact_map_payload(map_payload)
                    session_meta["map_key"] = map_payload.get("key") or map_payload.get("map_key")
                    session_meta["map_name"] = map_payload.get("name") or map_payload.get("map_name")
                    session_meta["map_extent"] = map_payload.get("extent")
                    session_meta["slots_total"] = len(map_payload.get("slots", []))
                    for key in ("map_seed", "seed", "target_idx", "stage_label", "expected_orientation"):
                        if key in map_payload:
                            session_meta[key] = map_payload.get(key)
                    if "map_seed" not in session_meta and "seed" in session_meta:
                        session_meta["map_seed"] = session_meta["seed"]
                    continue

                try:
                    for key in ("map_seed", "target_idx", "stage_label", "expected_orientation"):
                        if key in packet and key not in session_meta:
                            session_meta[key] = packet.get(key)
                    if "target_slot" in packet and "target_slot" not in session_meta:
                        session_meta["target_slot"] = packet.get("target_slot")
                    cmd = planner_step(packet)
                    payload = json.dumps(cmd, ensure_ascii=False) + "\n"
                    sock.sendall(payload.encode("utf-8"))
                    frames.append(
                        {
                            "t": packet.get("t"),
                            "obs": packet,
                            "cmd": cmd,
                        }
                    )
                except BrokenPipeError:
                    print("[algo] send failed: broken pipe")
                    return
                except Exception as exc:
                    print(f"[algo] planner/send error: {exc}")

    except (ConnectionResetError, ConnectionAbortedError) as exc:
        print(f"[algo] connection error: {exc}")
    except Exception as exc:
        print(f"[algo] unexpected error while talking to simulator: {exc}")
    finally:
        session_meta["end_time"] = datetime.now().isoformat(timespec="seconds")
        session_meta["frame_count"] = len(frames)
        replay_path = save_student_replay(frames, session_meta)
        rendered_sim = render_latest_sim_replay_png(time.time())
        if rendered_sim is None:
            render_student_replay_png(replay_path)


def run_client(host: str, port: int) -> None:
    """시뮬레이터가 열어둔 포트에 접속해 세션을 유지합니다."""

    backoff = 1.0
    while True:
        try:
            print(f"[algo] connecting to simulator at {host}:{port} ...")
            with socket.create_connection((host, port), timeout=2.0) as sock:
                sock.settimeout(0.2)
                run_session(sock, sock.getpeername())
                backoff = 1.0  # 연결이 정상 종료되면 지연을 초기화
        except KeyboardInterrupt:
            print("\n[algo] stopping by keyboard interrupt")
            break
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            print(f"[algo] connect failed ({exc}); retrying in {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff + 0.5, 5.0)
            continue

        # 시뮬레이터가 연결을 닫은 경우 짧게 대기 후 재시도
        print("[algo] lost connection - waiting 1.0s before retry")
        time.sleep(1.0)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55556)
    options = parser.parse_args(argv)

    # Ctrl+C 입력 시 즉시 종료
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    run_client(options.host, options.port)


if __name__ == "__main__":
    main()
