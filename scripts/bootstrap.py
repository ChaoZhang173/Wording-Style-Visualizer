"""Create a local environment and open Token Atlas, with no shell interpolation."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".venv"
APP_ID = {"application": "wording-style-visualizer", "protocol": 1}


def is_running_app(port: int) -> bool:
    """Only reuse a healthy server that identifies itself as this application."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.4)
    try:
        connection.request("GET", "/app/static/token-atlas.json")
        response = connection.getresponse()
        marker = response.read(1025)
        if response.status != 200 or len(marker) > 1024 or json.loads(marker) != APP_ID:
            return False
        connection.request("GET", "/_stcore/health")
        response = connection.getresponse()
        return response.status == 200 and response.read(16).strip() == b"ok"
    except (OSError, http.client.HTTPException, ValueError):
        return False
    finally:
        connection.close()


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def choose_port(preferred: int) -> tuple[int, bool]:
    for port in range(preferred, min(preferred + 20, 65535) + 1):
        if is_running_app(port):
            return port, True
        if is_port_available(port):
            return port, False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1]), False


def open_app(port: int, reused: bool) -> None:
    url = f"http://127.0.0.1:{port}"
    if reused:
        print("\n文迹已经运行，正在打开页面 / Token Atlas is already running; opening it.", flush=True)
    print(f"\n打开 / Open: {url}", flush=True)
    if reused:
        try:
            opened = webbrowser.open(url)
        except webbrowser.Error:
            opened = False
        if not opened:
            print("请在浏览器打开上方地址 / Open the address above in your browser.", flush=True)
        print("可以关闭本窗口，原来的程序会继续运行 / You may close this window; the existing app will keep running.", flush=True)
    else:
        print("保留此窗口。按 Ctrl+C 停止 / Keep this window open. Press Ctrl+C to stop.\n", flush=True)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def environment_python() -> Path:
    return ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def install_requirements(python: Path, filename: str) -> None:
    requirement = ROOT / filename
    signature = hashlib.sha256(requirement.read_bytes()).hexdigest()
    marker = ENV / f".token-atlas-{filename}.sha256"
    if marker.exists() and marker.read_text(encoding="utf-8") == signature:
        return
    print(f"\n安装组件 / Installing {filename}…", flush=True)
    print("首次安装需要联网，可能需要几分钟 / First setup needs internet and may take a few minutes.", flush=True)
    run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirement)])
    marker.write_text(signature, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Token Atlas in a local browser.")
    parser.add_argument("--light", action="store_true", help="Install only the interface and precomputed demo support.")
    parser.add_argument("--check", action="store_true", help="Prepare dependencies without opening the app.")
    parser.add_argument("--pacmap", action="store_true", help="Also install the optional PaCMAP algorithm.")
    parser.add_argument("--port", type=int, default=8501, help="Preferred local port; automatically switches if busy (default: 8501).")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        print("请使用 Python 3.10–3.12 / Please use Python 3.10–3.12.", file=sys.stderr)
        return 1
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    stage = "setup"
    try:
        if not args.check and not args.pacmap and is_running_app(args.port):
            open_app(args.port, reused=True)
            return 0
        python = environment_python()
        if not python.exists():
            print("创建独立运行环境 / Creating a local environment…", flush=True)
            run([sys.executable, "-m", "venv", str(ENV)])
        version_check = subprocess.run([
            str(python), "-c",
            "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)",
        ], cwd=ROOT, check=False)
        if version_check.returncode:
            print("项目内 .venv 的 Python 版本不兼容。请把 .venv 文件夹重命名后重新启动。", file=sys.stderr)
            print("The project's .venv uses an unsupported Python. Rename .venv, then restart.", file=sys.stderr)
            return 1
        install_requirements(python, "requirements.txt")
        if not args.light:
            install_requirements(python, "requirements-models.txt")
        if args.pacmap:
            install_requirements(python, "requirements-pacmap.txt")
        if args.check:
            print("准备完成 / Setup complete.")
            return 0
        stage = "launch"
        for attempt in range(2):
            port, reused = choose_port(args.port)
            if port != args.port:
                print(f"端口 {args.port} 已占用，自动使用 {port} / Port {args.port} is busy; using {port}.", flush=True)
            open_app(port, reused)
            if reused:
                return 0
            try:
                run([
                    str(python), "-m", "streamlit", "run", str(ROOT / "app.py"),
                    "--server.address=127.0.0.1", f"--server.port={port}",
                    "--server.enableStaticServing=true", "--browser.gatherUsageStats=false",
                ])
                return 0
            except subprocess.CalledProcessError:
                if attempt == 0 and not is_port_available(port):
                    print("端口刚被占用，正在重新选择 / Port became busy; choosing another one.", flush=True)
                    continue
                raise
    except KeyboardInterrupt:
        return 0
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"\n启动未完成 / Startup did not finish: {error}", file=sys.stderr)
        if stage == "setup":
            print("运行环境准备失败，请查看上方安装错误；下载失败时再检查网络。排错说明见 README.md。", file=sys.stderr)
            print("Environment setup failed. Check the installation error above; check your connection if a download failed. See README.en.md.", file=sys.stderr)
        else:
            print("程序启动失败，请查看上方具体错误。端口占用会自动处理。排错说明见 README.md。", file=sys.stderr)
            print("App startup failed. See the specific error above; occupied ports are handled automatically. See README.en.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
