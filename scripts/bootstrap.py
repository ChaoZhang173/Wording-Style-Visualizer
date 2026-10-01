"""Create a local environment and open Token Atlas, with no shell interpolation."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".venv"


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
    parser.add_argument("--port", type=int, default=8501, help="Local app port (default: 8501).")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        print("请使用 Python 3.10–3.12 / Please use Python 3.10–3.12.", file=sys.stderr)
        return 1
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    try:
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
        print(f"\n打开 / Open: http://localhost:{args.port}", flush=True)
        print("保留此窗口。按 Ctrl+C 停止 / Keep this window open. Press Ctrl+C to stop.\n", flush=True)
        run([
            str(python), "-m", "streamlit", "run", str(ROOT / "app.py"),
            "--server.address=127.0.0.1", f"--server.port={args.port}",
            "--browser.gatherUsageStats=false",
        ])
    except KeyboardInterrupt:
        return 0
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"\n启动未完成 / Startup did not finish: {error}", file=sys.stderr)
        print("请检查网络并再次运行启动文件。排错说明见 README.md。", file=sys.stderr)
        print("Check your connection and try again. Troubleshooting: README.en.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
