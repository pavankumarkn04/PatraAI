"""Install missing project requirements and launch the local website."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP = ROOT / "Indian.py"
REQUIREMENTS = ROOT / "requirements.txt"
INDICTRANS2_REQUIREMENTS = ROOT / "requirements-indictrans2.txt"
INDICTRANS2_MODULES = ("torch", "transformers", "sentencepiece", "sacremoses", "indicnlp", "regex")


REQUIRED_IMPORTS = {
    "streamlit": "streamlit",
    # Import a generated module used during Streamlit startup. A package folder
    # can exist even when an interrupted install left this subpackage missing.
    "streamlit.proto.RootContainer_pb2": "streamlit",
    "requests": "requests",
    "PIL": "Pillow",
    "fitz": "PyMuPDF",
    "docx": "python-docx",
    "fontTools": "fonttools",
}


def package_issues() -> tuple[list[str], list[str]]:
    """Return missing import names and installed distributions that fail import."""
    missing: list[str] = []
    broken: list[str] = []
    broken_seen: set[str] = set()
    for module, distribution in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            root_module = module.split(".", 1)[0]
            # A package folder that exists but fails to import (for example an
            # interrupted install missing streamlit.proto) needs a reinstall;
            # a plain pip install treats it as already satisfied.
            if importlib.util.find_spec(root_module) is None:
                missing.append(module)
            else:
                if distribution not in broken_seen:
                    broken.append(distribution)
                    broken_seen.add(distribution)
        except (ImportError, OSError, RuntimeError):
            if distribution not in broken_seen:
                broken.append(distribution)
                broken_seen.add(distribution)
    return missing, broken


def available_port(preferred: int = 8501) -> int:
    for port in (preferred, 8502, 8503, 8504, 8505):
        with socket.socket() as server_socket:
            try:
                server_socket.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def wait_until_ready(port: int, process: subprocess.Popen[bytes], timeout: int = 90) -> bool:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200 and response.read().strip() == b"ok":
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.4)
    return False


def ollama_ready() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def ensure_ollama_runtime() -> None:
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        print("Ollama is not installed or is not on PATH. Install Ollama from https://ollama.com and reopen this app.", flush=True)
        return
    if ollama_ready():
        return
    print("Starting the local Ollama service...", flush=True)
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with open(os.devnull, "wb") as devnull:
        subprocess.Popen([ollama_path, "serve"], stdout=devnull, stderr=devnull,
                         creationflags=create_no_window, close_fds=True)
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if ollama_ready():
            return
        time.sleep(0.5)
    raise RuntimeError("Ollama did not start. Check that the Ollama service is installed and can run locally.")


def ensure_ollama_models() -> None:
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        return
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8")) if response.status == 200 else {}
    except Exception:
        payload = {}
    installed = {entry.get("name") for entry in payload.get("models", []) if isinstance(entry, dict)}
    # One model serves text, scanned pages and the translation fallback.
    required = [os.getenv("OLLAMA_TEXT_MODEL", "qwen3.5:4b")]
    missing = [model for model in required if model not in installed]
    if not missing:
        return
    print("Downloading required local Ollama models: " + ", ".join(missing), flush=True)
    for model in missing:
        result = subprocess.run([ollama_path, "pull", model], cwd=ROOT)
        if result.returncode:
            raise RuntimeError(f"Ollama could not download {model}. Check the model name and internet connection.")


def ensure_indictrans2() -> None:
    """Install IndicTrans2 packages and fetch the model once; never blocks startup."""
    if os.getenv("INDICTRANS2_LOCAL", "auto").strip() == "0" or os.getenv("INDIC_TRANS2_URL", "").strip():
        return
    missing = [module for module in INDICTRANS2_MODULES if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing IndicTrans2 translation packages: " + ", ".join(missing), flush=True)
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(INDICTRANS2_REQUIREMENTS)], cwd=ROOT)
        if result.returncode:
            print("IndicTrans2 packages could not be installed; translation will use the general local model.", flush=True)
            return
    load_env_file()
    try:
        from huggingface_hub import get_token, snapshot_download, try_to_load_from_cache
    except ImportError:
        return
    model = os.getenv("INDICTRANS2_MODEL", "ai4bharat/indictrans2-en-indic-dist-200M")
    if isinstance(try_to_load_from_cache(model, "model.safetensors"), str):
        return
    if not get_token():
        print("IndicTrans2: no Hugging Face token found. Add HF_TOKEN to .env to enable it (see README).", flush=True)
        return
    print(f"Downloading IndicTrans2 model {model} (one time, about 1 GB)...", flush=True)
    try:
        # The repository also holds an identical pytorch_model.bin; skip it.
        snapshot_download(model, ignore_patterns=["*.bin"])
        print("IndicTrans2 model ready.", flush=True)
    except Exception as exc:  # Gated access or network failure: report, keep starting.
        hint = "Accept the model's terms on Hugging Face for this token" if "gated" in str(exc).casefold() or "401" in str(exc) or "403" in str(exc) else "Check your internet connection"
        print(f"IndicTrans2 download failed ({type(exc).__name__}: {str(exc)[:200]}). {hint}, then start PatraAI again; translation uses the fallback model until then.", flush=True)


def load_env_file() -> None:
    """Load .env so the launcher sees HF_TOKEN the same way the app does."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#") and "=" in entry:
            key, value = (part.strip() for part in entry.split("=", 1))
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and value:
                os.environ.setdefault(key, value)


def warm_up_model() -> None:
    """Load the analysis model into memory now, so the first analysis does not
    wait for it to load from disk. Runs in the background; failures are ignored."""
    import threading

    model = os.getenv("OLLAMA_TEXT_MODEL", "qwen3.5:4b")
    # Same context size as real requests, otherwise Ollama reloads the model on first use.
    body = json.dumps({"model": model, "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
                       "options": {"num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192"))}}).encode("utf-8")

    def load() -> None:
        try:
            request = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(request, timeout=120).read()
        except (OSError, urllib.error.URLError):
            pass

    threading.Thread(target=load, daemon=True).start()


def previous_instances() -> list[int]:
    """PIDs of PatraAI servers or launchers already started from this folder."""
    if os.name != "nt":
        return []
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
        "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
    )
    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    targets = (str(APP).casefold(), str(ROOT / "run_app.py").casefold())
    pids = []
    for line in output.splitlines():
        pid, _, command = line.partition("\t")
        command = command.replace("/", "\\").casefold()
        if pid.strip().isdigit() and int(pid) != os.getpid() and any(target in command for target in targets):
            pids.append(int(pid))
    return pids


def stop_previous_instances() -> int:
    """Stop older PatraAI servers so a new start always runs the current code."""
    pids = previous_instances()
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    if pids:
        print(f"Stopped {len(pids)} earlier PatraAI process(es).", flush=True)
        time.sleep(1)
    return len(pids)


def main() -> int:
    if "--stop" in sys.argv[1:]:
        if not stop_previous_instances():
            print("PatraAI is not running.", flush=True)
        return 0
    stop_previous_instances()
    missing, broken = package_issues()
    if missing or broken:
        if missing:
            print("Installing missing required packages: " + ", ".join(missing), flush=True)
        if broken:
            print("Repairing incomplete package installation(s): " + ", ".join(broken), flush=True)
        pip_timeout = os.getenv("INDIAN_APP_PIP_TIMEOUT", "300")
        pip_retries = os.getenv("INDIAN_APP_PIP_RETRIES", "5")
        install_command = [
            sys.executable, "-m", "pip", "install",
            "--disable-pip-version-check",
            "--default-timeout", pip_timeout,
            "--retries", pip_retries,
        ]
        if broken:
            install_command.extend(["--force-reinstall", "--no-cache-dir"])
        install_command.extend([
            "-r", str(REQUIREMENTS),
        ])
        result = subprocess.run(install_command, cwd=ROOT)
        if result.returncode:
            print("Package installation failed. Check your internet connection and try again.", flush=True)
            return result.returncode
        still_missing, still_broken = package_issues()
        if still_missing or still_broken:
            details = ", ".join(still_missing + still_broken)
            print(f"Required package imports still fail in {sys.executable}: {details}", flush=True)
            return 1

    ensure_indictrans2()
    try:
        ensure_ollama_runtime()
        ensure_ollama_models()
    except Exception as exc:
        print(f"Ollama startup failed: {exc}", flush=True)
        if os.getenv("INDIAN_APP_ALLOW_NO_OLLAMA", "0") != "1":
            return 1

    warm_up_model()
    port = available_port()
    command = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.headless", "true",
        "--server.address", "127.0.0.1",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]
    # The server binds IPv4 only; using localhost can resolve to ::1 first on Windows
    # and cause the browser's Streamlit WebSocket to reset.
    print("Starting PatraAI at http://127.0.0.1:" + str(port), flush=True)
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        if not wait_until_ready(port, process):
            code = process.poll()
            if code is None:
                process.terminate()
                code = process.wait(timeout=10)
            print("The website did not start. Review the Streamlit error above.", flush=True)
            return code or 1

        url = f"http://127.0.0.1:{port}"
        print("PatraAI ready: " + url, flush=True)
        if os.getenv("INDIAN_APP_NO_BROWSER") != "1":
            webbrowser.open_new_tab(url)
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=10)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
