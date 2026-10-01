"""Launcher behavior when the tool or another service already owns its port."""

import errno
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import bootstrap


@pytest.mark.parametrize(
    ("marker", "health", "marker_status", "expected"),
    [
        (b'{"application":"wording-style-visualizer","protocol":1}', b"ok", 200, True),
        (b'<html>Another Streamlit app</html>', b"ok", 200, False),
        (b'{"application":"another-tool","protocol":1}', b"ok", 200, False),
        (b'{"application":"wording-style-visualizer","protocol":1}', b"unavailable", 200, False),
        (b'{"application":"wording-style-visualizer","protocol":1}', b"ok", 302, False),
        (b" " * 1100 + b"{}", b"ok", 200, False),
    ],
)
def test_only_our_healthy_app_is_recognized(monkeypatch, marker, health, marker_status, expected):
    responses = {
        "/app/static/token-atlas.json": (marker_status, marker),
        "/_stcore/health": (200, health),
    }

    class Connection:
        def __init__(self, host, port, timeout):
            assert host == "127.0.0.1"
            assert port == 8501
            assert 0 < timeout <= 2

        def request(self, method, path):
            assert method == "GET"
            self.path = path

        def getresponse(self):
            status, body = responses[self.path]
            return SimpleNamespace(status=status, read=lambda limit: body[:limit])

        def close(self):
            pass

    monkeypatch.setattr(bootstrap.http.client, "HTTPConnection", Connection)
    assert bootstrap.is_running_app(8501) is expected


def test_unresponsive_service_is_not_treated_as_our_app(monkeypatch):
    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            raise TimeoutError("The local service did not respond")

        def close(self):
            pass

    monkeypatch.setattr(bootstrap.http.client, "HTTPConnection", Connection)
    assert bootstrap.is_running_app(8501) is False


def mock_ports(monkeypatch, occupied, *, fallback=49152):
    attempted = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def bind(self, address):
            host, port = address
            assert host == "127.0.0.1"
            assert 0 <= port <= 65535
            attempted.append(port)
            if port in occupied:
                raise OSError(errno.EADDRINUSE, "Address already in use")
            self.port = port or fallback

        def getsockname(self):
            return "127.0.0.1", self.port

        def close(self):
            pass

    monkeypatch.setattr(bootstrap.socket, "socket", lambda *args, **kwargs: Socket())
    return attempted


def test_existing_tool_is_reused_without_binding_a_second_server(monkeypatch):
    monkeypatch.setattr(bootstrap, "is_running_app", lambda port: port == 8501)
    attempted = mock_ports(monkeypatch, {8501})
    assert bootstrap.choose_port(8501) == (8501, True)
    assert not attempted


def test_unrelated_services_are_left_alone_and_next_free_port_is_used(monkeypatch):
    monkeypatch.setattr(bootstrap, "is_running_app", lambda port: False)
    mock_ports(monkeypatch, {8501, 8502})
    assert bootstrap.choose_port(8501) == (8503, False)


def test_tool_on_alternative_port_is_reused(monkeypatch):
    monkeypatch.setattr(bootstrap, "is_running_app", lambda port: port == 8502)
    mock_ports(monkeypatch, {8501, 8502})
    assert bootstrap.choose_port(8501) == (8502, True)


@pytest.mark.parametrize("preferred", [8501, 65535])
def test_os_allocates_port_when_requested_range_is_full(monkeypatch, preferred):
    monkeypatch.setattr(bootstrap, "is_running_app", lambda port: False)
    mock_ports(monkeypatch, set(range(preferred, 65536)), fallback=52000)
    assert bootstrap.choose_port(preferred) == (52000, False)


def test_reopening_tool_opens_loopback_url(monkeypatch, capsys):
    opened = []
    monkeypatch.setattr(bootstrap.webbrowser, "open", lambda url: opened.append(url))
    bootstrap.open_app(8502, reused=True)
    assert opened == ["http://127.0.0.1:8502"]
    assert "8502" in capsys.readouterr().out


@pytest.fixture
def prepared_launcher(monkeypatch, tmp_path):
    python = tmp_path / "python"
    python.write_text("placeholder", encoding="utf-8")
    events = SimpleNamespace(installed=[], launched=[], opened=[])
    monkeypatch.setattr(bootstrap.sys, "argv", ["bootstrap.py"])
    monkeypatch.setattr(bootstrap, "environment_python", lambda: python)
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(bootstrap, "install_requirements", lambda python, name: events.installed.append(name))
    monkeypatch.setattr(bootstrap, "run", lambda command: events.launched.append(command))
    monkeypatch.setattr(bootstrap, "is_running_app", lambda port: False)
    monkeypatch.setattr(bootstrap, "is_port_available", lambda port: True)
    monkeypatch.setattr(bootstrap, "choose_port", lambda port: (8502, False))
    monkeypatch.setattr(bootstrap, "open_app", lambda port, reused=False: events.opened.append((port, reused)))
    return events


def test_repeat_launch_skips_setup_and_does_not_start_another_process(monkeypatch, prepared_launcher):
    monkeypatch.setattr(bootstrap, "is_running_app", lambda port: True)
    assert bootstrap.main() == 0
    assert prepared_launcher.opened == [(8501, True)]
    assert not prepared_launcher.installed
    assert not prepared_launcher.launched


def test_setup_check_never_opens_or_starts_server(monkeypatch, prepared_launcher):
    monkeypatch.setattr(bootstrap.sys, "argv", ["bootstrap.py", "--check"])
    monkeypatch.setattr(bootstrap, "is_running_app", lambda port: True)
    assert bootstrap.main() == 0
    assert "requirements.txt" in prepared_launcher.installed
    assert "requirements-models.txt" in prepared_launcher.installed
    assert not prepared_launcher.opened
    assert not prepared_launcher.launched


def test_new_server_uses_selected_port_and_loopback_only(prepared_launcher):
    assert bootstrap.main() == 0
    assert len(prepared_launcher.launched) == 1
    command = prepared_launcher.launched[0]
    assert "--server.port=8502" in command
    assert "--server.address=127.0.0.1" in command
    assert str(Path(bootstrap.ROOT) / "app.py") in command


def test_existing_app_found_on_alternative_port_is_opened_without_launching(monkeypatch, prepared_launcher):
    monkeypatch.setattr(bootstrap, "choose_port", lambda port: (8502, True))
    assert bootstrap.main() == 0
    assert prepared_launcher.opened == [(8502, True)]
    assert not prepared_launcher.launched


def test_port_claimed_during_startup_is_retried(monkeypatch, prepared_launcher):
    selected = iter([(8502, False), (8503, False)])
    monkeypatch.setattr(bootstrap, "choose_port", lambda port: next(selected))
    monkeypatch.setattr(bootstrap, "is_port_available", lambda port: False)

    def launch(command):
        prepared_launcher.launched.append(command)
        if len(prepared_launcher.launched) == 1:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(bootstrap, "run", launch)
    assert bootstrap.main() == 0
    assert len(prepared_launcher.launched) == 2
    assert "--server.port=8503" in prepared_launcher.launched[-1]


def test_server_launch_failure_does_not_blame_network(monkeypatch, prepared_launcher, capsys):
    def fail_launch(command):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(bootstrap, "run", fail_launch)
    assert bootstrap.main() == 1
    message = capsys.readouterr().err
    assert "Startup did not finish" in message
    assert "Check your connection" not in message
    assert "检查网络" not in message
