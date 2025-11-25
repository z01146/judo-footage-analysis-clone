import pytest
import requests
import threading
import time
from judo_footage_analysis.label_studio.debug_backend.wsgi import main
from pathlib import Path

APP_STARTUP_RETRIES = 10
APP_STARTUP_DELAY = 0.5
APP_PORT = 9876
APP_URL = f"http://localhost:{APP_PORT}"


@pytest.fixture(scope="session")
def tmp_model_path(tmp_path_factory):
    return tmp_path_factory.mktemp("model")


@pytest.fixture(scope="session", autouse=True)
def app(tmp_model_path):
    # monkeypatch sys.argv to avoid passing command-line arguments
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "sys.argv",
            [
                "wsgi.py",
                "--port",
                f"{APP_PORT}",
                "--model-dir",
                f"{tmp_model_path}",
                "--log-level",
                "DEBUG",
            ],
        )
        # run the app in the background in a separate thread or process
        thread = threading.Thread(target=main, daemon=True)
        thread.start()
        # wait for the app to start
        # NOTE: ugly code
        for _ in range(APP_STARTUP_RETRIES):
            try:
                resp = requests.get(APP_URL)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(APP_STARTUP_DELAY)
        if resp.status_code != 200:
            raise RuntimeError("Failed to start the app")
        yield


def test_app_health(tmp_model_path):
    resp = requests.get(f"{APP_URL}/health")
    data = resp.json()
    assert resp.status_code == 200
    assert data["status"] == "UP"
    assert data["model_dir"] == str(tmp_model_path)


def test_app_setup():
    request = {
        "project": "test",
        "schema": "<View/>",
        "model_version": "test_model",
    }
    resp = requests.post(f"{APP_URL}/setup", json=request)
    assert resp.status_code == 200
    print(resp.json())


def test_app_predict(tmp_model_path):
    request = {
        "tasks": [{}],
        "model_version": "test_model",
    }
    resp = requests.post(f"{APP_URL}/predict", json=request)
    assert resp.status_code == 200
    print(resp.json())
    print(tmp_model_path)
