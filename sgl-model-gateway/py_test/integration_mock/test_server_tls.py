"""
Integration tests for server-side TLS (HTTPS/HTTP2) support.

Tests verify that:
1. Router can serve HTTPS requests with proper certificates
2. HTTP/2 works over TLS
3. Router with mTLS (require client cert) works with proper client certificates
4. Router with mTLS fails when client doesn't provide certificates
"""

import subprocess
import time
from pathlib import Path
from typing import Tuple

import pytest
import requests

from ..fixtures.ports import find_free_port


def get_test_certs_dir() -> Path:
    """Get the path to the test certificates directory."""
    return Path(__file__).parent.parent / "fixtures" / "test_certs"


def _spawn_mock_worker(port: int, worker_id: str) -> Tuple[subprocess.Popen, str]:
    """Spawn a plain HTTP mock worker."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "py_test" / "fixtures" / "mock_worker.py"

    cmd = [
        "python3",
        str(script),
        "--port",
        str(port),
        "--worker-id",
        worker_id,
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    url = f"http://127.0.0.1:{port}"

    # Wait for worker to start
    time.sleep(2)

    if proc.poll() is not None:
        _, stderr = proc.communicate()
        raise RuntimeError(f"Worker failed to start.\nStderr: {stderr}")

    # Wait for worker to be ready
    _wait_health(url)
    return proc, url


def _wait_health(url: str, timeout: float = 10.0):
    """Wait for HTTP worker to become healthy."""
    start = time.time()
    with requests.Session() as s:
        while time.time() - start < timeout:
            try:
                r = s.get(f"{url}/health", timeout=1)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.2)
    raise TimeoutError(f"Worker at {url} did not become healthy")


@pytest.mark.integration
def test_server_tls_https_request(router_manager, test_certificates):
    """Test that router can serve HTTPS requests with TLS enabled."""
    certs_dir = test_certificates

    # Start a plain HTTP mock worker
    worker_port = find_free_port()
    worker_id = f"worker-{worker_port}"
    worker_proc, worker_url = _spawn_mock_worker(worker_port, worker_id)

    try:
        # Start router with server TLS enabled
        rh = router_manager.start_router(
            worker_urls=[worker_url],
            policy="round_robin",
            extra={
                "server_tls_cert_path": str(certs_dir / "server-cert.pem"),
                "server_tls_key_path": str(certs_dir / "server-key.pem"),
                # For health check during startup
                "server_tls_ca_cert_for_client": str(certs_dir / "ca-cert.pem"),
            },
        )

        # Make HTTPS request to router - should succeed
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=5,
            verify=str(certs_dir / "ca-cert.pem"),
        )

        assert r.status_code == 200, f"Request failed: {r.status_code} {r.text}"
        data = r.json()
        assert "choices" in data
        assert data.get("worker_id") == worker_id

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()


@pytest.mark.integration
def test_server_tls_health_endpoint(router_manager, test_certificates):
    """Test that health endpoint works over HTTPS."""
    certs_dir = test_certificates

    # Start a plain HTTP mock worker
    worker_port = find_free_port()
    worker_id = f"worker-{worker_port}"
    worker_proc, worker_url = _spawn_mock_worker(worker_port, worker_id)

    try:
        # Start router with server TLS enabled
        rh = router_manager.start_router(
            worker_urls=[worker_url],
            policy="round_robin",
            extra={
                "server_tls_cert_path": str(certs_dir / "server-cert.pem"),
                "server_tls_key_path": str(certs_dir / "server-key.pem"),
                "server_tls_ca_cert_for_client": str(certs_dir / "ca-cert.pem"),
            },
        )

        # Check health endpoint over HTTPS
        r = requests.get(
            f"{rh.url}/health",
            timeout=5,
            verify=str(certs_dir / "ca-cert.pem"),
        )

        assert r.status_code == 200, f"Health check failed: {r.status_code} {r.text}"

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()


@pytest.mark.integration
def test_server_tls_fails_without_ca_verification(router_manager, test_certificates):
    """Test that HTTPS request fails when client doesn't verify server cert."""
    certs_dir = test_certificates

    # Start a plain HTTP mock worker
    worker_port = find_free_port()
    worker_id = f"worker-{worker_port}"
    worker_proc, worker_url = _spawn_mock_worker(worker_port, worker_id)

    try:
        # Start router with server TLS enabled
        rh = router_manager.start_router(
            worker_urls=[worker_url],
            policy="round_robin",
            extra={
                "server_tls_cert_path": str(certs_dir / "server-cert.pem"),
                "server_tls_key_path": str(certs_dir / "server-key.pem"),
                "server_tls_ca_cert_for_client": str(certs_dir / "ca-cert.pem"),
            },
        )

        # Try to make HTTPS request WITHOUT verifying server cert (but not disabling verification)
        # This should fail because the server uses a self-signed cert
        with pytest.raises(requests.exceptions.SSLError):
            requests.get(
                f"{rh.url}/health",
                timeout=5,
                verify=True,  # Use system CA bundle, which won't have our test CA
            )

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()


@pytest.mark.integration
def test_server_mtls_successful_with_client_cert(router_manager, test_certificates):
    """Test that router with mTLS accepts requests with valid client certificates."""
    certs_dir = test_certificates

    # Start a plain HTTP mock worker
    worker_port = find_free_port()
    worker_id = f"worker-{worker_port}"
    worker_proc, worker_url = _spawn_mock_worker(worker_port, worker_id)

    try:
        # Start router with server mTLS enabled (requires client certificate)
        rh = router_manager.start_router(
            worker_urls=[worker_url],
            policy="round_robin",
            extra={
                "server_tls_cert_path": str(certs_dir / "server-cert.pem"),
                "server_tls_key_path": str(certs_dir / "server-key.pem"),
                "server_tls_client_ca_cert_path": str(certs_dir / "ca-cert.pem"),
                "server_tls_require_client_cert": True,
                # For health check during startup - provide client cert
                "server_tls_ca_cert_for_client": str(certs_dir / "ca-cert.pem"),
                "server_tls_client_cert_for_test": str(certs_dir / "client-cert.pem"),
                "server_tls_client_key_for_test": str(certs_dir / "client-key.pem"),
            },
        )

        # Make request WITH client certificate - should succeed
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=5,
            verify=str(certs_dir / "ca-cert.pem"),
            cert=(
                str(certs_dir / "client-cert.pem"),
                str(certs_dir / "client-key.pem"),
            ),
        )

        assert r.status_code == 200, f"Request failed: {r.status_code} {r.text}"
        data = r.json()
        assert "choices" in data
        assert data.get("worker_id") == worker_id

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()


@pytest.mark.integration
def test_server_mtls_fails_without_client_cert(router_manager, test_certificates):
    """Test that router with mTLS rejects requests without client certificates."""
    certs_dir = test_certificates

    # Start a plain HTTP mock worker
    worker_port = find_free_port()
    worker_id = f"worker-{worker_port}"
    worker_proc, worker_url = _spawn_mock_worker(worker_port, worker_id)

    try:
        # Start router with server mTLS enabled (requires client certificate)
        rh = router_manager.start_router(
            worker_urls=[worker_url],
            policy="round_robin",
            extra={
                "server_tls_cert_path": str(certs_dir / "server-cert.pem"),
                "server_tls_key_path": str(certs_dir / "server-key.pem"),
                "server_tls_client_ca_cert_path": str(certs_dir / "ca-cert.pem"),
                "server_tls_require_client_cert": True,
                # For health check during startup - provide client cert
                "server_tls_ca_cert_for_client": str(certs_dir / "ca-cert.pem"),
                "server_tls_client_cert_for_test": str(certs_dir / "client-cert.pem"),
                "server_tls_client_key_for_test": str(certs_dir / "client-key.pem"),
            },
        )

        # Try to make request WITHOUT client certificate - should fail
        with pytest.raises(requests.exceptions.SSLError):
            requests.get(
                f"{rh.url}/health",
                timeout=5,
                verify=str(certs_dir / "ca-cert.pem"),
                # Note: no client cert provided
            )

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()


@pytest.mark.integration
def test_server_tls_http2_support(router_manager, test_certificates):
    """Test that HTTP/2 works over TLS."""
    certs_dir = test_certificates

    # Start a plain HTTP mock worker
    worker_port = find_free_port()
    worker_id = f"worker-{worker_port}"
    worker_proc, worker_url = _spawn_mock_worker(worker_port, worker_id)

    try:
        # Start router with server TLS enabled
        rh = router_manager.start_router(
            worker_urls=[worker_url],
            policy="round_robin",
            extra={
                "server_tls_cert_path": str(certs_dir / "server-cert.pem"),
                "server_tls_key_path": str(certs_dir / "server-key.pem"),
                "server_tls_ca_cert_for_client": str(certs_dir / "ca-cert.pem"),
            },
        )

        # Use httpx to make HTTP/2 request (requests doesn't support HTTP/2)
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not installed, skipping HTTP/2 test")

        # Create HTTP/2 client
        with httpx.Client(
            http2=True,
            verify=str(certs_dir / "ca-cert.pem"),
        ) as client:
            r = client.get(f"{rh.url}/health", timeout=5)

            assert r.status_code == 200, f"Health check failed: {r.status_code} {r.text}"
            # Check if HTTP/2 was used
            assert r.http_version == "HTTP/2", f"Expected HTTP/2 but got {r.http_version}"

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()


@pytest.mark.integration
def test_server_tls_streaming_response(router_manager, test_certificates):
    """Test that streaming responses work over HTTPS."""
    certs_dir = test_certificates

    # Start a plain HTTP mock worker
    worker_port = find_free_port()
    worker_id = f"worker-{worker_port}"
    worker_proc, worker_url = _spawn_mock_worker(worker_port, worker_id)

    try:
        # Start router with server TLS enabled
        rh = router_manager.start_router(
            worker_urls=[worker_url],
            policy="round_robin",
            extra={
                "server_tls_cert_path": str(certs_dir / "server-cert.pem"),
                "server_tls_key_path": str(certs_dir / "server-key.pem"),
                "server_tls_ca_cert_for_client": str(certs_dir / "ca-cert.pem"),
            },
        )

        # Make streaming HTTPS request
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 5,
                "stream": True,
            },
            timeout=10,
            verify=str(certs_dir / "ca-cert.pem"),
            stream=True,
        )

        assert r.status_code == 200, f"Request failed: {r.status_code}"

        # Collect streaming chunks
        chunks = []
        for line in r.iter_lines():
            if line:
                chunks.append(line.decode("utf-8"))

        # Should have received some SSE data
        assert len(chunks) > 0, "Expected streaming response chunks"
        # Check that chunks are SSE formatted
        assert any(chunk.startswith("data:") for chunk in chunks), "Expected SSE format"

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()
