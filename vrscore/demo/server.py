"""Hydra-configured loopback-only demo server with bearer-token API auth."""

import base64
import hmac
import ipaddress
import json
import logging
import os
import signal
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as BaseThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import hydra
from omegaconf import DictConfig, OmegaConf

from vrscore.demo.service import DemoService
from vrscore.llm import LLMConfig, OpenRouterClient

ASSETS = Path(__file__).resolve().parents[1] / "web" / "demo"
FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/fight.css": ("fight.css", "text/css; charset=utf-8"),
    "/fsm.css": ("fsm.css", "text/css; charset=utf-8"),
    "/typography.css": ("typography.css", "text/css; charset=utf-8"),
    "/logo.png": ("logo.png", "image/png"),
    "/favicon.ico": ("favicon.ico", "image/vnd.microsoft.icon"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class ThreadingHTTPServer(BaseThreadingHTTPServer):
    """Wait for active requests before closing during a service restart."""

    daemon_threads = False


class Handler(BaseHTTPRequestHandler):
    """Serve only known files and require a token for every data operation."""

    def __init__(self, *args, service, token, port, public_origin="", **kwargs):
        self.service, self.token, self.port = service, token, port
        self.public_origin = public_origin
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Avoid logging credentials, inputs, and request payloads."""

    def respond(self, status, body, content_type="application/json; charset=utf-8"):
        """Write a no-cache response with defensive browser headers."""
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:; "
            "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def permitted(self):
        """Reject DNS rebinding and non-local browser origins."""
        hosts = {f"localhost:{self.port}", f"127.0.0.1:{self.port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        allowed = {f"http://{h}" for h in hosts}
        if self.public_origin:
            allowed.add(self.public_origin)
            hosts.add(urlsplit(self.public_origin).netloc)
        return host in hosts and (not origin or origin in allowed)

    def authenticated(self):
        """Compare the API token without exposing it in URLs or server logs."""
        supplied = self.headers.get("Authorization", "")
        encoded = base64.urlsafe_b64encode(self.token.encode()).rstrip(b"=").decode()
        return hmac.compare_digest(
            supplied.encode(), f"Bearer utf8.{encoded}".encode()
        ) or (hmac.compare_digest(supplied.encode(), f"Bearer {self.token}".encode()))

    def do_GET(self):
        if not self.permitted():
            return self.respond(403, {"error": "Forbidden host or origin"})
        path = urlsplit(self.path).path
        if path in FILES:
            name, content_type = FILES[path]
            return self.respond(200, (ASSETS / name).read_bytes(), content_type)
        if path == "/api/demo":
            if not self.authenticated():
                return self.respond(401, {"error": "Требуется токен"})
            try:
                result = self.service.catalog()
            except Exception as error:
                logging.error(
                    "Demo catalog extraction failed (%s)", type(error).__name__
                )
                return self.respond(502, {"error": "Не удалось обработать вакансии."})
            return self.respond(200, result)
        self.respond(404, {"error": "Not found"})

    def do_POST(self):
        if not self.permitted():
            return self.respond(403, {"error": "Forbidden host or origin"})
        path = urlsplit(self.path).path
        if path not in {"/api/demo/analyze", "/api/demo/gate"}:
            return self.respond(404, {"error": "Not found"})
        if not self.authenticated():
            return self.respond(401, {"error": "Требуется токен"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.respond(415, {"error": "Expected JSON"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid body size")
            data = json.loads(self.rfile.read(length))
            if path == "/api/demo/gate":
                if (
                    not isinstance(data, dict)
                    or not isinstance(data.get("resume_id"), str)
                    or data["resume_id"] not in {r["id"] for r in self.service.resumes}
                ):
                    raise ValueError("Unknown resume")
            elif not isinstance(data, dict) or not isinstance(
                data.get("vacancy_id"), str
            ):
                raise ValueError("Invalid request")
            resume_id = data.get("resume_id")
            if resume_id is not None and not isinstance(resume_id, str):
                raise ValueError("Invalid resume")
            if path == "/api/demo/analyze" and data["vacancy_id"] not in {
                v["id"] for v in self.service.vacancies
            }:
                raise ValueError("Unknown vacancy")
            if resume_id is not None and resume_id not in {
                r["id"] for r in self.service.resumes
            }:
                raise ValueError("Unknown resume")
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return self.respond(400, {"error": "Некорректный запрос."})
        try:
            result = (
                self.service.gate(resume_id)
                if path == "/api/demo/gate"
                else self.service.analyze(data["vacancy_id"], resume_id=resume_id)
            )
        except Exception as error:
            logging.error("Demo extraction failed (%s)", type(error).__name__)
            return self.respond(
                502,
                {
                    "error": "Не удалось обработать ответ модели. "
                    "Проверьте настройки и попробуйте ещё раз."
                },
            )
        self.respond(200, result)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[2] / "configs"),
    config_name="demo",
)
def main(config: DictConfig):
    """Start on a private interface; secrets come from the environment."""
    token = os.getenv(config.auth.token_env, "")
    if len(token) < config.auth.min_token_length or not token:
        raise RuntimeError(
            f"Set {config.auth.token_env} to a token of at least "
            f"{config.auth.min_token_length} characters"
        )
    address = ipaddress.ip_address(config.server.bind)
    if not address.is_private or address.is_unspecified:
        raise ValueError("Demo may only bind to a private address")
    origin = config.server.public_origin
    if origin:
        parsed = urlsplit(origin)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("public_origin must be an HTTPS origin")
    llm = LLMConfig.model_validate(OmegaConf.to_container(config.llm, resolve=True))
    with OpenRouterClient(llm) as client:
        server = ThreadingHTTPServer(
            (str(address), config.server.port),
            partial(
                Handler,
                service=DemoService(
                    client,
                    prompts=OmegaConf.to_container(
                        config.llm.prompts.extraction, resolve=True
                    ),
                    vocabulary=OmegaConf.to_container(
                        config.feature_extraction.vocabulary, resolve=True
                    ),
                    labels=OmegaConf.to_container(
                        config.feature_extraction.labels, resolve=True
                    ),
                    postprocessing=OmegaConf.to_container(
                        config.feature_extraction.postprocessing, resolve=True
                    ),
                    gate_config=OmegaConf.to_container(config.first_gate, resolve=True),
                    cache_dir=config.server.cache_dir,
                    extra_vacancies_path=config.server.extra_vacancies_path,
                    matching_gt_path=config.server.matching_gt_path,
                    real_resumes_path=(
                        config.server.real_resumes_path
                        if config.server.real_resumes_enabled
                        else None
                    ),
                    vacancy_template=OmegaConf.to_container(
                        config.feature_extraction.vacancy_template, resolve=True
                    ),
                ),
                token=token,
                port=config.server.port,
                public_origin=origin,
            ),
        )
        print(f"Demo: http://127.0.0.1:{config.server.port}", flush=True)
        stopping = threading.Event()

        def stop_on_signal(signum, frame):
            if not stopping.is_set():
                stopping.set()
                threading.Thread(target=server.shutdown, daemon=True).start()

        previous_handler = signal.signal(signal.SIGTERM, stop_on_signal)
        try:
            server.serve_forever()
        finally:
            server.server_close()
            signal.signal(signal.SIGTERM, previous_handler)
