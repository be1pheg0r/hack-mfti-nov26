"""Hydra-configured loopback-only demo server with bearer-token API auth."""

import asyncio
import base64
import hmac
import ipaddress
import json
import logging
import os
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
    "/journey.js": ("journey.js", "text/javascript; charset=utf-8"),
    "/funnel.webp": ("funnel.webp", "image/webp"),
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
        if path not in {
            "/api/demo/analyze",
            "/api/demo/gate",
            "/api/demo/pipeline",
        }:
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
            if path in {"/api/demo/gate", "/api/demo/pipeline"}:
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
                else self.service.pipeline(resume_id)
                if path == "/api/demo/pipeline"
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


class DemoASGI:
    """ASGI implementation of the demo HTTP contract with deferred startup."""

    def __init__(self, service=None, token="", port=18765, public_origin=""):
        self.service = service
        self.token = token
        self.port = int(port)
        self.public_origin = public_origin
        self.client = None

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        headers = {
            key.decode("latin1").lower(): value.decode("latin1")
            for key, value in scope.get("headers", [])
        }
        status, body, content_type = await self._dispatch(scope, headers, receive)
        response_headers = [
            (b"content-type", content_type.encode("latin1")),
            (b"content-length", str(len(body)).encode()),
            (b"cache-control", b"no-store"),
            (b"x-content-type-options", b"nosniff"),
            (
                b"content-security-policy",
                (
                    b"default-src 'self'; style-src 'self' "
                    b"https://fonts.googleapis.com; "
                    b"font-src https://fonts.gstatic.com; img-src 'self' data:; "
                    b"connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"
                ),
            ),
        ]
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": response_headers,
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _lifespan(self, receive, send):
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    if self.service is None:
                        self._start_service()
                    await send({"type": "lifespan.startup.complete"})
                except Exception:
                    await send(
                        {
                            "type": "lifespan.startup.failed",
                            "message": "Demo service startup failed",
                        }
                    )
                    return
            elif message["type"] == "lifespan.shutdown":
                if self.client is not None:
                    self.client.__exit__(None, None, None)
                    self.client = None
                await send({"type": "lifespan.shutdown.complete"})
                return

    def _start_service(self):
        config = _load_asgi_config()
        self.token, self.port = _validate_server_config(config)
        llm = LLMConfig.model_validate(OmegaConf.to_container(config.llm, resolve=True))
        self.client = OpenRouterClient(llm)
        self.client.__enter__()
        self.service = _make_service(config, self.client)
        self.public_origin = config.server.public_origin

    def _permitted(self, headers):
        hosts = {f"localhost:{self.port}", f"127.0.0.1:{self.port}"}
        allowed_origins = {f"http://{host}" for host in hosts}
        if self.public_origin:
            hosts.add(urlsplit(self.public_origin).netloc)
        origin = headers.get("origin")
        if self.public_origin:
            allowed_origins.add(self.public_origin)
        return headers.get("host", "") in hosts and (
            not origin or origin in allowed_origins
        )

    def _authenticated(self, headers):
        supplied = headers.get("authorization", "")
        encoded = base64.urlsafe_b64encode(self.token.encode()).rstrip(b"=").decode()
        return hmac.compare_digest(
            supplied.encode(), f"Bearer utf8.{encoded}".encode()
        ) or hmac.compare_digest(supplied.encode(), f"Bearer {self.token}".encode())

    async def _dispatch(self, scope, headers, receive):
        def response(status, value, content_type="application/json; charset=utf-8"):
            body = (
                value
                if isinstance(value, bytes)
                else json.dumps(value, ensure_ascii=False).encode()
            )
            return status, body, content_type

        if not self._permitted(headers):
            return response(403, {"error": "Forbidden host or origin"})
        method, path = scope["method"], scope["path"]
        if method == "GET" and path in FILES:
            name, content_type = FILES[path]
            try:
                body = (ASSETS / name).read_bytes()
            except OSError:
                return response(404, {"error": "Not found"})
            return response(200, body, content_type)
        if method == "GET" and path == "/api/demo":
            if not self._authenticated(headers):
                return response(401, {"error": "Требуется токен"})
            try:
                return response(200, await asyncio.to_thread(self.service.catalog))
            except Exception as error:
                logging.error(
                    "Demo catalog extraction failed (%s)", type(error).__name__
                )
                return response(502, {"error": "Не удалось обработать вакансии."})
        routes = {"/api/demo/analyze", "/api/demo/gate", "/api/demo/pipeline"}
        if method != "POST" or path not in routes:
            return response(404, {"error": "Not found"})
        if not self._authenticated(headers):
            return response(401, {"error": "Требуется токен"})
        if headers.get("content-type", "").split(";")[0] != "application/json":
            return response(415, {"error": "Expected JSON"})
        chunks, size = [], 0
        while True:
            message = await receive()
            size += len(message.get("body", b""))
            if size > 4096:
                return response(400, {"error": "Некорректный запрос."})
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        try:
            data = json.loads(b"".join(chunks))
            if not isinstance(data, dict):
                raise ValueError
            resume_id = data.get("resume_id")
            if path in {"/api/demo/gate", "/api/demo/pipeline"}:
                if not isinstance(resume_id, str) or resume_id not in {
                    r["id"] for r in self.service.resumes
                }:
                    raise ValueError
            elif not isinstance(data.get("vacancy_id"), str):
                raise ValueError
            if resume_id is not None and (
                not isinstance(resume_id, str)
                or resume_id not in {r["id"] for r in self.service.resumes}
            ):
                raise ValueError
            if path == "/api/demo/analyze" and data["vacancy_id"] not in {
                v["id"] for v in self.service.vacancies
            }:
                raise ValueError
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError, KeyError):
            return response(400, {"error": "Некорректный запрос."})
        try:
            if path == "/api/demo/gate":
                result = await asyncio.to_thread(self.service.gate, resume_id)
            elif path == "/api/demo/pipeline":
                result = await asyncio.to_thread(self.service.pipeline, resume_id)
            else:
                result = await asyncio.to_thread(
                    self.service.analyze, data["vacancy_id"], resume_id=resume_id
                )
            return response(200, result)
        except Exception as error:
            logging.error("Demo extraction failed (%s)", type(error).__name__)
            return response(
                502,
                {
                    "error": "Не удалось обработать ответ модели. "
                    "Проверьте настройки и попробуйте ещё раз."
                },
            )


def _load_asgi_config():
    """Load Hydra-composed settings passed by the parent or use project defaults."""
    text = os.environ.get("VRSCORE_DEMO_CONFIG_YAML")
    if text:
        return OmegaConf.create(text)
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="demo")


def _validate_server_config(config):
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
    return token, int(config.server.port)


def _make_service(config, client):
    return DemoService(
        client,
        prompts=OmegaConf.to_container(config.llm.prompts.extraction, resolve=True),
        vocabulary=OmegaConf.to_container(
            config.feature_extraction.vocabulary, resolve=True
        ),
        labels=OmegaConf.to_container(config.feature_extraction.labels, resolve=True),
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
        embedding_url=config.pipeline.embedding_url,
        embedding_timeout=config.pipeline.embedding_timeout,
        shortlist_size=config.pipeline.shortlist_size,
        demo_delay_seconds=config.pipeline.demo_delay_seconds,
        vacancy_template=OmegaConf.to_container(
            config.feature_extraction.vacancy_template, resolve=True
        ),
    )


def _hydra_app():
    """Import target used by uvicorn reload workers."""
    return DemoASGI()


app = _hydra_app()


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[2] / "configs"),
    config_name="demo",
)
def main(config: DictConfig):
    """Launch the ASGI app with Hydra settings and a private bind by default."""
    token, port = _validate_server_config(config)
    del token  # Validate early without copying the secret into logs or arguments.
    os.environ["VRSCORE_DEMO_CONFIG_YAML"] = OmegaConf.to_yaml(config, resolve=True)
    import uvicorn

    address = str(ipaddress.ip_address(config.server.bind))
    reload_dirs = [
        str(Path(__file__).resolve().parent),
        str(Path(__file__).resolve().parents[2] / "configs"),
    ]
    print(f"Demo: http://127.0.0.1:{port}", flush=True)
    uvicorn.run(
        "vrscore.demo.server:app",
        host=address,
        port=port,
        reload=bool(config.server.reload),
        reload_dirs=reload_dirs if config.server.reload else None,
        reload_includes=["*.py", "*.yaml"] if config.server.reload else None,
        access_log=False,
        log_config=None,
        lifespan="on",
    )
