"""Read-only, explicitly scoped HTTP serving for the private dataset explorer."""

import ipaddress
import json
import logging
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[2]
GROUPS = [
    {"id": "it", "label": "IT и разработка"},
    {"id": "finance", "label": "Финансы и учёт"},
    {"id": "education", "label": "Образование"},
    {"id": "engineering", "label": "Инженерные роли"},
    {"id": "office_service", "label": "Офис и сервис"},
]
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def load_dataset(directory: Path) -> dict:
    """Load exact dataset records and explicitly browser-safe manifest fields.

    Args:
        directory: Directory containing both JSONL files and their manifest.

    Returns:
        Records, safe provenance metadata and display groups.

    Raises:
        ValueError: Record IDs, kinds or declared counts are inconsistent.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must be an object")
    records, seen = [], set()
    for filename, kind, count_key in [
        ("resumes.jsonl", "resume", "resumes"),
        ("vacancies.jsonl", "vacancy", "vacancies"),
    ]:
        rows = [
            json.loads(line)
            for line in (directory / filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Dataset records must be objects")
            identifier = row.get("id")
            if not identifier or identifier in seen:
                raise ValueError(f"Duplicate or missing record ID: {identifier}")
            if row.get("kind") != kind:
                raise ValueError(f"Unexpected record kind for {identifier}")
            seen.add(identifier)
        if len(rows) != manifest["actual"][count_key]:
            raise ValueError(f"Dataset count differs from manifest: {filename}")
        records.extend(rows)
    safe_manifest = {
        key: manifest[key]
        for key in [
            "name",
            "version",
            "actual",
            "source_published_date_ranges",
            "labels",
        ]
        if key in manifest
    }
    provenance = manifest.get("provenance", {})
    safe_manifest["provenance"] = {
        key: provenance[key]
        for key in [
            "original_publisher",
            "copy_repository",
            "original_access_date_reported_by_copy",
        ]
        if key in provenance
    }
    known = {group["id"] for group in GROUPS}
    additional = sorted(
        {
            row["sampling_group"]
            for row in records
            if isinstance(row.get("sampling_group"), str)
            and row["sampling_group"]
            and row["sampling_group"] not in known
        }
    )
    groups = GROUPS + [{"id": group, "label": group} for group in additional]
    return {"records": records, "manifest": safe_manifest, "groups": groups}


def discover_datasets(root: Path, known_paths: set[Path]) -> dict[str, dict]:
    """Load compatible direct children inside the configured discovery root.

    Dataset directories cannot be symlinks; required files may link to other
    files inside the root, as when a corpus shares another corpus's resumes.
    Broken or incompatible datasets are skipped without breaking other sets.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        return {}
    discovered = {}
    for directory in sorted(root.iterdir(), key=lambda path: path.name):
        if directory.is_symlink() or not directory.is_dir():
            continue
        try:
            resolved = directory.resolve(strict=True)
            if resolved in known_paths:
                continue
            for filename in ("manifest.json", "resumes.jsonl", "vacancies.jsonl"):
                file = (directory / filename).resolve(strict=True)
                if not file.is_relative_to(root) or not file.is_file():
                    raise ValueError("Dataset file is outside the discovery root")
            discovered[directory.name] = load_dataset(directory)
            known_paths.add(resolved)
        except (OSError, ValueError, KeyError, TypeError) as error:
            logging.getLogger(__name__).warning(
                "Skipping incompatible dataset %s: %s", directory.name, error
            )
    return discovered


class ViewerHandler(BaseHTTPRequestHandler):
    """Serve an asset allowlist; never expose the repository as a directory."""

    def __init__(
        self,
        *args,
        payloads: dict,
        catalog: bytes,
        default_dataset: str,
        assets: Path,
        allowed_hosts: set,
        **kwargs,
    ):
        self.payloads = payloads
        self.catalog = catalog
        self.default_dataset = default_dataset
        self.assets = assets
        self.allowed_hosts = allowed_hosts
        super().__init__(*args, **kwargs)

    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        """Write consistent security and cache headers for every response."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Robots-Tag", "noindex, nofollow, noarchive")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'self'",
        )
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:
        """Handle a read-only API or an explicitly known frontend asset."""
        try:
            hostname = urlsplit("//" + self.headers.get("Host", "")).hostname
        except ValueError:
            hostname = None
        if not hostname or hostname.lower() not in self.allowed_hosts:
            self._respond(403, b"Forbidden host", "text/plain; charset=utf-8")
            return
        url = urlsplit(self.path)
        path = unquote(url.path)
        if path == "/api/datasets":
            self._respond(200, self.catalog, "application/json; charset=utf-8")
            return
        if path == "/api/dataset":
            params = parse_qs(url.query, keep_blank_values=True)
            ids = params.get("dataset", [self.default_dataset])
            payload = self.payloads.get(ids[0]) if len(ids) == 1 else None
            if payload is None:
                self._respond(404, b"Not found", "text/plain; charset=utf-8")
                return
            self._respond(200, payload, "application/json; charset=utf-8")
            return
        asset = ASSETS.get(path)
        if asset:
            filename, content_type = asset
            file = self.assets / filename
            if file.is_file():
                self._respond(200, file.read_bytes(), content_type)
                return
        self._respond(404, b"Not found", "text/plain; charset=utf-8")

    def do_HEAD(self) -> None:
        """Return the same headers without response bodies."""
        self.do_GET()

    def log_message(self, format: str, *args) -> None:
        """Log request metadata, never dataset payloads or authorization headers."""
        logging.getLogger(__name__).info(format, *args)


def create_server(
    dataset_dir: Path,
    assets_dir: Path,
    host: str,
    port: int,
    *,
    allowed_hosts: list[str] | None = None,
    allow_private_bind: bool = False,
    extra_datasets: dict[str, Path] | None = None,
    discovery_root: Path | None = None,
) -> ThreadingHTTPServer:
    """Create a loopback server or an explicit private reverse-proxy backend.

    Args:
        dataset_dir: Corpus directory; never made directly HTTP-accessible.
        assets_dir: Frontend directory, restricted by ASSETS.
        host: Loopback or explicitly permitted private interface address.
        port: TCP port, or zero for an ephemeral test port.
        allowed_hosts: Additional reverse-proxy hostnames to accept.
        allow_private_bind: Permit a private bridge interface, never a public bind.
        extra_datasets: Additional explicitly configured dataset paths.
        discovery_root: Optional root whose direct child datasets are discovered.

    Returns:
        Unstarted HTTP server.

    Raises:
        ValueError: Address would expose an unauthenticated public backend.
    """
    address = ipaddress.ip_address("127.0.0.1" if host == "localhost" else host)
    permitted_private = (
        allow_private_bind and address.is_private and not address.is_unspecified
    )
    if not address.is_loopback and not permitted_private:
        raise ValueError(
            "Use loopback or an explicitly allowed private proxy interface"
        )
    directories = {"mini": dataset_dir}
    if extra_datasets:
        if "mini" in extra_datasets:
            raise ValueError("Cannot override the default dataset")
        directories.update(extra_datasets)
    datasets = {}
    for key, directory in directories.items():
        if discovery_root is not None and not Path(directory).exists():
            logging.getLogger(__name__).info("Configured dataset %s is absent", key)
            continue
        datasets[key] = load_dataset(directory)
    if discovery_root is not None:
        known_paths = {Path(directory).resolve() for directory in directories.values()}
        for key, data in discover_datasets(discovery_root, known_paths).items():
            if key not in datasets:
                datasets[key] = data
    if not datasets:
        raise ValueError("No compatible datasets found")
    default_dataset = "mini" if "mini" in datasets else next(iter(datasets))
    payloads, catalog = {}, []
    for key, data in datasets.items():
        payloads[key] = json.dumps(data, ensure_ascii=False).encode("utf-8")
        catalog.append(
            {
                "id": key,
                "name": data["manifest"].get("name", key),
                "actual": data["manifest"].get("actual", {}),
            }
        )
    catalog_bytes = json.dumps(
        {"datasets": catalog, "default_dataset": default_dataset}, ensure_ascii=False
    ).encode("utf-8")
    hosts = {"localhost", "127.0.0.1", "::1", str(address)}
    hosts.update(h.lower() for h in (allowed_hosts or []))
    handler = partial(
        ViewerHandler,
        payloads=payloads,
        catalog=catalog_bytes,
        default_dataset=default_dataset,
        assets=Path(assets_dir),
        allowed_hosts=hosts,
    )
    return ThreadingHTTPServer((str(address), port), handler)


@hydra.main(
    version_base=None, config_path=str(ROOT / "configs"), config_name="dataset_viewer"
)
def main(config: DictConfig) -> None:
    """Start the viewer using Hydra paths, host and port overrides."""

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    server = create_server(
        resolve(config.dataset_dir),
        resolve(config.assets_dir),
        config.server.host,
        config.server.port,
        allowed_hosts=list(config.server.allowed_hosts),
        allow_private_bind=config.server.allow_private_bind,
        extra_datasets={
            key: resolve(path) for key, path in config.extra_datasets.items()
        },
        discovery_root=(
            resolve(config.discovery_root) if config.discovery_root else None
        ),
    )
    print(
        f"Dataset viewer ready: http://{config.server.host}:{server.server_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
