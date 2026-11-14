"""Exercise the configured CLI through the local embedding HTTP protocol."""

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def test_cli_ranks_json_documents_via_embedding_service(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            vectors = [[1.0, 0.0] for _ in payload["texts"]]
            body = json.dumps({"embeddings": vectors, "dimension": 2}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    skill = {"category": "hard_skill", "value": "Python", "polarity": "present"}
    resume = {
        "id": "r",
        "kind": "resume",
        "text": "Python developer",
        "entities": [skill],
    }
    vacancies = [
        {
            "id": "v",
            "kind": "vacancy",
            "text": "Python developer wanted",
            "entities": [skill],
        }
    ]
    rpath, vpath = tmp_path / "r.json", tmp_path / "v.json"
    rpath.write_text(json.dumps(resume))
    vpath.write_text(json.dumps(vacancies))
    try:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                str(root / ".venv/bin/python"),
                "-m",
                "vrscore.retrieval_cli",
                f"resume={rpath}",
                f"vacancies={vpath}",
                f"server=http://127.0.0.1:{server.server_port}",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout)
        assert output["results"][0]["id"] == "v"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
