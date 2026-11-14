FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 UV_LINK_MODE=copy
WORKDIR /app
RUN pip install --no-cache-dir uv==0.13.7 \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/.data /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH" HOME=/home/app HF_HOME=/home/app/.cache/huggingface
COPY --chown=app:app configs ./configs
COPY --chown=app:app vrscore ./vrscore
USER app

FROM base AS demo
CMD ["python", "-m", "vrscore.demo", "server.reload=false"]

FROM base AS embeddings
USER root
# Install the lockfile's exact CPU torch builds without downloading CUDA runtimes.
RUN uv export --locked --no-dev --no-emit-project --no-hashes --extra embeddings > /tmp/requirements-all.txt \
    && python -c "from pathlib import Path; import re; p=Path('/tmp/requirements-all.txt'); s=p.read_text(); Path('/tmp/requirements-cpu.txt').write_text(''.join(line for line in s.splitlines(keepends=True) if not re.match(r'^(nvidia-[^=]*|triton|torch|torchvision)==', line))); print('\\n'.join(line.split(' ;')[0] for line in s.splitlines() if re.match(r'^(torch|torchvision)==', line)))" > /tmp/torch-cpu.txt \
    && uv pip install --python /app/.venv/bin/python --no-deps --index-url https://download.pytorch.org/whl/cpu -r /tmp/torch-cpu.txt \
    && uv pip install --python /app/.venv/bin/python --no-deps -r /tmp/requirements-cpu.txt \
    && rm /tmp/requirements-all.txt /tmp/requirements-cpu.txt /tmp/torch-cpu.txt
USER app
CMD ["python", "-m", "vrscore.embedding_server"]
