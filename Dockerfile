FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CODESEARCH_DATABASE=/var/lib/codesearch/metadata.sqlite3
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install .[inference]

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /var/lib/codesearch \
    && chown -R app:app /app /var/lib/codesearch
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1
CMD ["codesearch", "serve", "--host", "0.0.0.0", "--port", "8000"]
