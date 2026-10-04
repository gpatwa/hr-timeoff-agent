# One image for every service (web app, A2A agents, MCP server, init): the command differs.
#
#   docker build -t hr-timeoff-agent .
#   docker run --rm hr-timeoff-agent web --host 0.0.0.0
#
# The embedding models are downloaded at build time, so a container never reaches out to
# Hugging Face when it starts (and starts the same way with no outbound network).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    HR_EMBED_LOCAL_ONLY=1 \
    HR_WEB_HOME=/var/lib/hr \
    HR_A2A_HOME=/var/lib/hr

WORKDIR /app

# Dependencies first, so editing code does not reinstall them. The package directory is
# copied as a stub, so the install resolves everything from pyproject.toml alone.
COPY pyproject.toml README.md ./
RUN mkdir hr_timeoff_agent && touch hr_timeoff_agent/__init__.py \
    && pip install -e '.[web,mcp,a2a,postgres,otel]' \
    && rm -rf hr_timeoff_agent

# The code and the data it reads by relative path (data/, fixtures/ sit beside the package).
COPY hr_timeoff_agent hr_timeoff_agent
COPY data data
COPY fixtures fixtures
RUN pip install -e '.[web,mcp,a2a,postgres,otel]' --no-deps

# Bake the embedding models into the image.
RUN HR_EMBED_LOCAL_ONLY=0 python -c "from fastembed import SparseTextEmbedding, TextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25')"

# Not root. The app writes only under /var/lib/hr (caches and local state).
RUN useradd --system --uid 10001 --home-dir /var/lib/hr --create-home --shell /usr/sbin/nologin hr \
    && chown -R hr:hr /var/lib/hr \
    && chmod -R a+rX /opt/fastembed /app
USER hr

EXPOSE 8000 8100 8101 8200
ENTRYPOINT ["python", "-m", "hr_timeoff_agent"]
CMD ["web", "--host", "0.0.0.0"]
