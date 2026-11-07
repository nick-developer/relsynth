# relsynth with every database driver, for CI/CD jobs and scheduled runs.
#
#   docker build -t relsynth:2.0.0 .
#   docker run --rm -v "$PWD:/work" -e PGPASSWORD relsynth:2.0.0 run /work/spec.yaml --json
#
# The image runs as an unprivileged user; mount the folder with the spec (and any
# CSV / Parquet files) at /work.  Credentials come from environment variables.
#
# The base image is pinned by the digest of its multi-arch index (Dependabot's docker
# updates move the digest); python:3.12-slim-bookworm when the digest was taken.
ARG PYTHON=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
FROM ${PYTHON} AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install --no-cache-dir build==1.2.2 && python -m build --wheel --outdir /dist

FROM ${PYTHON}
LABEL org.opencontainers.image.title="relsynth" \
      org.opencontainers.image.source="https://github.com/nick-developer/relsynth" \
      org.opencontainers.image.licenses="Apache-2.0"
ARG DEBIAN_FRONTEND=noninteractive
# a failed curl in `curl | gpg` must fail the build, not leave an empty keyring
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# Microsoft ODBC Driver 18 for SQL Server (pyodbc). It is proprietary software under
# Microsoft's own licence terms (accepted by ACCEPT_EULA=Y), not under Apache-2.0: see NOTICE.
# curl and gnupg only to add the repository
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gnupg ca-certificates unixodbc \
 && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg \
 && echo "deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
      > /etc/apt/sources.list.d/mssql-release.list \
 && apt-get update \
 && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
 && apt-get purge -y curl gnupg && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build /dist/*.whl /tmp/
COPY LICENSE NOTICE /usr/share/doc/relsynth/
RUN pip install --no-cache-dir "$(ls /tmp/relsynth-*.whl)[all]" && rm /tmp/*.whl \
 && useradd --create-home --uid 10001 relsynth
USER relsynth
WORKDIR /work
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["relsynth"]
CMD ["--help"]
