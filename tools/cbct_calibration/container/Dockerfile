FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential cmake ninja-build libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY tools/cbct_calibration/requirements.txt /tmp/cbct-calibration-requirements.txt
RUN pip install --no-cache-dir \
    -r /tmp/cbct-calibration-requirements.txt pytest

WORKDIR /workspace
