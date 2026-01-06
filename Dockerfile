# RABEL MCP Server - Docker Image
# AI-to-AI messaging via I-Poll protocol and AInternet
#
# Build: docker build -t mcp-server-rabel .
# Run:   docker run -i mcp-server-rabel
#
# Part of HumoticaOS/SymbAIon - https://humotica.com

FROM python:3.11-slim

LABEL maintainer="Jasper van de Meent <info@humotica.com>"
LABEL org.opencontainers.image.source="https://github.com/jaspertvdm/mcp-server-rabel"
LABEL org.opencontainers.image.description="RABEL - AI memory and I-Poll messaging for MCP"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"

# Install from PyPI
RUN pip install --no-cache-dir mcp-server-rabel

# Create data directory for memories
RUN mkdir -p /data
ENV RABEL_DATA_DIR=/data

# MCP servers communicate via stdio
# Run with: docker run -i mcp-server-rabel
ENTRYPOINT ["mcp-server-rabel"]
