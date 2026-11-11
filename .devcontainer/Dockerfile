FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# Tools for the network firewall
# (The base image ships a Yarn package source with an expired key; we don't use it.)
RUN rm -f /etc/apt/sources.list.d/yarn.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      iptables ipset iproute2 dnsutils aggregate jq curl \
 && rm -rf /var/lib/apt/lists/*

# Headless browser so Claude can screenshot the app and check the design (installed at build time,
# before the firewall starts; the app itself does not need it).
RUN apt-get update \
 && apt-get install -y --no-install-recommends chromium fonts-liberation fonts-noto-color-emoji \
 && rm -rf /var/lib/apt/lists/*

# Firewall script. The vscode user may run this one script as root and nothing else,
# so code inside the container can't switch the firewall off.
COPY .devcontainer/init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod 755 /usr/local/bin/init-firewall.sh \
 && rm -f /etc/sudoers.d/vscode \
 && echo "vscode ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/vscode-firewall \
 && chmod 0440 /etc/sudoers.d/vscode-firewall

USER vscode

# Folders that the named volumes mount over (so they're owned by vscode)
RUN mkdir -p /home/vscode/.claude /home/vscode/cache

# Python packages go in a venv the vscode user owns (no root needed later)
RUN python -m venv /home/vscode/.venv
ENV PATH="/home/vscode/.venv/bin:/home/vscode/.local/bin:${PATH}"

COPY --chown=vscode:vscode requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Dev-only: drives the system Chromium for screenshots. Not in requirements.txt, so the deployed app skips it.
RUN pip install --no-cache-dir playwright

# Claude Code (official installer). Updates happen by rebuilding the container.
RUN curl -fsSL https://claude.ai/install.sh | bash
