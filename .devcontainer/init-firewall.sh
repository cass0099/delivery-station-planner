#!/bin/bash
# Blocks all outgoing network traffic from the container except to an allowlist.
# Runs at every container start. Adapted from Anthropic's Claude Code reference devcontainer.
set -euo pipefail
IFS=$'\n\t'

# Sites this project is allowed to reach
ALLOWED_DOMAINS=(
  # Python packages
  "pypi.org"
  "files.pythonhosted.org"
  # Claude Code
  "api.anthropic.com"
  "claude.ai"
  "console.anthropic.com"
  "platform.claude.com"
  # Map data (OSMnx)
  "overpass-api.de"
  "nominatim.openstreetmap.org"
  # Census data
  "api.census.gov"
  "www2.census.gov"
  # GitHub content not covered by GitHub's published ranges
  "raw.githubusercontent.com"
  "objects.githubusercontent.com"
  # VS Code extensions
  "marketplace.visualstudio.com"
  "vscode.blob.core.windows.net"
  "update.code.visualstudio.com"
)

# Keep Docker's internal DNS rules before clearing everything
DOCKER_DNS_RULES=$(iptables-save -t nat | grep "127\.0\.0\.11" || true)

iptables -F; iptables -X
iptables -t nat -F; iptables -t nat -X
iptables -t mangle -F; iptables -t mangle -X
ipset destroy allowed-domains 2>/dev/null || true

if [ -n "$DOCKER_DNS_RULES" ]; then
  iptables -t nat -N DOCKER_OUTPUT 2>/dev/null || true
  iptables -t nat -N DOCKER_POSTROUTING 2>/dev/null || true
  echo "$DOCKER_DNS_RULES" | xargs -L 1 iptables -t nat
fi

# DNS and loopback
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A INPUT  -p udp --sport 53 -j ACCEPT
iptables -A INPUT  -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

ipset create allowed-domains hash:net

# GitHub's published IP ranges
echo "Fetching GitHub IP ranges..."
gh_meta=$(curl -fsS https://api.github.com/meta)
echo "$gh_meta" | jq -r '(.web + .api + .git)[]' | grep -v ':' | aggregate -q | while read -r cidr; do
  ipset add -exist allowed-domains "$cidr"
done

# Everything else by DNS lookup
for domain in "${ALLOWED_DOMAINS[@]}"; do
  ips=$(dig +short A "$domain" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' || true)
  if [ -z "$ips" ]; then
    echo "WARNING: could not resolve $domain (skipping)"
    continue
  fi
  while read -r ip; do
    ipset add -exist allowed-domains "$ip"
  done <<< "$ips"
done

# Allow talking to the host (needed for VS Code and port forwarding)
HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
HOST_NET=$(echo "$HOST_IP" | sed 's/\.[0-9]*$/.0\/24/')
iptables -A INPUT  -s "$HOST_NET" -j ACCEPT
iptables -A OUTPUT -d "$HOST_NET" -j ACCEPT

# Default: block
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP
iptables -A INPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m set --match-set allowed-domains dst -j ACCEPT
iptables -A OUTPUT -j REJECT --reject-with icmp-admin-prohibited

# Block IPv6 entirely so it can't be used to get around the rules
if command -v ip6tables >/dev/null 2>&1; then
  ip6tables -F 2>/dev/null || true
  ip6tables -A INPUT  -i lo -j ACCEPT 2>/dev/null || true
  ip6tables -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true
  ip6tables -P INPUT DROP 2>/dev/null || true
  ip6tables -P FORWARD DROP 2>/dev/null || true
  ip6tables -P OUTPUT DROP 2>/dev/null || true
fi

# Check it worked
if curl -fsS --connect-timeout 5 https://example.com >/dev/null 2>&1; then
  echo "ERROR: firewall check failed, example.com is reachable"
  exit 1
fi
if ! curl -fsS --connect-timeout 5 https://api.github.com/zen >/dev/null 2>&1; then
  echo "ERROR: firewall check failed, GitHub is not reachable"
  exit 1
fi
echo "Firewall ready: only allowlisted sites are reachable."
