#!/bin/bash
# Alpaca Web Entrypoint
#
# Runs as root so it can repair filesystem ownership on the bind-mounted
# ./data directory (subdirectories may have been created by a root process and
# are unwritable by the non-root `alpaca` runtime user), then drops privileges
# to the `alpaca` user to run the dashboard.
#
# The docker socket group (DOCKER_GID) is re-registered in the container's
# /etc/group before dropping privileges — `su` recomputes the target user's
# supplementary groups from /etc/group, so without this step the runtime user
# would lose access to the docker socket (group_add only applies to the
# container's initial user, not to users switched to afterwards).
set -e

fix_permissions() {
    # Ensure every data directory is writable by the alpaca runtime user.
    chown -R alpaca:alpaca /app/data 2>/dev/null || true
    mkdir -p /app/data/llm_benchmarks/models /app/data/shared_llm_benchmarks/models
    chown -R alpaca:alpaca /app/data/llm_benchmarks/models /app/data/shared_llm_benchmarks/models 2>/dev/null || true
}

restore_docker_group() {
    if [ -n "${DOCKER_GID:-}" ]; then
        if ! getent group "$DOCKER_GID" >/dev/null 2>&1; then
            groupadd -g "$DOCKER_GID" hostdocker 2>/dev/null || true
        fi
        usermod -aG "$DOCKER_GID" alpaca 2>/dev/null || true
    fi
}

if [ "$(id -u)" = "0" ]; then
    fix_permissions
    restore_docker_group
    exec su -s /bin/bash alpaca -c "cd /app && exec python web/app.py"
else
    exec python web/app.py
fi
