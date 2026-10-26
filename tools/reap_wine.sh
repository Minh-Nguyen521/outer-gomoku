#!/usr/bin/env bash
# Kill abandoned oog-wine containers.
#
# third_party/wine-engine runs `docker run --rm -i --name oogwine-$$`, and
# `exec` keeps the PID, so the docker client IS the wrapper process. That makes
# an orphan trivially identifiable: a container named oogwine-<pid> whose <pid>
# no longer exists.
#
# Why they pile up: `--rm` only fires when the docker client exits cleanly.
# SIGKILL the wrapper -- which is how matches get stopped here -- and the
# container detaches and keeps running, holding a CPU share and a Wine prefix.
# Seven containers against five live wrappers is what prompted this.
#
# Containers created before wine-engine started naming them have no oogwine-
# name. Those are only removed with --all, because there is no safe way to tell
# an in-use one from an abandoned one.
#
# Usage:
#   tools/reap_wine.sh            # kill orphans (named, PID gone)
#   tools/reap_wine.sh --all      # also kill every unnamed oog-wine container
#   tools/reap_wine.sh --dry-run  # just report

set -u
MODE="${1:-}"

docker ps >/dev/null 2>&1 || { echo "docker not running"; exit 0; }

killed=0
kept=0
unnamed=0

while read -r id name; do
    [ -z "$id" ] && continue
    case "$name" in
        oogwine-*)
            pid="${name#oogwine-}"
            # `kill -0` is not enough. A docker client can outlive the match
            # that spawned it: the process still exists, so kill -0 succeeds,
            # while the container sits there burning CPU. The tell is the
            # PARENT -- when the owner dies the client is reparented to init
            # (ppid 1), so ppid==1 means abandoned. Live ones have a real
            # match.py or shell as their parent.
            ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
            if kill -0 "$pid" 2>/dev/null && [ "${ppid:-1}" != "1" ]; then
                kept=$((kept + 1))
            else
                if [ "$MODE" = "--dry-run" ]; then
                    why=$([ -n "${ppid:-}" ] && [ "${ppid:-1}" = "1" ] \
                          && echo "orphaned, ppid 1" || echo "pid $pid gone")
                    echo "  would kill $id ($name, $why)"
                else
                    docker kill "$id" >/dev/null 2>&1 && echo "  killed $id ($name)"
                fi
                killed=$((killed + 1))
            fi
            ;;
        *)
            unnamed=$((unnamed + 1))
            if [ "$MODE" = "--all" ]; then
                docker kill "$id" >/dev/null 2>&1 && echo "  killed $id (unnamed, --all)"
                killed=$((killed + 1))
            elif [ "$MODE" = "--dry-run" ]; then
                echo "  would keep $id (unnamed -- predates PID naming; use --all)"
            fi
            ;;
    esac
done < <(docker ps --filter ancestor=oog-wine:latest --format '{{.ID}} {{.Names}}')

echo "reaped $killed, in-use $kept, unnamed left $([ "$MODE" = "--all" ] && echo 0 || echo $unnamed)"
