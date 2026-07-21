#!/usr/bin/env bash
set -uo pipefail

env OTEL_TRACES_SAMPLER=always_off /opt/jaeger/jaeger &
jaeger_pid=$!

"$@" &
main_pid=$!

terminate() {
  kill -TERM "${jaeger_pid}" "${main_pid}" 2>/dev/null || true
  wait "${jaeger_pid}" "${main_pid}" 2>/dev/null || true
}

trap 'terminate; exit 0' INT TERM

wait -n "${jaeger_pid}" "${main_pid}"
status=$?
terminate
exit "${status}"
