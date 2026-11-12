#!/bin/bash
# Universal training entrypoint
set -e
uv run python "$@"
