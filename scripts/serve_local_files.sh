#!/usr/bin/env bash
# https://github.com/HumanSignal/label-studio/blob/master/scripts/serve_local_files.sh
# Modified to use nginx and to reverse proxy label-studio
PORT=${PORT:-"8080"}

echo "Usage: sh serve_local_files.sh PORT"
echo "This script starts web server on the port PORT [$PORT by default] that serves files from INPUT_DIR"
echo

echo "Running web server on the port ${PORT}"

# Get the absolute path of the script's parent
SCRIPT_DIR="$(realpath $(dirname ${BASH_SOURCE[0]}))"

# get the path relative to this script
nginx -c $(realpath ${SCRIPT_DIR}/../config/label-studio/nginx.conf)
