#!/bin/bash
# PARAM / IITP demo launcher — override paths via env vars before running.
#
# Example:
#   export EARTHDIAL_GPU=1
#   export EARTHDIAL_MODEL_PATH=/home/rihak_iitp/EarthDial_Models/EarthDial_4B_RGB
#   export EARTHDIAL_CONTROLLER_URL=http://0.0.0.0:40000
#   bash earthdial_demo.sh

set -euo pipefail

EARTHDIAL_GPU="${EARTHDIAL_GPU:-1}"
EARTHDIAL_MODEL_PATH="${EARTHDIAL_MODEL_PATH:-/home/rihak_iitp/EarthDial_Models/EarthDial_4B_RGB}"
EARTHDIAL_CONTROLLER_URL="${EARTHDIAL_CONTROLLER_URL:-http://0.0.0.0:40000}"
EARTHDIAL_SD_WORKER_URL="${EARTHDIAL_SD_WORKER_URL:-http://0.0.0.0:39999}"

mkdir -p logs bash_logs

# Run the Streamlit app
nohup streamlit run app.py --server.port 10003 -- \
  --controller_url "${EARTHDIAL_CONTROLLER_URL}" \
  --sd_worker_url "${EARTHDIAL_SD_WORKER_URL}" \
  > logs/streamlit_app.log 2>&1 &

echo "Streamlit app started on port 10003. Logs: logs/streamlit_app.log"

sleep 10

# Run the controller
nohup python controller.py --host 0.0.0.0 --port 40000 > logs/controller.log 2>&1 &

echo "Controller started on port 40000. Logs: logs/controller.log"

echo "Now loading model into UI"
# Run the model worker
CUDA_VISIBLE_DEVICES="${EARTHDIAL_GPU}" nohup python model_worker.py \
    --host 0.0.0.0 \
    --controller "${EARTHDIAL_CONTROLLER_URL}" \
    --port 40001 \
    --worker http://0.0.0.0:40001 \
    --model-path "${EARTHDIAL_MODEL_PATH}" \
    > bash_logs/model_worker.log 2>&1 &

echo " Model running on this http://localhost:10003/ "
echo " GPU=${EARTHDIAL_GPU}  MODEL=${EARTHDIAL_MODEL_PATH}"
echo "All processes have been started."
