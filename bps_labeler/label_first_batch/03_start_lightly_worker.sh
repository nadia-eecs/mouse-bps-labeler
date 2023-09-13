#!/bin/bash

# read the values from the .env file
source .env

# pull the latest version of the lightly worker
docker pull lightly/worker:latest 

# start the lightly worker
docker run --shm-size "1024m" --gpus all --rm -it \
    -e LIGHTLY_TOKEN='${MY_LIGHTLY_TOKEN}' \
    -e LIGHTLY_WORKER_ID='${LIGHTLY_WORKER_ID}' \
    lightly/worker:latest