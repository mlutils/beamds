#!/bin/bash

IMAGE=$1
NAME=$2
INITIALS=$3
INITIALS=$(printf '%03d' $(echo $INITIALS | rev) | rev)
HOME_DIR=$4
MORE_ARGS=${@:5}

echo "Running a new container named: $NAME, Based on image: $IMAGE"
echo "Jupyter port will be available at: ${INITIALS}88"

echo $MORE_ARGS

# Get total system memory in kilobytes (kB)
total_memory_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
# Calculate 90% backoff of total memory
backoff_memory_kb=$(awk -v x=$total_memory_kb 'BEGIN {printf "%.0f", x * 0.9}')
# Convert to megabytes for Docker
backoff_memory_mb=$(awk -v x=$backoff_memory_kb 'BEGIN {printf "%.0f", x / 1024}')

# -e USER_HOME=${HOME}
echo "Home directory: ${HOME_DIR}"


output=$(docker context ls)

# Search for the word "rootless" in the output
if echo "$output" | grep -q "rootless"; then
  # If "rootless" is found, set the flag to true
  root_args=" "
else
  # If "rootless" is not found, set the flag to false
#  root_args=" --cap-add=NET_ADMIN --ipc=host --ulimit memlock=-1"
  root_args=" --ipc=host --ulimit memlock=-1"
fi

if echo ${MORE_ARGS} | grep "network=host"; then
  port_mapping=""
else
  port_mapping="-p ${INITIALS}00-${INITIALS}99:${INITIALS}00-${INITIALS}99"
fi

echo "Port mapping: ${port_mapping}"

docker run ${port_mapping} ${root_args} --gpus=all --shm-size=8g --memory=${backoff_memory_mb}m --ulimit stack=67108864 --restart unless-stopped -it -v ${HOME_DIR}:${HOME_DIR} -v /mnt/:/mnt/ ${MORE_ARGS} --name ${NAME} --hostname ${NAME} ${IMAGE} ${INITIALS}
# docker run -p 28000-28099:28000-28099 --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/:/home/ -v /mnt/:/mnt/ --name <name> beam:<date> 28
