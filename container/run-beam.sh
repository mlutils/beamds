#!/bin/bash

# Function to display help
show_help() {
  echo "Usage: $0 IMAGE NAME INITIALS HOME_DIR [GPU_FLAG] [MODE] [COMMAND] [MORE_DOCKER_ARGS]"
  echo ""
  echo "  IMAGE           Docker image to use"
  echo "  NAME            Name of the Docker container"
  echo "  INITIALS        Initials for port mapping, formatted to 3 digits"
  echo "  HOME_DIR        Home directory to mount inside the container"
  echo "  [GPU_FLAG]      'gpu-off' to disable GPU, default is GPU on"
  echo "  [MODE]          'it' for interactive mode, 'd' for detached mode, default is 'itd'"
  echo "  [COMMAND]       Command to run inside the container (optional)"
  echo "  [MORE_DOCKER_ARGS]     Additional docker run arguments (optional)"
  echo ""
  echo "Examples:"
  echo "  $0 my_image my_container 123 /home/user gpu-on ltd \"-v /path:/path\" bash"
  echo "  $0 my_image my_container 123 /home/user gpu-off 'echo Hello' '-p 1234:1234 --env KEY=value'"
}

# Validate and handle arguments
if [[ $1 == "-h" || $1 == "--help" ]]; then
  show_help
  exit 0
fi

if [ "$#" -lt 4 ]; then
  echo "Error: Missing required arguments."
  show_help
  exit 1
fi

IMAGE=$1
NAME=$2
INITIALS=$3
HOME_DIR=$4
GPU_FLAG=${5:-"gpu-on"}
MODE=${6:-"itd"}
MORE_DOCKER_ARGS=${7:-""}
COMMAND=${@:8}

echo "before Formatted INITIALS: ${INITIALS}"
# INITIALS=$(printf '%03d' $(echo $INITIALS | rev) | rev)
INITIALS=$(printf '%03d' "$(echo "$INITIALS" | rev)" | rev)
echo "after Formatted INITIALS: ${INITIALS}"
echo "Running a new container named: $NAME, Based on image: $IMAGE"
echo "Jupyter port will be available at: ${INITIALS}88"
echo "Additional arguments: ${MORE_DOCKER_ARGS}"

# Get total system memory in kilobytes (kB)
total_memory_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
# Calculate 90% backoff of total memory
backoff_memory_kb=$(awk -v x=$total_memory_kb 'BEGIN {printf "%.0f", x * 0.9}')
# Convert to megabytes for Docker
backoff_memory_mb=$(awk -v x=$backoff_memory_kb 'BEGIN {printf "%.0f", x / 1024}')

echo "Home directory: $HOME_DIR"
gpu_option="--gpus all"
if [ "$GPU_FLAG" == "gpu-off" ]; then
  gpu_option=""
fi

port_mapping="-p ${INITIALS}00-${INITIALS}99:${INITIALS}00-${INITIALS}99"
if echo "$MORE_DOCKER_ARGS" | grep "network=host"; then
  port_mapping=""
fi

echo "Port mapping: $port_mapping"
echo "Executing Docker command:"

run_mode=""
if [ "$MODE" == "it" ]; then
  run_mode="-it"
elif [ "$MODE" == "d" ]; then
  run_mode="-d"
else
  run_mode="-itd"  # Default is interactive, tty, detached
fi

DOCKER_RUN_COMMAND="docker run $port_mapping --ipc=host --ulimit memlock=-1 $gpu_option --shm-size=8g --memory=${backoff_memory_mb}m --ulimit stack=67108864 --restart unless-stopped $run_mode -v $HOME_DIR:$HOME_DIR -v /mnt/:/mnt/ -v /var/run/docker.sock:/var/run/docker.sock -e INITIALS=${INITIALS} $MORE_DOCKER_ARGS --name $NAME --hostname $NAME $IMAGE $COMMAND"

# Print the final docker run command and execute it
echo "$DOCKER_RUN_COMMAND"
$DOCKER_RUN_COMMAND