IMAGE=$1
NAME=$2
INITIALS=$3
HOME=$4
INITIALS=$(printf '%03d' $INITIALS)
docker run -p ${INITIALS}00-${INITIALS}99:${INITIALS}00-${INITIALS}99 --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/${HOME}:/home/${HOME} -v /mnt/:/mnt/ --name ${NAME} ${IMAGE} ${INITIALS}
# docker run -p 28000-28099:28000-28099 --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/:/home/ -v /mnt/:/mnt/ --name <name> beam:<date> 28
