IMAGE=$1
NAME=$2
INITIALS=$3
HOME=$4
docker run -p ${INITIALS}000-${INITIALS}099:${INITIALS}000-${INITIALS}099 --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -itd -v /home/${HOME}:/home/${HOME} -v /mnt/:/mnt/ --name ${NAME} ${IMAGE} ${INITIALS}
