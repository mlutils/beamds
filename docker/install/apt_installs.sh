#!/bin/bash

apt update
apt update

apt install -y openssh-server
apt install -y libkrb5
apt install -y libkrb5-dev
apt install -y htop
apt install -y tmux
apt install -y libtool
apt install -y dh-autoreconf
apt install -y debhelper
apt install -y libaio-dev
apt install -y telnet
apt install -y iputils-ping
apt install -y net-tools
apt install -y python3-sphinx
apt install -y graphviz
apt install -y pkg-config
apt install -y expect
apt install -y libgraphviz-dev
apt install -y cron
apt install -y default-jdk
apt install -y ltrace
apt install -y binutils
apt install -y alien
apt install -y libaio1
apt install -y binwalk
apt install -y ncdu
apt install -y vim
apt install -y zip
apt install -y unzip
apt install -y unixodbc-dev
apt install -y sqlite3
apt install -y python3.10-venv
apt install -y redis-server
apt install -y rabbitmq-server
apt install -y ca-certificates
apt install -y curl
apt install -y gnupg
apt install -y lsb-release
#
## don't install libopenmpi as it messes up the pytorch geometric installation
#apt install -y libopenmpi-dev

pip install pykerberos


## for hfds-fuse installation
#apt install -y libprotobuf-c-dev
#apt install -y protobuf-c-compiler
#apt install -y libfuse-dev
#apt install -y uncrustify

# applications

# ssh serve
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# ssh connection immediately disconnects after session start with exit code 254:
# https://unix.stackexchange.com/questions/148714/cant-ssh-connection-terminates-immediately-with-exit-status-254
sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config