#!/bin/bash

apt update
apt update

apt install -y openssh-server
apt install -y htop -y tmux -y libtool -y dh-autoreconf -y debhelper -y libaio-dev
apt install -y telnet -y iputils-ping -y python-dev -y net-tools -y python3-sphinx
apt install -y graphviz  -y python3-dev graphviz pkg-config -y expect -y libgraphviz-dev -y cron
apt install -y default-jdk
apt install -y ltrace -y binutils
apt install -y alien -y libaio1 -y binwalk -y

# don't install libopenmpi as it messes up the pytorch geometric installation
# libopenmpi-dev


## for hfds-fuse installation
#apt install -y libprotobuf-c-dev
#apt install -y protobuf-c-compiler
#apt install -y libfuse-dev
#apt install -y uncrustify

# applications

# ssh server
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config