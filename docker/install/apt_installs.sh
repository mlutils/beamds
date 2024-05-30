#!/bin/bash

apt update
apt update

echo "Installing openssh-server"
apt install -y openssh-server
echo "Installing libkrb5"
apt install -y libkrb5
echo "Installing libkrb5-dev"
apt install -y libkrb5-dev
echo "Installing htop"
apt install -y htop
echo "Installing tmux"
apt install -y tmux
echo "Installing libtool"
apt install -y libtool
echo "Installing dh-autoconf"
apt install -y dh-autoreconf
echo "Installing debhelper"
apt install -y debhelper
echo "Installing libaio-dev"
apt install -y libaio-dev
echo "Installing telnet"
apt install -y telnet
echo "Installing iputils-ping"
apt install -y iputils-ping
echo "Installing net-tools"
apt install -y net-tools
echo "Installing python3-sphinx"
apt install -y python3-sphinx
echo "Installing graphviz"
apt install -y graphviz
echo "Installing pkg-config"
apt install -y pkg-config
echo "Installing expect"
apt install -y expect
echo "Installing libgraphviz-dev"
apt install -y libgraphviz-dev
echo "Installing cron"
apt install -y cron
echo "Installing default-jdk"
apt install -y default-jdk
echo "Installing ltrace"
apt install -y ltrace
echo "Installing binutils"
apt install -y binutils
echo "Installing alien"
apt install -y alien
echo "Installing libaio1"
apt install -y libaio1
echo "Installing supervisor"
apt install supervisor
echo "Installing websocat"
wget https://github.com/vi/websocat/releases/download/v1.8.0/websocat_amd64-linux -O websocat
chmod +x websocat
mv websocat /usr/local/bin/

#apt install -y websocat

# for now don't install binwalk as it messes up the python environment, consider adding virtualenv
#echo "Installing binwalk"
#apt install -y binwalk

echo "Installing ncdu"
apt install -y ncdu
echo "Installing vim"
apt install -y vim
echo "Installing zip"
apt install -y zip
echo "Installing unzip"
apt install -y unzip
echo "Installing unixodbc-dev"
apt install -y unixodbc-dev
echo "Installing sqlite3"
apt install -y sqlite3
echo "Installing python3.10-venv"
apt install -y python3.10-venv
echo "Installing redis-server"
apt install -y redis-server
echo "Installing rabbitmq-server"
apt install -y rabbitmq-server
echo "Installing ca-certificates"
apt install -y ca-certificates
echo "Installing curl"
apt install -y curl
echo "Installing gnupg"
apt install -y gnupg
echo "Installing lsb-release"
apt install -y lsb-release

echo "Installing iptables"
apt install -y iptables

echo "Installing smbclient"
apt install -y smbclient

echo "Installing tree"
apt install -y tree

echo "Installing snapd"
apt install -y snapd

echo "install kinit"
apt install -y krb5-user

echo "Installing nvtop"
snap install nvtop

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
echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config
mkdir ~/.ssh
touch ~/.ssh/environment

# ssh connection immediately disconnects after session start with exit code 254:
# https://unix.stackexchange.com/questions/148714/cant-ssh-connection-terminates-immediately-with-exit-status-254
sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config