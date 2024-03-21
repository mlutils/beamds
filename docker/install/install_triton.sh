RELEASE=$1

apt-get update         && apt-get install -y ca-certificates curl gnupg         && install -m 0755 -d /etc/apt/keyrings         && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg         && chmod a+r /etc/apt/keyrings/docker.gpg         && echo             "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu             "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" |             tee /etc/apt/sources.list.d/docker.list > /dev/null         && apt-get update         && apt-get install -y docker.io docker-buildx-plugin

apt-get update     && apt-get install -y --no-install-recommends  ca-certificates  autoconf  automake  build-essential  git  gperf  libre2-dev  libssl-dev  libtool  libcurl4-openssl-dev  libb64-dev  libgoogle-perftools-dev  patchelf  python3-dev  python3-pip  python3-setuptools  rapidjson-dev  scons  software-properties-common  pkg-config  unzip  wget  zlib1g-dev  libarchive-dev  libxml2-dev  libnuma-dev  wget     && rm -rf /var/lib/apt/lists/*


wget -O /tmp/boost.tar.gz         https://sourceforge.net/projects/boost/files/boost/1.84.0/boost_1_84_0.tar.gz/download &&     (cd /tmp && tar xzf boost.tar.gz) &&     cd /tmp/boost_1_84_0 && ./bootstrap.sh --prefix=/usr && ./b2 install &&     mv /tmp/boost_1_84_0/boost /usr/include/boost


apt update -q=2 \
    && apt install -y gpg wget \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update -q=2 \
    && apt-get install -y --no-install-recommends cmake=3.27.7* cmake-data=3.27.7*


apt-key del 7fa2af80
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get install -y datacenter-gpu-manager


mkdir /workspace/triton/
cd /workspace/triton/
mkdir build
git clone https://github.com/triton-inference-server/server.git
cd server
git checkout r$RELEASE
./build.py -v --no-container-build --build-dir=/workspace/triton/build --enable-all --backend onnxruntime --backend pytorch