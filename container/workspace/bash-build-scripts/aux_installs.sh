
# for rayL pynvml does not work well with gpustats
pip uninstall --yes pynvml
pip install -U nvidia-ml-py
#apt install -y libopenmpi-dev
#pip install mpi4py
# install mpi4py
#apt-get remove --purge -y libopenmpi-dev openmpi-bin openmpi-common
#apt-get update
#apt-get install -y libopenmpi-dev openmpi-bin
#pip install setuptools==42.0.0
#pip install --no-build-isolation  mpi4py
#pip install -U setuptools
python -c "from opencv_fixer import AutoFix; AutoFix()"
git config --global http.sslVerify false
echo "cd /home" >> ~/.bashrc

# make a default home directory if does not exist
mkdir -p /home

