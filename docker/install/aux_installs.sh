
# for rayL pynvml does not work well with gpustats
pip uninstall --yes pynvml
pip install -U nvidia-ml-py

apt install -y libopenmpi-dev

git config --global http.sslVerify false

echo "cd $USER_HOME" >> ~/.bashrc

