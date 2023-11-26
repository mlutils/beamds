
# for rayL pynvml does not work well with gpustats
pip uninstall --yes pynvml
pip install -U nvidia-ml-py


echo "cd \$USER_HOME" >> ~/.bashrc

