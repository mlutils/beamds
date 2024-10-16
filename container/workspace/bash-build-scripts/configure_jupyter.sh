# add the beam message
cp /workspace/configuration/motd /etc/motd

pip install jupyterlab-tensorboard-pro notebook-shim async_lru
# jupyter nbextension enable --py widgetsnbextension

rm -f /root/.jupyter/jupyter_lab_config.py

jupyter-lab --generate-config
echo "c.ServerApp.notebook_dir = '/home/'" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_remote_access = True" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py
echo "c.LabServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py


#
#if [ "$USER_NAME" != "root" ]; then
#    su - "$USER_NAME" << 'EOF'
#       rm -f ~/.jupyter/jupyter_lab_config.py && jupyter-lab --generate-config
#       echo "c.ServerApp.root_dir = '/home/$USER'" >> ~/.jupyter/jupyter_lab_config.py
#       echo "c.ServerApp.allow_remote_access = True" >> ~/.jupyter/jupyter_lab_config.py
#       echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py
#       echo "c.ServerApp.allow_root = False" >> ~/.jupyter/jupyter_lab_config.py
#       echo "c.LabApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py
#EOF
#fi


# language servers -experimental for future use
# see https://jupyterlab-lsp.readthedocs.io/en/latest/Language%20Servers.html

# moved to requirements.txt for now
#pip install -U jedi-language-server