# add the beam message
cp /workspace/configuration/motd /etc/motd
# todo: make sure it is not displayed twice
echo "cat /etc/motd" >> /root/.bashrc

pip install jupyterlab-tensorboard-pro notebook-shim async_lru
jupyter nbextension enable --py widgetsnbextension

#jupyter-lab --generate-config
#echo "c.ServerApp.notebook_dir = '/home/'" >> /root/.jupyter/jupyter_lab_config.py
#echo "c.ServerApp.allow_remote_access = True" >> /root/.jupyter/jupyter_lab_config.py
#echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py
#echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py
#echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py
#echo "c.LabServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py



su - "$USER_NAME" << EOF
jupyter-lab --generate-config
CONF_FILE="$USER_HOME_DIR/.jupyter/jupyter_lab_config.py"
echo "c.ServerApp.root_dir = '/home/'" >> "$CONF_FILE"
echo "c.ServerApp.allow_remote_access = True" >> "$CONF_FILE"
echo "c.ServerApp.ip = '0.0.0.0'" >> "$CONF_FILE"
echo "c.IdentityProvider.token = ''" >> "$CONF_FILE"
echo "c.ServerApp.allow_root = False" >> "$CONF_FILE"
echo "c.LabServerApp.open_browser = False" >> "$CONF_FILE"
EOF

# language servers -experimental for future use
# see https://jupyterlab-lsp.readthedocs.io/en/latest/Language%20Servers.html

# moved to requirements.txt for now
#pip install -U jedi-language-server

