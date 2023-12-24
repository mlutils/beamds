# add the beam message

cp /workspace/install/motd /etc/motd
echo "cat /etc/motd" >> /root/.bashrc

pip install jupyterlab-tensorboard-pro notebook-shim async_lru

jupyter nbextension enable --py widgetsnbextension

jupyter-lab --generate-config
echo "c.ServerApp.notebook_dir = '/home/'" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_remote_access = True" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py

echo "c.LabServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py

# echo "c.ServerApp.jpserver_extensions = ['/workspace/beamds/notebooks/beam_setup.py']" >> /root/.jupyter/jupyter_lab_config.py

# language servers -experimental for future use
# see https://jupyterlab-lsp.readthedocs.io/en/latest/Language%20Servers.html
pip install -U jedi-language-server
apt install -y nodejs
apt install -y npm

