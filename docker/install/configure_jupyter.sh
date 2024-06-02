# add the beam message
cp /workspace/beam_image/runs/motd /etc/motd
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
su - beam <<EOF
jupyter-lab --generate-config
echo "c.ServerApp.notebook_dir = '/home/'" >> /home/beam/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_remote_access = True" >> /home/beam/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.ip = '0.0.0.0'" >> /home/beam/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.token = ''" >> /home/beam/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_root = False" >> /home/beam/.jupyter/jupyter_lab_config.py
echo "c.LabServerApp.open_browser = False" >> /home/beam/.jupyter/jupyter_lab_config.py
EOF
# language servers -experimental for future use
# see https://jupyterlab-lsp.readthedocs.io/en/latest/Language%20Servers.html

# moved to requirements.txt for now
#pip install -U jedi-language-server

