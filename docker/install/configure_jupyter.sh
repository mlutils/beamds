# add the beam message

cp /workspace/install/motd /etc/motd
echo "cat /etc/motd" >> /root/.bashrc

jupyter nbextension enable --py widgetsnbextension

jupyter-lab --generate-config
echo "c.ServerApp.notebook_dir = '/home/'" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_remote_access = True" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.jpserver_extensions = ['/workspace/beamds/notebooks/beam_setup.py']" >> /root/.jupyter/jupyter_lab_config.py