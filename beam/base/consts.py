import os
from argparse import Namespace

base_path = os.path.join(os.path.expanduser('~'), 'beam_data')

base_paths = Namespace(projects_data=os.path.join(base_path, 'projects', 'data'),
                       projects_experiments=os.path.join(base_path, 'projects', 'experiment'),
                       deepspeed_data=os.path.join(base_path, 'projects', 'deepspeed'),
                       logs=os.path.join(base_path, 'logs'),
                       autobeam_cache=os.path.join(base_path, '.cache', 'autobeam'),
                       global_config=os.path.join(base_path, 'config.pkl'),
                       docker_config_dir=os.path.join(base_path, '.docker'),
                       projects_hpo=os.path.join(base_path, 'projects', 'hpo'),
                       )

tmp_paths = Namespace(beam_kv_store='/tmp/beam/kv_store',
                      code_repos='/tmp/beam/repos',)
