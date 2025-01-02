from beam.orchestration import (RayClusterConfig, RayCluster, RnDCluster, BeamManagerConfig,
                                RnDClusterConfig, ServeClusterConfig, ServeCluster, BeamManager)
from beam.resources import resource, this_dir
from beam.logging import beam_logger as logger
import os
import sys


def main():

    ## config = resource('/home/dayosupp/projects/beamds/examples/orchestration/orchestration_serve_cluster.yaml').read()
    elasticsearch_config = ServeClusterConfig()
    elasticsearch_config = ServeClusterConfig('/home/dayosupp/projects/beamds/examples/orchestration/orchestration_elasticsearch.yaml', **elasticsearch_config)

    logger.info(str(elasticsearch_config))
    manager = BeamManager(elasticsearch_config)
    manager.launch_serve_cluster(elasticsearch_config)


    elasticsearch_host = manager.get_cluster_service(elasticsearch_config.deployment_name, elasticsearch_config.project_name, port=9200)

    kibana_config = ServeClusterConfig()
    kibana_config = ServeClusterConfig('/home/dayosupp/projects/beamds/examples/orchestration/orchestration_kibana.yaml', **kibana_config)
    kibana_config.update({'entrypoint_envs': {'ELASTICSEARCH_HOST': elasticsearch_host}})
    logger.info(str(kibana_config))
    manager = BeamManager(kibana_config)
    manager.launch_serve_cluster(kibana_config)

    logstash_config = ServeClusterConfig()
    logstash_config = ServeClusterConfig('/home/dayosupp/projects/beamds/examples/orchestration/orchestration_logstash.yaml', **logstash_config)
    logstash_config.update({'entrypoint_envs': {'ELASTICSEARCH_HOST': elasticsearch_host}})
    logger.info(str(logstash_config))
    manager = BeamManager(logstash_config)
    manager.launch_serve_cluster(logstash_config)


if __name__ == '__main__':
    main()


    # manager.launch_ray_cluster('/home/dayosupp/projects/beamds/examples/orchestration_raydeploy.yaml')
    # manager.launch_serve_cluster('/home/dayosupp/projects/beamds/examples/orchestration/orchestration_serve_cluster.yaml')
    # manager.launch_cron_job('/home/dayosupp/projects/beamds/examples/orchestration_beamdemo.yaml')
    # manager.launch_job('/home/dayosupp/projects/beamds/examples/orchestration_beamdemo.yaml')
    # manager.launch_rnd_cluster('/home/dayosupp/projects/beamds/examples/orchestration/orchestration_rnd_cluster.yaml')
    # print(manager.info())
    # manager.monitor_thread()
    # manager.retrieve_cluster_logs('rnd_cluster_name')


# script_dir = os.path.dirname(os.path.realpath(__file__))
# conf_path = resource(os.path.join(script_dir, 'orchestration_manager.yaml')).str
# config = BeamManagerConfig(conf_path)

# logger.info(f"hello world")
# logger.info(f"API URL: {config.api_url}")
# logger.info(f"API Token: {config.api_token}")

# manager = BeamManager(deployment=None, clusters=config['clusters'], config=config)

# manager.monitor_thread()

# config = ServeClusterConfig()

    # logger.info(f"API URL: {config.api_url}")
    # logger.info(f"API Token: {config.api_token}")
    # logger.info("deploy manager with config:")
    # config.update({'project_name': 'dev',
    #     'deployment_name': 'elasticsearch',
    #     'labels': {'app': 'elk'},
    #     'alg': '/tmp/elasticsearch',
    #     'debug_sleep': False})
