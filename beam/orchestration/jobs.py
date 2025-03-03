from typing import List, Union, Dict
from .k8s import BeamK8S
from ..base import BeamBase
from .deploy import BeamDeploy
from ..logging import beam_logger as logger
from .pod import BeamPod
from .config import JobConfig, CronJobConfig
from .dataclasses import (ServiceConfig, StorageConfig, RayPortsConfig, UserIdmConfig,
                          MemoryStorageConfig, SecurityContextConfig)

class BeamJobManager(BeamBase):
    def __init__(self, deployment: Union[BeamDeploy, Dict[str, BeamDeploy], None], config, pods: List[BeamPod] = None,
                 *args, **kwargs):
        super().__init__(*args, hparams=config, **kwargs)
        self.pods = pods
        self.deployment = deployment
        self.k8s = BeamK8S(
            api_url=self.hparams.api_url,
            api_token=self.hparams.api_token,
            project_name=self.hparams.project_name,
            namespace=self.hparams.project_name,
        )

        self.security_context_config = SecurityContextConfig(**config.get('security_context_config', {}))
        self.memory_storage_configs = [MemoryStorageConfig(**v) for v in config.get('memory_storage_configs', [])]
        self.service_configs = [ServiceConfig(**v) for v in config.get('service_configs', [])]
        self.storage_configs = [StorageConfig(**v) for v in config.get('storage_configs', [])]
        self.ray_ports_configs = [RayPortsConfig(**v) for v in config.get('ray_ports_configs', [])]
        self.user_idm_configs = [UserIdmConfig(**v) for v in config.get('user_idm_configs', [])]
        self.entrypoint_args = config['entrypoint_args']
        self.entrypoint_envs = config['entrypoint_envs']


class BeamCronJob(BeamJobManager):
    def __init__(self, config, *args, k8s=None, **kwargs):
        super().__init__(None, config,*args,  k8s=k8s, **kwargs)
        self.manager = BeamJobManager(self.k8s)

    @classmethod
    def _deploy_and_launch(cls, config, k8s=None):

        if k8s is None:
            k8s = BeamK8S(
                api_url=config['api_url'],
                api_token=config['api_token'],
                project_name=config['project_name'],
                namespace=config['project_name'],
            )

        cron_job =  BeamDeploy(config, k8s)
        # pods = cron_job.launch_cron_job(job_schedule=config.get('job_schedule'))
        pods = cron_job.launch_cron_job()
        # return cls(config, k8s), pods
        return cls(config, k8s, pods)

    @classmethod
    def deploy_cron_job(cls, config, k8s=None):
        return cls._deploy_and_launch(config=config, k8s=k8s)

    def delete(self):
        self.manager.delete_cron_job(self.hparams.cron_job_name, self.hparams.project_name)

    def monitor(self):
        self.manager.monitor_cron_job(self.hparams.cron_job_name, self.hparams.project_name)


class BeamJob:
    def __init__(self, config, k8s=None):
        self.hparams = JobConfig(**config)

        if k8s is None:
            self.k8s = k8s or BeamK8S(
                api_url=config['api_url'],
                api_token=config['api_token'],
                project_name=config['project_name'],
                namespace=config['project_name'],
            )

        self.manager = BeamJobManager(self.k8s)

    @classmethod
    def _deploy(cls, config, k8s=None):
        job = cls(config, k8s)
        pods = job.manager.deploy_job(job.config)
        return cls(config, k8s), pods

    def delete(self):
        self.manager.delete_job(self.hparams.job_name, self.hparams.project_name)

    def monitor(self):
        self.manager.monitor_job(self.hparams.job_name, self.hparams.project_name)

# class BeamJob:
#     Handles Job deployment, monitoring, logs, and interaction via k8s API
#
#     def __init__(self, config, job_name=None, pods=None, *args, **kwargs):
#         super(BeamJob, self).__init__(deployment=None, job_name=job_name, config=config,
#                                       pods=pods, *args, **kwargs)
#
#     @classmethod
#     def deploy_job(cls, config, k8s=None):
#         """
#         Deploy a Job to the cluster.
#         """
#         if k8s is None:
#             k8s = BeamK8S(
#                 api_url=config['api_url'],
#                 api_token=config['api_token'],
#                 project_name=config['project_name'],
#                 namespace=config['project_name'],
#             )
#
#         cron_job_config = CronJobConfig(**config)
#         cron_job = BeamDeploy(cron_job_config, k8s)
#
#         config = JobConfig(**config)
#         job = BeamDeploy(config, k8s)
#
#         # Use the launch_job method from BeamDeploy to deploy the Job
#         pods = job.launch_job()
#
#         if isinstance(pods, BeamPod):
#             pods = [pods]
#
#         job_instance = cls(config=config, job_name=config['job_name'], pods=pods)
#         return job_instance
#
#     def delete_job(self):
#         """
#         Delete the Job.
#         """
#         try:
#             self.k8s.delete_job(self.job.metadata.name, self.hparams['project_name'])
#             logger.info(f"Job {self.job.metadata.name} deleted successfully.")
#         except Exception as e:
#             logger.error(f"Error occurred while deleting the Job: {str(e)}")
#
#     def monitor_job(self):
#         """
#         Monitor the Job status and retrieve logs upon completion.
#         """
#         try:
#             # Monitor the job status
#             self.k8s.monitor_job(job_name=self.hparams['job_name'], namespace=self.hparams['project_name'])
#
#             # Once completed, fetch and print logs
#             logs = self.k8s.get_job_logs(job_name=self.hparams['job_name'], namespace=self.hparams['project_name'])
#             if logs:
#                 logger.info(f"Logs for Job '{self.hparams['job_name']}':\n{logs}")
#         except Exception as e:
#             logger.error(f"Failed to monitor job '{self.hparams['job_name']}': {str(e)}")




# class BeamCronJob(BeamCluster):
#     # Handles CronJob deployment, monitoring, logs, and interaction via k8s API
#
#     def __init__(self, config, cron_job_name=None, pods=None, *args, **kwargs):
#         super(BeamCronJob, self).__init__(deployment=None, config=config, cron_job_name=cron_job_name,
#                                           pods=pods, *args, **kwargs)
#
#     @classmethod
#     def deploy_cron_job(cls, config, k8s=None):
#         """
#         Deploy a CronJob to the cluster.
#         """
#         if k8s is None:
#             k8s = BeamK8S(
#                 api_url=config['api_url'],
#                 api_token=config['api_token'],
#                 project_name=config['project_name'],
#                 namespace=config['project_name'],
#             )
#
#         cron_job_config = CronJobConfig(**config)
#
#         cron_job = cls(cron_job_config, k8s)
#
#         # Use the launch_cron_job method from BeamDeploy to deploy the CronJob
#         pods = cron_job.launch_cron_job()
#
#         if isinstance(pods, BeamPod):
#             pods = [pods]
#
#         cron_job_instance = cls(config=config, pods=pods)
#         return cron_job_instance
#
#     def delete_cron_job(self):
#         """
#         Delete the CronJob.
#         """
#         try:
#             self.k8s.delete_cron_job(self.deployment.metadata.name, self.hparams['project_name'])
#             logger.info(f"CronJob {self.deployment.metadata.name} deleted successfully.")
#         except Exception as e:
#             logger.error(f"Error occurred while deleting the CronJob: {str(e)}")
#
#     def monitor_cron_job(self):
#         """
#         Monitor the CronJob status and retrieve logs of any triggered jobs.
#         """
#         try:
#             # Monitor the cron job's spawned jobs
#             self.k8s.monitor_cron_job(cron_job_name=self.hparams['cron_job_name'], namespace=self.hparams['project_name'])
#
#             # Fetch logs from jobs spawned by the cron job
#             jobs = self.k8s.get_pods_by_label({'cronjob-name': self.hparams['cron_job_name']}, self.hparams['project_name'])
#             if jobs:
#                 for job in jobs:
#                     logs = self.k8s.get_job_logs(job.metadata.name, namespace=self.hparams['project_name'])
#                     if logs:
#                         logger.info(f"Logs for CronJob '{self.hparams['cron_job_name']}':\n{logs}")
#         except Exception as e:
#             logger.error(f"Failed to monitor cron job '{self.hparams['cron_job_name']}': {str(e)}")
#
#     def get_cron_job_logs(self):
#         """
#         Retrieve logs of the last job triggered by this cron job.
#         """
#         return self.k8s.get_job_logs(self.cron_job_name, self.namespace)

    # def deploy_job(self, config):
    #     """
    #     Deploy a Job to the cluster.
    #     """
    #     try:
    #         job = BeamDeploy(config, self.k8s)
    #         pods = job.launch_job()
    #
    #         # Ensure pods is always a list of BeamPod objects
    #         if isinstance(pods, BeamPod):
    #             pods = [pods]
    #
    #         logger.info(f"Job '{config.job_name}' deployed successfully.")
    #         return pods
    #     except Exception as e:
    #         logger.error(f"Failed to deploy Job '{config.job_name}': {str(e)}")
    #         raise
    #
    # def deploy_cron_job(self, config):
    #     """
    #     Deploy a CronJob to the cluster.
    #     """
    #     try:
    #         cron_job = BeamDeploy(config, self.k8s)
    #         pods = cron_job.launch_cron_job()
    #
    #         # Ensure pods is always a list of BeamPod objects
    #         if isinstance(pods, BeamPod):
    #             pods = [pods]
    #
    #         logger.info(f"CronJob '{config.cron_job_name}' deployed successfully.")
    #         return pods
    #     except Exception as e:
    #         logger.error(f"Failed to deploy CronJob '{config.cron_job_name}': {str(e)}")
    #         raise
    #
    # def delete_job(self, job_name: str, namespace: str):
    #     """
    #     Delete a Job.
    #     """
    #     try:
    #         self.k8s.delete_job(job_name, namespace)
    #         logger.info(f"Job '{job_name}' deleted successfully.")
    #     except Exception as e:
    #         logger.error(f"Failed to delete Job '{job_name}': {str(e)}")
    #
    # def delete_cron_job(self, cron_job_name: str, namespace: str):
    #     """
    #     Delete a CronJob.
    #     """
    #     try:
    #         self.k8s.delete_cron_job(cron_job_name, namespace)
    #         logger.info(f"CronJob '{cron_job_name}' deleted successfully.")
    #     except Exception as e:
    #         logger.error(f"Failed to delete CronJob '{cron_job_name}': {str(e)}")
    #
    # def monitor_job(self, job_name: str, namespace: str):
    #     """
    #     Monitor the status of a Job.
    #     """
    #     try:
    #         self.k8s.monitor_job(job_name, namespace)
    #     except Exception as e:
    #         logger.error(f"Failed to monitor Job '{job_name}': {str(e)}")
    #
    # def monitor_cron_job(self, cron_job_name: str, namespace: str):
    #     """
    #     Monitor the status of a CronJob.
    #     """
    #     try:
    #         self.k8s.monitor_cron_job(cron_job_name, namespace)
    #     except Exception as e:
    #         logger.error(f"Failed to monitor CronJob '{cron_job_name}': {str(e)}")