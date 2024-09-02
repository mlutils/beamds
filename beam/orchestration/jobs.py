from .cluster import BeamCluster
from .k8s import BeamK8S
from .deploy import BeamDeploy
from ..logging import beam_logger as logger
from .pod import BeamPod
from .config import JobConfig, CronJobConfig


class BeamJob(BeamCluster):
    # Handles Job deployment, monitoring, logs, and interaction via k8s API

    def __init__(self, config, job_name=None, pods=None):
        super(BeamJob, self).__init__(deployment=None, job_name=job_name, config=config, pods=pods)

    @classmethod
    def deploy_job(cls, config, k8s=None):
        """
        Deploy a Job to the cluster.
        """
        if k8s is None:
            k8s = BeamK8S(
                api_url=config['api_url'],
                api_token=config['api_token'],
                project_name=config['project_name'],
                namespace=config['project_name'],
            )

        job_config = JobConfig(**config)
        job = BeamDeploy(job_config, k8s)

        # Use the launch_job method from BeamDeploy to deploy the Job
        pods = job.launch_job(job_name=job_config.name)

        if isinstance(pods, BeamPod):
            pods = [pods]

        job_instance = cls(config=config, job_name=config['job_name'], pods=pods)
        return job_instance

    def delete_job(self):
        """
        Delete the Job.
        """
        try:
            self.k8s.delete_job(self.job.metadata.name, self.config['project_name'])
            logger.info(f"Job {self.job.metadata.name} deleted successfully.")
        except Exception as e:
            logger.error(f"Error occurred while deleting the Job: {str(e)}")


class BeamCronJob(BeamCluster):
    # Handles CronJob deployment, monitoring, logs, and interaction via k8s API

    def __init__(self, config, cron_job_name=None, pods=None):
        super(BeamCronJob, self).__init__(deployment=None, config=config, cron_job_name=cron_job_name, pods=pods)

    @classmethod
    def deploy_cron_job(cls, config, k8s=None):
        """
        Deploy a CronJob to the cluster.
        """
        if k8s is None:
            k8s = BeamK8S(
                api_url=config['api_url'],
                api_token=config['api_token'],
                project_name=config['project_name'],
                namespace=config['project_name'],
            )

        cron_job_config = CronJobConfig(**config)
        cron_job = BeamDeploy(cron_job_config, k8s)

        # Use the launch_cron_job method from BeamDeploy to deploy the CronJob
        pods = cron_job.launch_cron_job()

        if isinstance(pods, BeamPod):
            pods = [pods]

        cron_job_instance = cls(config=config, pods=pods)
        return cron_job_instance

    def delete_cron_job(self):
        """
        Delete the CronJob.
        """
        try:
            self.k8s.delete_cron_job(self.deployment.metadata.name, self.config['project_name'])
            logger.info(f"CronJob {self.deployment.metadata.name} deleted successfully.")
        except Exception as e:
            logger.error(f"Error occurred while deleting the CronJob: {str(e)}")