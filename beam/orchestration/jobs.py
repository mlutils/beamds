from .cluster import BeamCluster
from .k8s import BeamK8S
from .deploy import BeamDeploy
from ..logging import beam_logger as logger
from .pod import BeamPod
from .config import JobConfig, CronJobConfig


class BeamJob(BeamCluster):
    # Handles Job deployment, monitoring, logs, and interaction via k8s API

    def __init__(self, config, job_name=None, pods=None, *args, **kwargs):
        super(BeamJob, self).__init__(deployment=None, job_name=job_name, config=config,
                                      pods=pods, *args, **kwargs)

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

        cron_job_config = CronJobConfig(**config)
        cron_job = BeamDeploy(cron_job_config, k8s)

        config = JobConfig(**config)
        job = BeamDeploy(config, k8s)

        # Use the launch_job method from BeamDeploy to deploy the Job
        pods = job.launch_job()

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

    def monitor_job(self):
        """
        Monitor the Job status and retrieve logs upon completion.
        """
        try:
            # Monitor the job status
            self.k8s.monitor_job(job_name=self.config['job_name'], namespace=self.config['project_name'])

            # Once completed, fetch and print logs
            logs = self.k8s.get_job_logs(job_name=self.config['job_name'], namespace=self.config['project_name'])
            if logs:
                logger.info(f"Logs for Job '{self.config['job_name']}':\n{logs}")
        except Exception as e:
            logger.error(f"Failed to monitor job '{self.config['job_name']}': {str(e)}")


class BeamCronJob(BeamCluster):
    # Handles CronJob deployment, monitoring, logs, and interaction via k8s API

    def __init__(self, config, cron_job_name=None, pods=None, *args, **kwargs):
        super(BeamCronJob, self).__init__(deployment=None, config=config, cron_job_name=cron_job_name,
                                          pods=pods, *args, **kwargs)

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

    def monitor_cron_job(self):
        """
        Monitor the CronJob status and retrieve logs of any triggered jobs.
        """
        try:
            # Monitor the cron job's spawned jobs
            self.k8s.monitor_cron_job(cron_job_name=self.config['cron_job_name'], namespace=self.config['project_name'])

            # Fetch logs from jobs spawned by the cron job
            jobs = self.k8s.get_pods_by_label({'cronjob-name': self.config['cron_job_name']}, self.config['project_name'])
            if jobs:
                for job in jobs:
                    logs = self.k8s.get_job_logs(job.metadata.name, namespace=self.config['project_name'])
                    if logs:
                        logger.info(f"Logs for CronJob '{self.config['cron_job_name']}':\n{logs}")
        except Exception as e:
            logger.error(f"Failed to monitor cron job '{self.config['cron_job_name']}': {str(e)}")

    def get_cron_job_logs(self):
        """
        Retrieve logs of the last job triggered by this cron job.
        """
        return self.k8s.get_job_logs(self.cron_job_name, self.namespace)