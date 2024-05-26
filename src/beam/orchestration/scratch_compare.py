# deploy.py
class BeamDeploy(Processor):
    # ... (rest of your initialization code remains the same)

    def launch(self, replicas=None):
        if replicas is None:
            replicas = self.replicas

        if self.check_project_exists is True:
            self.k8s.create_project(self.namespace)

        if self.create_service_account is True:
            self.k8s.create_service_account(self.service_account_name, self.namespace)
        else:
            self.service_account_name = 'default'
            logger.info(f"using default service account '{self.service_account_name}' in namespace '{self.namespace}'.")

        if self.storage_configs:
            for storage_config in self.storage_configs:
                try:
                    self.k8s.core_v1_api.read_namespaced_persistent_volume_claim(name=storage_config.pvc_name,
                                                                                 namespace=self.namespace)
                    logger.info(f"PVC '{storage_config.pvc_name}' already exists in namespace '{self.namespace}'.")
                except ApiException as e:
                    if e.status == 404 and storage_config.create_pvc:
                        logger.info(f"Creating PVC for storage config: {storage_config.pvc_name}")
                        self.k8s.create_pvc(
                            pvc_name=storage_config.pvc_name,
                            pvc_size=storage_config.pvc_size.as_str,
                            pvc_access_mode=storage_config.pvc_access_mode,
                            namespace=self.namespace
                        )
                    else:
                        logger.info(f"Skipping PVC creation for: {storage_config.pvc_name} as create_pvc is False")

        enabled_memory_storages = [config for config in self.memory_storage_configs if config.enabled]

        for svc_config in self.service_configs:
            service_name = f"{self.deployment_name}-{svc_config.service_name}-{svc_config.port}"
            self.k8s.create_service(
                base_name=f"{self.deployment_name}-{svc_config.service_name}-{svc_config.port}",
                namespace=self.namespace,
                ports=[svc_config.port],
                labels=self.labels,
                service_type=svc_config.service_type
            )

            if svc_config.create_route:
                self.k8s.create_route(
                    service_name=service_name,
                    namespace=self.namespace,
                    protocol=svc_config.route_protocol,
                    port=svc_config.port
                )

            if svc_config.create_ingress:
                self.k8s.create_ingress(
                    service_configs=[svc_config],
                )

        if self.user_idm_configs:
            self.k8s.create_role_bindings(self.user_idm_configs)

        if self.use_scc is True:
            self.k8s.add_scc_to_service_account(self.service_account_name, self.namespace, self.scc_name)

        extracted_ports = [svc_config.port for svc_config in self.service_configs]

        if self.enable_ray_ports is True:
            for ray_ports_config in self.ray_ports_configs:
                extracted_ports += [ray_port for ray_port in ray_ports_config.ray_ports]

        deployment = self.k8s.create_deployment(
            image_name=self.image_name,
            labels=self.labels,
            deployment_name=self.deployment_name,
            namespace=self.namespace,
            project_name=self.project_name,
            replicas=replicas,
            ports=extracted_ports,
            create_service_account=self.create_service_account,
            service_account_name=self.service_account_name,
            storage_configs=self.storage_configs,
            memory_storage_configs=enabled_memory_storages,
            use_node_selector=self.node_selector,
            node_selector=self.node_selector,
            cpu_requests=self.cpu_requests,
            cpu_limits=self.cpu_limits,
            memory_requests=self.memory_requests,
            memory_limits=self.memory_limits,
            use_gpu=self.use_gpu,
            gpu_requests=self.gpu_requests,
            gpu_limits=self.gpu_limits,
            security_context_config=self.security_context_config,
            entrypoint_args=self.entrypoint_args,
            entrypoint_envs=self.entrypoint_envs,
        )

        pod_infos = self.k8s.apply_deployment(deployment, namespace=self.namespace)

        beam_pod_instances = []

        if isinstance(pod_infos, list) and pod_infos:
            for pod_info in pod_infos:
                pod_name = getattr(pod_info, 'name', None)
                if pod_name:
                    actual_pod_info = self.k8s.get_pod_info(pod_name, self.namespace)
                    beam_pod_instance = BeamPod(pod_infos=[actual_pod_info], namespace=self.namespace, k8s=self.k8s)
                    beam_pod_instances.append(beam_pod_instance)
                else:
                    logger.warning("PodInfo object does not have a 'name' attribute.")

        elif pod_infos and hasattr(pod_infos, 'name'):
            pod_name = pod_infos.name
            actual_pod_info = self.k8s.get_pod_info(pod_name, self.namespace)
            return BeamPod(pod_infos=[actual_pod_info], namespace=self.namespace, k8s=self.k8s)

        if not beam_pod_instances:
            logger.error("Failed to apply deployment or no pods were returned.")
            return None

        return beam_pod_instances if len(beam_pod_instances) > 1 else beam_pod_instances[0]

    def generate_beam_pod(self, pod_infos):
        return BeamPod(pod_infos=pod_infos, k8s=self.k8s, namespace=self.namespace)

    def delete_deployment(self):
        try:
            self.k8s.apps_v1_api.delete_namespaced_deployment(
                name=self.deployment.metadata.name,
                namespace=self.deployment.metadata.namespace,
                body=client.V1DeleteOptions()
            )
            logger.info(f"Deleted deployment '{self.deployment.metadata.name}' "
                        f"from namespace '{self.deployment.metadata.namespace}'.")
        except ApiException as e:
            logger.error(f"Error deleting deployment '{self
