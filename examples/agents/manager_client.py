import asyncio
import random
import uuid
import datetime
import pathlib

from python_a2a import AgentNetwork, A2AClient, Flow
from beam import logger

from examples.agents.config import AgentConfig


class FlowManager:
    def __init__(self, peer_urls, timeout=120, storage_dir="./storage"):
        self.storage_dir = pathlib.Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Setup network
        self.network = AgentNetwork()
        for url in peer_urls:
            peer_name = f"peer-{uuid.uuid4().hex[:6]}"
            client = A2AClient(url, timeout=timeout)
            self.network.add(peer_name, client)

    async def run_flow(self, topic: str):
        network = self.network
        flow = Flow(agent_network=network, name="Paper-Authoring")

        # Step 1: Brainstorm
        brainstorm_step = flow.parallel()
        for agent_info in network.list_agents():
            brainstorm_step.ask(agent_info["name"], f"Brainstorm {topic}")
        brainstorm_step.end_parallel()

        def merge_brainstorms(results, ctx):
            return "\n\n".join(results.values())
        brainstorm_ideas = flow.execute_function(merge_brainstorms, brainstorm_step)

        # Step 2: Claims
        lead = random.choice(network.list_agents())["name"]
        outline = flow.ask(lead, "Claims {latest_result}")

        # Step 3: Verify
        for agent_info in network.list_agents():
            if agent_info["name"] != lead:
                flow.ask(agent_info["name"], "Verify {latest_result}")

        def merge_verifications(results, ctx):
            return "\n\n".join(results.values())
        vetted_outline = flow.execute_function(merge_verifications)

        # Step 4: Draft
        draft_md = flow.ask(lead, "Draft {latest_result}")

        # Step 5: Audit
        audit_step = flow.parallel()
        for agent_info in network.list_agents():
            audit_step.ask(agent_info["name"], "Audit {latest_result}")
        audit_step.end_parallel()

        def compile_final(audit_results, ctx):
            reviews = "\n\n".join(audit_results.values())
            return f"# Final Draft\n\n{draft_md.value}\n\n---\n\n## Peer Reviews\n{reviews}"
        final = flow.execute_function(compile_final, audit_step)

        logger.debug("Running the flow...")
        result = await flow.run({"topic": topic})
        logger.info(f"Final paper: {result}")

        fname = f"{datetime.date.today()}_{uuid.uuid4().hex[:6]}.md"
        with open(self.storage_dir / fname, "w") as f:
            f.write(result)

        return result


if __name__ == "__main__":

    hparams = AgentConfig()
    manager = FlowManager(hparams.peers, timeout=hparams.timeout, storage_dir=hparams.storage_dir)
    asyncio.run(manager.run_flow(hparams.topic))
