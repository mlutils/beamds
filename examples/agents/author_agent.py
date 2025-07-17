import os
import asyncio
import uuid
import random
from typing import List

from python_a2a import (
    A2AServer, agent, skill, TaskStatus, TaskState,
    run_server, AgentNetwork, Flow, A2AClient
)
from python_a2a.agent_flow.storage.workflow_storage import FileWorkflowStorage
import pathlib, datetime, json, uuid


from pydantic_ai import Agent as LlmAgent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

from ai21_models_gateway.utils.ai21 import create_ai21_access_token
from beam.utils import cached_property
from beam import BeamParam, BeamConfig
from beam import logger


class AgentConfig(BeamConfig):
    parameters = [
        BeamParam('port', int, 5000, 'Port to run the A2A server on'),
        BeamParam('model', str, 'claude-3-7-sonnet', 'Model ID to use for the agent'),
        BeamParam('peers', List[str], [], 'List of peer agent URLs to connect to'),
        BeamParam("debug", bool, False, "Flask debug / auto-reload"),
        BeamParam("storage_dir", str, "./storage", "Where to persist artefacts"),
        BeamParam("timeout", int, 120, "HTTP timeout (s) for peer calls")
    ]


class AI21Model(OpenAIModel):
    """
    An AI21-compatible model gateway exposing the OpenAI API.
    """
    gateway = "https://tools.algo-agents.ai21.com/models-gateway"

    def __init__(self, model, *args, **kwargs):
        self.provider = OpenAIProvider(
            base_url=f"{AI21Model.gateway}/api/v1/openai",
            api_key=self.api_key
        )
        super().__init__(model, *args, provider=self.provider, **kwargs)

    @cached_property
    def api_key(self):
        return str(create_ai21_access_token(self.gateway))


@agent(
    name="Academic Author",
    description="Writes and reviews scholarly papers",
    version="0.1.0",
)
class AcademicAuthor(A2AServer):
    def __init__(self):
        super().__init__()
        # load CLI / env hyperparameters
        self.hparams = AgentConfig()

        # instantiate our LLM via AI21 (OpenAI-compatible)
        llm_model = AI21Model(self.hparams.model)
        self.llm = LlmAgent(
            llm_model,
            system_prompt="You are a diligent academic writer. Respond in valid markdown."
        )

        # peers read from CLI / env
        self.peer_urls: List[str] = self.hparams.peers

        self.storage = FileWorkflowStorage(str(pathlib.Path(self.hparams.storage_dir).expanduser()))

        # ─── One asyncio loop per‑process ───
        try:
            # Reuse an existing loop if the runtime has already created one
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            # Typical when running under Flask / gunicorn worker threads
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)



    # ───────────────────────────── Skills ───────────────────────────── #

    # ------------------------------------------------------------------ #
    # Run an async coroutine from sync code, no nested‑loop headaches.   #
    # If the loop is already running (rare for this server) we schedule  #
    # it thread‑safely; otherwise we run it synchronously.               #
    # ------------------------------------------------------------------ #
    def _await(self, coro):
        if self._loop.is_running():                # e.g. called from inside
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result()
        return self._loop.run_until_complete(coro)


    def run_sync(self, *args, **kwargs):
        """Override to use the LLM's run_sync method directly."""
        res = self.llm.run_sync(*args, **kwargs)
        logger.debug(f"LLM run_sync: {args} -> {res.output}")
        return res

    @skill(name="Brainstorm", description="Generate raw ideas for a topic")
    def brainstorm(self, topic: str) -> str:
        prompt = (
            f"brainstorm -> Brainstorm bullet ideas for a paper about {topic}"
        )
        return self.run_sync(prompt).output

    @skill(name="Claims", description="Produce claims & outline")
    def claims(self, ideas: str) -> str:
        prompt = (
            "claims -> Using these ideas, group into sections, and for each section list 1-2 concrete claims "
            f"that will be proven:\n\n{ideas}"
        )
        return self.run_sync(prompt).output

    @skill(name="Verify", description="Fact-check a list of scientific claims")
    def verify(self, claims: str) -> str:
        prompt = (
            "verify -> For each claim below, say `OK` if plausible with one citation, "
            "else `REVISE` and suggest a corrected claim.\n\n" + claims
        )
        return self.run_sync(prompt).output

    @skill(name="Draft", description="Write full paper from verified outline")
    def draft(self, vetted_outline: str) -> str:
        return self.run_sync(
            "draft -> Write a full academic survey paper (~2 000 words) in markdown using this outline:\n"
            + vetted_outline
        ).output

    @skill(name="Audit", description="Light peer-review of a draft")
    def audit(self, paper_md: str) -> str:
        return self.run_sync(
            "audit -> Act as a peer-reviewer: list up to five critical comments on the following draft:\n"
            + paper_md
        ).output

    @skill(name="Final", description="Compile final version of the paper")
    def final(self, paper_md: str) -> str:
        return self.run_sync(
            "final -> Compile the final version of the paper, including all peer reviews:\n"
            + paper_md
        ).output

    # ──────────────────── Task Entry & Workflow ───────────────────── #

    def handle_task(self, task):

        logger.warning(f"Handling task: {task.id} with message: {task.message}")
        # ── 2. Direct skill dispatch for internal prompts ───────────

        text = (task.message or {}).get("content", {}).get("text", "")

        if " -> " in text:
            cmd,  payload = text.split(" -> ")
            skill_map = {
                "brainstorm": self.brainstorm,
                "claims": self.claims,
                "verify": self.verify,
                "draft": self.draft,
                "audit": self.audit,
                "final": self.final,
            }
            if cmd.lower() in skill_map:  # call the matching skill
                output = skill_map[cmd.lower()](payload.strip())
                task.artifacts = [{"parts": [{"type": "text", "text": output}]}]
                task.status = TaskStatus(state=TaskState.COMPLETED)
                return task
            else:
                logger.error(f"Unknown command: {cmd}")
                task.artifacts = [{"parts": [{"type": "text", "text": f"Unknown command: {cmd}"}]}]
                task.status = TaskStatus(state=TaskState.FAILED)
                return task

        else:

            # result = asyncio.run(self._paper_flow(text))
            result = self._await(self._paper_flow(text))
            task.artifacts = [{"parts": [{"type": "text", "text": result}]}]
            task.status = TaskStatus(state=TaskState.COMPLETED)

            return task

    async def _paper_flow(self, topic: str):

        network = AgentNetwork()
        # register each peer URL
        for url in self.peer_urls:
            peer_name = f"peer-{uuid.uuid4().hex[:6]}"
            client = A2AClient(url, timeout=self.hparams.timeout)  # e.g. 120 s
            logger.info(f"Registering peer {peer_name} at {url}")
            network.add(peer_name, client)

        flow = Flow(agent_network=network, name="Paper-Authoring")

        # # 1. Brainstorm in parallel across all agents
        # brainstorm_step = flow.parallel()
        # for agent_info in network.list_agents():
        #     brainstorm_step.ask(agent_info["name"], "Brainstorm -> {topic}")
        # brainstorm_step.end_parallel()
        #
        # def merge_brainstorms(results, ctx):
        #     return "\n\n".join(results.values())

        # brainstorm_ideas = flow.execute_function(merge_brainstorms, brainstorm_step)

        # 1️⃣ parallel Brainstorm across all peers
        pb = flow.parallel()  # <‑‑ builder
        peers = network.list_agents()
        for i, agent in enumerate(peers):
            pb.ask(agent["name"], "Brainstorm -> {topic}")  # one branch
            if i < len(peers) - 1:  # not the last
                pb.branch()  # start next
        pb.end_parallel(max_concurrency=len(peers))  # close block

        # 2️⃣ merge the brainstorm outputs that come back as a dict
        def merge(results, _ctx):
            # results: {step_id: "…ideas…", …}
            return "\n\n".join(results.values())

        flow.execute_function(merge, "{latest_result}")  # latest = parallel




        # 2. Final version
        lead = random.choice(network.list_agents())["name"]

        # flow.ask(lead, "Brainstorm -> {topic}")

        final = flow.ask(lead, "Final -> {latest_result}")

        # Run the flow with the provided topic
        # logger.debug("Running the paper authoring flow...")
        result = await flow.run({"topic": topic})
        logger.info(f"Final paper draft: {result}")

        # persist the final draft & the audit bundle
        fname = f"{datetime.date.today()}_{uuid.uuid4().hex[:6]}.md"
        with open(pathlib.Path(self.storage.storage_dir) / fname, "w") as f:
            f.write(result)

        return result


if __name__ == "__main__":
    hparams = AgentConfig()
    agent = AcademicAuthor()
    run_server(agent, host="0.0.0.0", port=hparams.port, debug=hparams.debug)
