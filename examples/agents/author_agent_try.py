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
from beam import logger

from examples.agents.config import AgentConfig


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

    # ───────────────────────────── Skills ───────────────────────────── #

    def run_sync(self, *args, **kwargs):
        """Override to use the LLM's run_sync method directly."""
        res = self.llm.run_sync(*args, **kwargs)
        logger.debug(f"LLM run_sync: {args} -> {res.output}")
        return res

    @skill(name="Brainstorm", description="Generate raw ideas for a topic")
    def brainstorm(self, topic: str) -> str:
        return self.run_sync(f"Brainstorm bullet ideas for a paper about {topic}").output

    @skill(name="Claims", description="Produce claims & outline")
    def claims(self, ideas: str) -> str:
        prompt = (
            "Using these ideas, group into sections, and for each section list 1-2 concrete claims "
            f"that will be proven:\n\n{ideas}"
        )
        return self.run_sync(prompt).output

    @skill(name="Verify", description="Fact-check a list of scientific claims")
    def verify(self, claims: str) -> str:
        prompt = (
            "For each claim below, say `OK` if plausible with one citation, "
            "else `REVISE` and suggest a corrected claim.\n\n" + claims
        )
        return self.run_sync(prompt).output

    @skill(name="Draft", description="Write full paper from verified outline")
    def draft(self, vetted_outline: str) -> str:
        return self.run_sync(
            "Write a full academic survey paper (~2 000 words) in markdown using this outline:\n"
            + vetted_outline
        ).output

    @skill(name="Audit", description="Light peer-review of a draft")
    def audit(self, paper_md: str) -> str:
        return self.run_sync(
            "Act as a peer-reviewer: list up to five critical comments on the following draft:\n"
            + paper_md
        ).output


if __name__ == "__main__":
    hparams = AgentConfig()
    agent = AcademicAuthor()
    run_server(agent, host="0.0.0.0", port=hparams.port, debug=hparams.debug)
