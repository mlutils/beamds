from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from ai21_models_gateway.utils.ai21 import create_ai21_access_token
from beam.utils import cached_property
from beam import parallel, task


class AI21Model(OpenAIModel):
    """
    An AI21 agent that uses the AI21 Models Gateway to access OpenAI-compatible models.
    """

    gateway = "https://tools.algo-agents.ai21.com/models-gateway"

    def __init__(self, model, *args, **kwargs):

        self.provider = OpenAIProvider(
            base_url=f"{AI21Model.gateway}/api/v1/openai",  # must include the /v1 root
            api_key=self.api_key  # placeholder is fine if your server skips auth
        )

        super().__init__(model, *args,  provider=self.provider, **kwargs)


    @cached_property
    def api_key(self):
        """
        Create an API key for the AI21 Models Gateway.
        """
        return str(create_ai21_access_token(self.gateway))


llm = AI21Model("claude-3-7-sonnet")

agent = Agent(llm, instructions='Be fun!')
app = agent.to_a2a()


# if __name__ == "__main__":
#
#     llm = AI21Model("claude-3-7-sonnet")
#
#     agent = Agent(llm, instructions='Be fun!')
#     app = agent.to_a2a()

    # agent = Agent(llm, system_prompt="Be concise, reply with one word.")
    #
    # prompts = [
    #     'Where does "hello world" come from?',
    #     'Why is the sky blue?',
    #     'Summarise TCP',
    #     "What's Schrödinger’s cat thought-experiment?",
    # ]
    #
    # res = parallel([task(agent.run)(p) for p in prompts], n_workers=5)
    #
    # print(res)




# # 2. Claims & outline (choose a random lead)
        # lead = random.choice(network.list_agents())["name"]
        # outline = flow.ask(lead, "Claims -> {latest_result}")
        #
        # # 3. Verification by all non-leads
        # for agent_info in network.list_agents():
        #     if agent_info["name"] != lead:
        #         flow.ask(agent_info["name"], "Verify -> {latest_result}")
        #
        # # 4. Merge verifications
        # def merge_results(results, ctx):
        #     return "\n\n".join(results.values())
        # vetted_outline = flow.execute_function(merge_results)
        #
        # # 5. Draft by lead
        # draft_md = flow.ask(lead, "Draft -> {latest_result}")
        #
        # # 6. Audit in parallel
        # audit_step = flow.parallel()
        # for agent_info in network.list_agents():
        #     audit_step.ask(agent_info["name"], "Audit -> {latest_result}")
        # audit_step.end_parallel()
        #
        # # 7. Compile final
        # def compile_final(audit_results, ctx):
        #     reviews = "\n\n".join(audit_results.values())
        #     return f"# Final Draft\n\n{draft_md.value}\n\n---\n\n## Peer Reviews\n{reviews}"
        # final = flow.execute_function(compile_final, audit_step)

        # logger.debug(f"Lead {lead} claims outline: {outline}")
        # logger.debug(f"Vetted outline: {vetted_outline}")
        # logger.debug(f"Final paper draft: {final}")


# def handle_task(self, task):
#     """Entry point for every inbound A2A message.
#
#     Internal A2A prompts like “Brainstorm …”, “Claims …”, “Verify …”,
#     “Draft …”, and “Audit …” are handled **locally** instead of spawning
#     a brand-new paper-authoring flow, which prevents infinite recursion.
#     """
#     text = (task.message or {}).get("content", {}).get("text", "")
#     if not text:
#         task.status = TaskStatus(state=TaskState.INPUT_REQUIRED)
#         return task
#
#     # ── 1. Debug passthrough ────────────────────────────────────
#     if text.lower().startswith("debug:"):
#         text = text[len("debug:"):].strip()
#         output = self.run_sync(text).output
#         task.artifacts = [{"parts": [{"type": "text", "text": output}]}]
#         task.status = TaskStatus(state=TaskState.COMPLETED)
#         return task
#
#     # ── 2. Direct skill dispatch for internal prompts ───────────
#     if " -> " in text:
#         return  super().handle_task(task)  # call the base method to handle the task
#
#         # cmd,  payload = text.split(" -> ")
#         # skill_map = {
#         #     "brainstorm": self.brainstorm,
#         #     "claims": self.claims,
#         #     "verify": self.verify,
#         #     "draft": self.draft,
#         #     "audit": self.audit,
#         # }
#         # if cmd.lower() in skill_map:  # call the matching skill
#         #     output = skill_map[cmd.lower()](payload.strip())
#         #     task.artifacts = [{"parts": [{"type": "text", "text": output}]}]
#         #     task.status = TaskStatus(state=TaskState.COMPLETED)
#         #     return task
#         # else:
#         #     logger.error(f"Unknown command: {cmd}")
#         #     task.status = TaskStatus(state=TaskState.FAILED, error=f"Unknown command: {cmd}")
#         #     return task
#
#     # ── 3. Top-level user request → full paper workflow ─────────
#     asyncio.run(self._paper_flow(task, text))
#     return task