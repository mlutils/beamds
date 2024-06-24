from argparse import Namespace
from ..openai import simulate_openai_chat, simulate_openai_completion


openai_legacy_simulator = Namespace(ChatCompletion=Namespace(create=simulate_openai_chat),
                             Completion=Namespace(create=simulate_openai_completion))

