# Explicit imports for IDE
if len([]):

    from .resource import beam_llm
    from .utils import text_splitter, split_to_tokens, default_tokenizer, estimate_tokens, get_conversation_template

    from .tools import LLMTool, LLMToolProperty
    from .task import LLMTask

    from .simulators import openai as openai_simulator
    from .simulators import text_generation as tgi_simulator
    from .simulators import openai_legacy as openai_legacy_simulator
    from .tools import LLMGuidance


__all__ = ['beam_llm', 'text_splitter', 'split_to_tokens', 'default_tokenizer', 'estimate_tokens',
           'get_conversation_template', 'LLMTool', 'LLMToolProperty', 'LLMTask', 'openai_simulator',
           'tgi_simulator', 'openai_legacy_simulator', 'LLMGuidance']


def __getattr__(name):
    if name == 'beam_llm':
        from .resource import beam_llm
        return beam_llm
    elif name == 'text_splitter':
        from .utils import text_splitter
        return text_splitter
    elif name == 'split_to_tokens':
        from .utils import split_to_tokens
        return split_to_tokens
    elif name == 'default_tokenizer':
        from .utils import default_tokenizer
        return default_tokenizer
    elif name == 'estimate_tokens':
        from .utils import estimate_tokens
        return estimate_tokens
    elif name == 'get_conversation_template':
        from .utils import get_conversation_template
        return get_conversation_template
    elif name == 'LLMTool':
        from .tools import LLMTool
        return LLMTool
    elif name == 'LLMToolProperty':
        from .tools import LLMToolProperty
        return LLMToolProperty
    elif name == 'LLMTask':
        from .task import LLMTask
        return LLMTask
    elif name == 'openai_simulator':
        from .simulators import openai as openai_simulator
        return openai_simulator
    elif name == 'tgi_simulator':
        from .simulators import text_generation as tgi_simulator
        return tgi_simulator
    elif name == 'openai_legacy_simulator':
        from .simulators import openai_legacy as openai_legacy_simulator
        return openai_legacy_simulator
    elif name == 'LLMGuidance':
        from .tools import LLMGuidance
        return LLMGuidance
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
