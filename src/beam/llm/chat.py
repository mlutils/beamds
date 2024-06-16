from typing import List, Dict, Any, Union
from dataclasses import dataclass, field
from .hf_conversation import Conversation
from .utils import get_conversation_template
from .tools import image_content, build_content


@dataclass
class BeamChat:
    scheme: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    adapter: str = 'unknown'
    system_message: Union[str, None] = None
    tool_message: Union[str, None] = None

    def reset(self):
        self.messages = []
        self.system_message = None
        self.tool_message = None

    def remove_tool_message(self):
        self.tool_message = None

    def remove_system_message(self):
        self.system_message = None

    def add_system_message(self, message, overwrite=False):

        if overwrite or self.system_message is None:
            self.system_message = message
        else:
            self.messages.append({'role': 'system', 'content': message})

    def add_tool_message(self, message, overwrite=False):
        if overwrite or self.tool_message is None:
            self.tool_message = message
        else:
            self.messages.append({'role': 'system', 'content': message})

    def add_user_message(self, message=None, images: List = None):
        if not images:
            content = message
        else:
            content = []
            if message is not None:
                content.append({"type": "text", "text": message})
            for im in images:
                im = image_content(im)
                content.append({"type": "image_url", "image_url": im})
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, message=None):
        self.messages.append({"role": "assistant", "content": message})

    @property
    def openai_format(self):
        messages = []
        if self.system_message is not None:
            messages.append({'role': 'system', 'content': self.system_message})
        messages.extend(self.messages)
        return build_content(messages)

    @property
    def fastchat_format(self):

        conversation = get_conversation_template(self.adapter)

        system_message = conversation.system_message
        if self.system_message is not None:
            system_message = self.system_message

        if self.tool_message is not None:
            system_message = f"{system_message}\n{self.tool_message}\n"

        conversation.set_system_message(system_message)

        for m in self.messages:
            conversation.append_message(m['role'], m['content'])

        return conversation.get_prompt()

    @property
    def hf_format(self):
        conversation = Conversation()
        if self.system_message is not None:
            conversation.add_message({"role": "system", "content": self.system_message})
        if self.tool_message is not None:
            conversation.add_message({"role": "system", "content": self.tool_message})
        for m in self.messages:
            conversation.add_message(m)
        return conversation
