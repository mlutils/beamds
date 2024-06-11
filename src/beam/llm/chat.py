from typing import List, Dict, Any

from dataclasses import dataclass

from .conversation import Conversation
from ..utils import cached_property


@dataclass
class BeamChat:
    messages: List[Dict[str, Any]]

    @cached_property
    def conversation(self):
        return Conversation()

    def reset(self):
        self.messages = []
        self.conversation = Conversation()

    def add_message(self, message=None, images: List = None, name=None, role='user'):
        if images is None:
            content = message
        else:
            content = []
            if message is not None:
                content.append({"type": "text", "text": message})


    @property
    def conversation(self):
        return self._chat_history['chat']

    @property
    def chat_images(self):
        return self._chat_history['images']

    def reset_chat(self):
        self._chat_history = {'chat': Conversation(), 'images': []}

    @staticmethod
    def build_content(content, images=None):
        if images is None:
            return content
        content = [{"type": "text", "text": "What’s in this image?"}]
        for im in images:
            content.append({"type": "image_url", "image_url": {"url": im.url}})
        return content

    @property
    def chat_history(self):
        return [{'role': 'user' if m[0] else 'assistant', 'content': self.build_content(m[1], images=imi)}
                for m, imi in zip(self.conversation.iter_texts(), self._chat_history['images'])]

    def add_to_chat(self, text, is_user=True, images=None, name=None):

        if images is None:
            self.chat_images.append(None)
        else:
            images = [ImageContent(image=im) for im in images]
            self.chat_images.append(images)

        if is_user:
            self.conversation.add_user_input(text)
        else:
            self.conversation.append_response(text)
            self.conversation.mark_processed()

        if name is not None:
            message['name'] = name
        return self.chat_history


        # from def chat(...)
        # if system is not None:
        #     system = {'role': 'system', 'content': system}
        #     if system_name is not None:
        #         system['system_name'] = system_name
        #     messages.append(system)
        #
        # messages.extend(self.chat_history)
        #
        # # images = self.extract_images(image, images)
        #
        #
        # images = images or []
        # if image is not None:
        #     images.insert(0, image)
        #
        # message = self.add_to_chat(message, is_user=True, images=images, name=name)
        #
        # messages.append(message)

    def add_tool_message_to_chat(self, messages=None):

        if self.tools is None:
            return messages

        if messages is None:
            messages = []

        system_found = False
        for m in messages:
            if m['role'] == 'system':
                m['content'] = f"{m['content']}\n\n{self.tool_message}"
                system_found = True
                break
        if not system_found:
            messages.insert(0, {'role': 'system', 'content': self.tool_message})

        return messages