import inspect
import json
from typing import Any

from ..utils import lazy_property, jupyter_like_traceback
from dataclasses import dataclass, field
import re


@dataclass
class LLMToolProperty:
    # a class that holds an OpenAI tool parameter object

    name: str
    type: str
    description: str
    default: any = None
    required: bool = False
    enum: list = None

    def __str__(self):
        message = json.dumps(self.dict(), indent=4)
        return message

    def dict(self):
        obj = {'name': self.name,
                              'type': self.type,
                              'description': self.description,
                              'default': self.default,
                              'required': self.required,
                              'enum': self.enum}
        return obj

    @property
    def attributes(self):
        d = {'type': self.type, 'description': self.description}
        if self.default is not None:
            d['default'] = self.default
        if self.enum is not None:
            d['enum'] = self.enum
        return d


class LLMTool:
    # a class that holds an OpenAI tool object

    token_start = '[TOOL]'
    token_end = '[/TOOL]'

    def __init__(self, name=None, tool_type='function', description=None, func=None, required=None, **properties):

        self.func = func

        if name is None and func is not None:
            name = func.__name__

        if description is None and func is not None:
            description = inspect.getdoc(func)

        self.name = name or 'func'
        self.tool_type = tool_type
        self.description = description or 'See properties for more information.'

        self.properties = {}
        required = required or []
        for name, p in properties.items():
            if isinstance(p, LLMToolProperty):
                self.properties[name] = p
            else:
                r = name in required
                self.properties[name] = LLMToolProperty(name=name, required=r, **p)

        self.tool_token = f"[{self.name}]"

    @lazy_property
    def args(self):
        return [k for k, v in self.properties.items() if v.required]

    @lazy_property
    def kwargs(self):
        return [k for k, v in self.properties.items() if not v.required]

    @property
    def required(self):
        return [k for k, v in self.properties.items() if v.required]

    @lazy_property
    def tool_search_pattern(self):
        # Escape special characters in tokens
        escaped_token_start = re.escape(self.token_start)
        escaped_token_end = re.escape(self.token_end)

        # Pattern to match with or without square brackets and optional whitespace
        pattern = (rf"{escaped_token_start}\s*"
                   rf"\[?{self.name}\]?"
                   rf"\s*(.*?)\s*"
                   rf"{escaped_token_end}")

        return pattern

    def __call__(self, response):

        match = re.match(self.tool_search_pattern, response.text)
        if match:
            arguments = match.group(1)
            try:
                arguments = response.parse_text(arguments, protocol='json')
                args = arguments.get('args', [])
                kwargs = arguments.get('kwargs', {})
            except:
                return ExecutedTool(tool=self, success=False, executed=False,
                                    traceback=jupyter_like_traceback(), unparsed_arguments=arguments)
            executed = False
            success = False
            traceback = None
            res = None
            if self.func is not None:
                try:
                    res = self.func(*args, **kwargs)
                    success = True
                except:
                    traceback = jupyter_like_traceback()
                executed = True

            return ExecutedTool(tool=self, args=args, kwargs=kwargs, success=success, executed=executed,
                                traceback=traceback, response=res)

        return None

    def __str__(self):
        message = json.dumps(self.dict(), indent=4)
        return message

    def dict(self):
        return {'type': self.tool_type, self.tool_type: {'name': self.name, 'description': self.description,
                                               'parameters': {'type': 'object',
                                                              'properties': {p: v.attributes
                                                                             for p, v in self.properties.items()},
                                                              'required': self.required}}}

@dataclass
class ExecutedTool:
    tool: LLMTool
    args: tuple = None
    kwargs: dict = None
    success: bool = False
    executed: bool = False
    traceback: str = None
    response: Any = None
    unparsed_arguments: str = None
