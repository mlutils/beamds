from typing import List
from beam import BeamParam, BeamConfig


class AgentConfig(BeamConfig):
    parameters = [
        BeamParam('port', int, 5000, 'Port to run the A2A server on'),
        BeamParam('model', str, 'claude-3-7-sonnet', 'Model ID to use for the agent'),
        BeamParam('peers', List[str], [], 'List of peer agent URLs to connect to'),
        BeamParam("debug", bool, False, "Flask debug / auto-reload"),
        BeamParam("storage_dir", str, "./storage", "Where to persist artefacts"),
        BeamParam("timeout", int, 120, "HTTP timeout (s) for peer calls"),
        BeamParam("topic", str, "What is the meaning of life?", "Initial query for the agent"),
    ]
