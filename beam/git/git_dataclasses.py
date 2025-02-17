from dataclasses import dataclass, field, asdict
from typing import List, Union, Optional
from typing import Dict, Any


@dataclass
class GitFilesConfig:
    src_dir: str
    dest_dir: str
    file_path: List[str]
    file_names: List[str]