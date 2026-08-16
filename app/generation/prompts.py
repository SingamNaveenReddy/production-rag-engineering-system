from __future__ import annotations

from pathlib import Path

import yaml


class PromptTemplate:
    def __init__(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.version = str(data["version"])
        self.system = str(data["system"])
        self.template = str(data["template"])

    def render(self, question: str, context: str) -> str:
        return self.template.format(question=question, context=context)

