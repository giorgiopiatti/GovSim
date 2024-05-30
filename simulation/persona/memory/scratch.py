import json
import os

from ..common import ChatObservation


class Scratch:
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

        if os.path.exists(f"{base_path}/scratch.json"):
            saved_info = json.load(open(f"{base_path}/scratch.json", "r"))
            for key, value in saved_info.items():
                setattr(self, key, value)
