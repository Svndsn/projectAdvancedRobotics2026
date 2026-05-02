"""Placeholder real gripper interface for pipeline integration tests.

This module intentionally avoids any socket or hardware access for now.
"""

from __future__ import annotations


class Gripper:
    def __init__(self, confirm_actions: bool = False) -> None:
        self.confirm_actions = bool(confirm_actions)
        self._holding = False
        self._pregripped = False

    def _maybe_confirm(self, action: str, confirm: bool | None) -> None:
        should_confirm = self.confirm_actions if confirm is None else bool(confirm)
        print(f"[Gripper] {action}")
        if should_confirm:
            input("Press Enter to continue...")

    def open(self, confirm: bool | None = None) -> None:
        self._maybe_confirm("open()", confirm)
        self._holding = False
        self._pregripped = False

    def pregrip(self, confirm: bool | None = None) -> None:
        self._maybe_confirm("pregrip()", confirm)
        self._pregripped = True

    def grip(self, confirm: bool | None = None) -> None:
        self._maybe_confirm("grip()", confirm)
        self._pregripped = False
        self._holding = True

    def release(self, confirm: bool | None = None) -> None:
        self._maybe_confirm("release()", confirm)
        self._holding = False
        self._pregripped = False

    def is_holding(self) -> bool:
        print(f"[Gripper] is_holding() -> {self._holding}")
        return self._holding
