"""The gateway's routing table: served model name to upstream model server.

The table is generated from whatever registry of approved models you run (see
scripts/generate_routes.py for the shape it expects). The gateway never
computes served names itself, because that registry is the source of truth.
The baked model stays routable under ``served_model_name`` as an alias until
every model has moved to its own deployment.

The file is re-read when its modification time changes, so a regenerated table
takes effect without a restart. A file that is set but cannot be read or
parsed is a hard error: every request to a non-alias model then fails with
503, rather than routing to the wrong model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


class RouteFileError(RuntimeError):
    pass


@dataclass
class Route:
    served_model_name: str
    upstream: str
    registration_id: str = ""


@dataclass
class RouteStore:
    alias: str
    alias_upstream: str
    path: str = ""
    _mtime: float = -1.0
    _routes: dict[str, Route] = field(default_factory=dict)
    _error: str = ""

    def lookup(self, served_model_name: str) -> Route | None:
        """The route for a served name, or None. Raises RouteFileError when the table cannot be trusted."""
        if served_model_name == self.alias:
            return Route(self.alias, self.alias_upstream)
        self._refresh()
        if self._error:
            raise RouteFileError(self._error)
        return self._routes.get(served_model_name)

    def all(self) -> list[Route]:
        self._refresh()
        if self._error:
            raise RouteFileError(self._error)
        return [Route(self.alias, self.alias_upstream), *self._routes.values()]

    def _refresh(self) -> None:
        if not self.path:
            return
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError as exc:
            self._routes, self._error, self._mtime = {}, f"routes file {self.path} is unreadable: {exc}", -1.0
            return
        if mtime == self._mtime and self._error == "":
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                self._routes = parse_routes(json.load(fh), alias=self.alias)
            self._error = ""
        except (OSError, ValueError, RouteFileError) as exc:
            self._routes, self._error = {}, f"routes file {self.path} is invalid: {exc}"
        self._mtime = mtime


def parse_routes(document: object, *, alias: str = "") -> dict[str, Route]:
    """Validates a routes document. Duplicate names and names equal to the alias are refused."""
    if not isinstance(document, dict) or not isinstance(document.get("models"), list):
        raise RouteFileError("expected an object with a 'models' list")
    routes: dict[str, Route] = {}
    for entry in document["models"]:
        if not isinstance(entry, dict):
            raise RouteFileError("every model entry must be an object")
        name, upstream = entry.get("served_model_name"), entry.get("upstream")
        if not isinstance(name, str) or not name or not isinstance(upstream, str) or not upstream.startswith("http"):
            raise RouteFileError(f"model entry needs a served_model_name and an http upstream: {entry!r}")
        if name == alias:
            raise RouteFileError(f"served_model_name {name!r} is the baked alias and cannot be routed")
        if name in routes:
            raise RouteFileError(f"served_model_name {name!r} appears twice")
        routes[name] = Route(name, upstream.rstrip("/"), str(entry.get("registration_id", "")))
    return routes
