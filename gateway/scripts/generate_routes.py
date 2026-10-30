"""Writes the gateway's routing file from a registry of approved models.

The served names come from the registry you point this at. They are never
computed here, because that registry is the source of truth for what's
approved to serve traffic. A row whose served name is the baked alias is
skipped, since the gateway routes that name itself. Each model's upstream is
its own Service, http://<served name>.<namespace>.svc:8000.

Expects a registry endpoint that returns a JSON list of objects shaped like
``{"served_model_name": "...", "registration_id": "..."}`` -- swap
``fetch_rows`` for whatever your own registry's API looks like; everything
else (name validation, atomic write) is reusable as-is.

    REGISTRY_URL=https://... REGISTRY_BEARER=... python scripts/generate_routes.py /etc/gateway/routes.json

The file is written atomically, so the gateway never reads a half-written table.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile

import httpx

_SERVED_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_LIST_PATH = "/models/deployable"


def build_document(rows: list[dict], *, alias: str, namespace: str) -> dict:
    """The routes document for a deployable list. Rows the gateway cannot route are skipped, not guessed at."""
    models, seen = [], set()
    for row in rows:
        name = row.get("served_model_name")
        if not isinstance(name, str) or name == alias or not _SERVED_NAME.match(name) or name in seen:
            continue
        seen.add(name)
        models.append({
            "served_model_name": name,
            "upstream": f"http://{name}.{namespace}.svc:8000",
            "registration_id": row.get("registration_id") or "",
        })
    return {"models": models}


def fetch_rows(registry_url: str, bearer: str) -> list[dict]:
    resp = httpx.get(
        f"{registry_url.rstrip('/')}{_LIST_PATH}", headers={"authorization": f"Bearer {bearer}"}, timeout=15.0,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not isinstance(rows, list):
        raise ValueError("the deployable list did not return a list")
    return rows


def write_atomic(path: str, document: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".routes-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: generate_routes.py <output-file>", file=sys.stderr)
        return 2
    registry_url = os.environ["REGISTRY_URL"]
    bearer = os.environ["REGISTRY_BEARER"]
    alias = os.environ.get("GATEWAY_SERVED_MODEL_NAME", "default")
    namespace = os.environ.get("GATEWAY_MODEL_NAMESPACE", "inference")
    document = build_document(fetch_rows(registry_url, bearer), alias=alias, namespace=namespace)
    write_atomic(argv[1], document)
    print(f"wrote {len(document['models'])} route(s) to {argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
