"""The routing file is built only from the registry's served names, and is written atomically."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import generate_routes  # noqa: E402

from app.routes import RouteStore  # noqa: E402


def test_each_row_routes_to_its_own_service_under_the_namespace():
    doc = generate_routes.build_document(
        [{"served_model_name": "model-x-abcdef12", "registration_id": "reg-1"}],
        alias="default", namespace="inference",
    )
    assert doc == {"models": [{
        "served_model_name": "model-x-abcdef12",
        "upstream": "http://model-x-abcdef12.inference.svc:8000",
        "registration_id": "reg-1",
    }]}


def test_the_baked_alias_is_skipped_because_the_gateway_routes_it_already():
    doc = generate_routes.build_document([{"served_model_name": "default"}], alias="default", namespace="inference")
    assert doc["models"] == []


@pytest.mark.parametrize("name", ["", "UPPER", "a/b", "-lead", "x" * 41, None])
def test_names_that_are_not_dns_labels_are_skipped(name):
    doc = generate_routes.build_document([{"served_model_name": name}], alias="default", namespace="inference")
    assert doc["models"] == []


def test_a_repeated_name_is_routed_once():
    rows = [{"served_model_name": "model-a-00000000"}] * 2
    assert len(generate_routes.build_document(rows, alias="default", namespace="inference")["models"]) == 1


def test_the_written_file_is_read_back_by_the_gateway(tmp_path):
    out = tmp_path / "routes.json"
    doc = generate_routes.build_document(
        [{"served_model_name": "model-a-00000000"}], alias="default", namespace="inference",
    )
    generate_routes.write_atomic(str(out), doc)
    store = RouteStore(alias="default", alias_upstream="http://baked", path=str(out))
    route = store.lookup("model-a-00000000")
    assert route is not None and route.upstream == "http://model-a-00000000.inference.svc:8000"
    assert json.loads(out.read_text(encoding="utf-8")) == doc
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".routes-")]
