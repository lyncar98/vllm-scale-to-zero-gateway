"""A second trusted caller presents its own key. Unset, it is not accepted at all."""

import pytest
from fastapi import HTTPException

from app.config import Settings
from app.main import require_caller


def _check(settings, authorization):
    require_caller(settings=settings, authorization=authorization)


def _settings(**over):
    base = {"gateway_api_key": "primary-key", "require_auth": True}
    base.update(over)
    return Settings(**base)


def test_the_primary_key_still_works_with_a_secondary_key_configured():
    _check(_settings(gateway_secondary_api_key="secondary-key"), "Bearer primary-key")


def test_the_secondary_key_is_accepted_when_configured():
    _check(_settings(gateway_secondary_api_key="secondary-key"), "Bearer secondary-key")


@pytest.mark.parametrize("header", [None, "", "Bearer ", "Bearer wrong", "primary-key", "Bearer primary-key-x"])
def test_any_other_credential_is_refused(header):
    with pytest.raises(HTTPException) as exc:
        _check(_settings(gateway_secondary_api_key="secondary-key"), header)
    assert exc.value.status_code == 401


def test_with_no_secondary_key_configured_a_blank_bearer_is_still_refused():
    with pytest.raises(HTTPException):
        _check(_settings(), "Bearer ")


def test_with_no_secondary_key_configured_the_secondary_caller_is_not_accepted():
    with pytest.raises(HTTPException):
        _check(_settings(), "Bearer secondary-key")


def test_auth_off_for_local_runs_accepts_anything():
    _check(_settings(require_auth=False), None)
