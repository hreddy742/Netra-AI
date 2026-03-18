"""Tests for RBAC roles."""
import pytest
from fastapi import HTTPException
from backend.services.auth.roles import check_role, has_permission, role_index, ROLE_HIERARCHY


def test_admin_passes_any_role():
    user = {"sub": "u1", "role": "admin"}
    check_role(user, "admin")
    check_role(user, "manager")
    check_role(user, "operator")


def test_operator_blocked_from_manager():
    user = {"sub": "u1", "role": "operator"}
    with pytest.raises(HTTPException) as exc:
        check_role(user, "manager")
    assert exc.value.status_code == 403


def test_manager_blocked_from_admin():
    user = {"sub": "u1", "role": "manager"}
    with pytest.raises(HTTPException):
        check_role(user, "admin")


def test_manager_passes_manager_and_operator():
    user = {"sub": "u1", "role": "manager"}
    check_role(user, "manager")
    check_role(user, "operator")


def test_unknown_role_raises():
    user = {"sub": "u1", "role": "superadmin"}
    with pytest.raises(HTTPException) as exc:
        check_role(user, "admin")
    assert exc.value.status_code == 403


def test_has_permission_admin():
    assert has_permission("admin", "stores:write")
    assert has_permission("admin", "models:deploy")


def test_has_permission_operator_limited():
    assert has_permission("operator", "incidents:read")
    assert not has_permission("operator", "stores:write")
    assert not has_permission("operator", "users:write")


def test_role_hierarchy_order():
    assert role_index("operator") < role_index("manager") < role_index("admin")


def test_role_hierarchy_completeness():
    assert "operator" in ROLE_HIERARCHY
    assert "manager" in ROLE_HIERARCHY
    assert "admin" in ROLE_HIERARCHY
