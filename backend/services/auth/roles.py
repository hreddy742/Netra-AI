"""
Netra AI — RBAC roles and permission enforcement.

Roles (hierarchy, lowest → highest):
  operator  — incident review, camera monitoring only
  manager   — store-level analytics, alert config, camera management
  admin     — full system access (users, stores, model deployment, system config)

Usage in gateway:
    from backend.services.auth.roles import check_role

    def require_role(minimum_role: str):
        async def _dep(user: dict = Depends(get_current_user)) -> dict:
            check_role(user, minimum_role)
            return user
        return _dep
"""
from __future__ import annotations
from fastapi import HTTPException, status

# Hierarchical order — index = privilege level
ROLE_HIERARCHY: list[str] = ["operator", "manager", "admin"]

# Permission sets per role
_PERMISSIONS: dict[str, set[str]] = {
    "operator": {
        "incidents:read",
        "incidents:review",
        "cameras:read",
        "stream:read",
    },
    "manager": {
        "incidents:read",
        "incidents:review",
        "cameras:read",
        "cameras:write",
        "stream:read",
        "analytics:read",
        "analytics:export",
        "alerts:read",
        "alerts:write",
        "models:read",
    },
    "admin": {
        "incidents:read",
        "incidents:review",
        "cameras:read",
        "cameras:write",
        "stream:read",
        "analytics:read",
        "analytics:export",
        "alerts:read",
        "alerts:write",
        "users:read",
        "users:write",
        "stores:read",
        "stores:write",
        "system:config",
        "models:read",
        "models:deploy",
        "demo:control",
    },
}


def check_role(user: dict, minimum_role: str) -> None:
    """
    Raise HTTPException(403) if the user's role is below minimum_role.
    Called inline by require_role() dependency factory in the gateway.
    """
    role = user.get("role", "operator")
    try:
        user_level = ROLE_HIERARCHY.index(role)
        min_level  = ROLE_HIERARCHY.index(minimum_role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unknown role '{role}'",
        )
    if user_level < min_level:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{role}' cannot access this resource (requires '{minimum_role}')",
        )


def has_permission(role: str, permission: str) -> bool:
    """Check if role has a specific permission (non-raising)."""
    return permission in _PERMISSIONS.get(role, set())


def role_index(role: str) -> int:
    """Returns numeric privilege level for the role (-1 if unknown)."""
    try:
        return ROLE_HIERARCHY.index(role)
    except ValueError:
        return -1
