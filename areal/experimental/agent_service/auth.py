# SPDX-License-Identifier: Apache-2.0

"""Shared authentication utilities for the Agent Service."""

from __future__ import annotations

from fastapi import Header, HTTPException

DEFAULT_ADMIN_KEY = "areal-agent-admin"


async def verify_admin_key(
    authorization: str = Header(alias="Authorization"),
    *,
    expected_key: str,
) -> None:
    expected = f"Bearer {expected_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid admin key")


def make_admin_dependency(admin_key: str):
    async def _dep(authorization: str = Header(alias="Authorization")) -> None:
        await verify_admin_key(authorization, expected_key=admin_key)

    return _dep


def admin_headers(admin_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_key}"}
