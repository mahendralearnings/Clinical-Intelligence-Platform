"""In-memory user store with four hard-coded test users (one per role).

This module is the single place where user data lives. It is intentionally
isolated from FastAPI so it can be tested or swapped without touching routes.
"""

from clinical_platform.core.security import hash_password
from clinical_platform.domain.models import Role, User

# ---------------------------------------------------------------------------
# Bootstrap: build hashed passwords once at import time so the plain-text
# strings never linger in memory after the module loads.
# ---------------------------------------------------------------------------

_USERS_RAW: list[tuple[str, str, str, Role, str]] = [
    # (id, email, full_name, role, plain_password)
    ("usr-001", "alice@clinic.dev", "Alice Researcher", Role.RESEARCHER, "researcher_pass_1!"),
    ("usr-002", "bob@clinic.dev", "Bob Doctor", Role.DOCTOR, "doctor_pass_2!"),
    (
        "usr-003",
        "carol@clinic.dev",
        "Carol Compliance",
        Role.COMPLIANCE_OFFICER,
        "compliance_pass_3!",
    ),
    (
        "usr-004",
        "dave@clinic.dev",
        "Dave Ops",
        Role.CLINICAL_OPERATIONS,
        "clinical_ops_pass_4!",
    ),
]

# email → User  (primary lookup key)
USER_STORE: dict[str, User] = {
    email: User(
        id=uid,
        email=email,
        full_name=name,
        role=role,
        hashed_password=hash_password(plain_pw),
    )
    for uid, email, name, role, plain_pw in _USERS_RAW
}


class UserRepository:
    """Read-only in-memory user repository."""

    def get_by_email(self, email: str) -> User | None:
        """Return the User with the given *email*, or None if not found."""
        return USER_STORE.get(email)

    def get_by_id(self, user_id: str) -> User | None:
        """Return the User with the given *user_id*, or None if not found."""
        return next((u for u in USER_STORE.values() if u.id == user_id), None)
