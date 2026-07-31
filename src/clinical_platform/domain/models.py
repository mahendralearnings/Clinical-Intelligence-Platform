"""Domain models for the clinical platform."""

from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    RESEARCHER = "researcher"
    DOCTOR = "doctor"
    COMPLIANCE_OFFICER = "compliance_officer"
    CLINICAL_OPERATIONS = "clinical_operations"


class Permission(StrEnum):
    # Document / knowledge base access
    READ_DOCUMENTS = "read_documents"
    UPLOAD_DOCUMENTS = "upload_documents"

    # Clinical data
    VIEW_PATIENT_DATA = "view_patient_data"
    ANNOTATE_PATIENT_DATA = "annotate_patient_data"

    # Compliance
    VIEW_AUDIT_LOGS = "view_audit_logs"
    EXPORT_REPORTS = "export_reports"

    # Operations
    MANAGE_PIPELINES = "manage_pipelines"
    VIEW_ANALYTICS = "view_analytics"


# ---------------------------------------------------------------------------
# Role → Permission mapping (single source of truth)
# ---------------------------------------------------------------------------

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.RESEARCHER: frozenset(
        {
            Permission.READ_DOCUMENTS,
            Permission.UPLOAD_DOCUMENTS,
            Permission.VIEW_ANALYTICS,
        }
    ),
    Role.DOCTOR: frozenset(
        {
            Permission.READ_DOCUMENTS,
            Permission.VIEW_PATIENT_DATA,
            Permission.ANNOTATE_PATIENT_DATA,
        }
    ),
    Role.COMPLIANCE_OFFICER: frozenset(
        {
            Permission.READ_DOCUMENTS,
            Permission.VIEW_AUDIT_LOGS,
            Permission.EXPORT_REPORTS,
        }
    ),
    Role.CLINICAL_OPERATIONS: frozenset(
        {
            Permission.READ_DOCUMENTS,
            Permission.MANAGE_PIPELINES,
            Permission.VIEW_ANALYTICS,
            Permission.VIEW_AUDIT_LOGS,
        }
    ),
}


@dataclass(frozen=True)
class User:
    """Immutable domain user. Holds no sensitive data beyond the hashed password."""

    id: str
    email: str
    full_name: str
    role: Role
    hashed_password: str = field(repr=False)

    def has_permission(self, permission: Permission) -> bool:
        return permission in ROLE_PERMISSIONS.get(self.role, frozenset())
