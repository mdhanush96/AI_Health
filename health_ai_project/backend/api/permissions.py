"""
MedAI – Custom Permissions
Extensible permission classes for future authentication integration.
"""

from rest_framework.permissions import BasePermission


class IsServiceHealthCheck(BasePermission):
    """Allow unauthenticated access to health-check endpoints only."""

    def has_permission(self, request, view):
        return True