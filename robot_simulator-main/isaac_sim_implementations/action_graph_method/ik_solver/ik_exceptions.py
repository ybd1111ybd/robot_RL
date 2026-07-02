"""
IK Solver Exception Classes

Defines custom exceptions for better error handling and debugging.
"""


class IKException(Exception):
    """Base exception for IK solver errors."""
    pass


class IKConfigurationError(IKException):
    """Raised when IK configuration is invalid."""
    pass


class IKConvergenceError(IKException):
    """Raised when IK fails to converge within tolerance."""
    pass


class IKSingularityError(IKException):
    """Raised when Jacobian is singular or near-singular."""
    pass


class IKJointLimitError(IKException):
    """Raised when joint limits are violated or unreachable."""
    pass


class IKInputValidationError(IKException):
    """Raised when input data is invalid (NaN, inf, wrong shape, etc.)."""
    pass


class IKArticulationError(IKException):
    """Raised when articulation view access fails."""
    pass


class IKTimeoutError(IKException):
    """Raised when IK computation exceeds timeout."""
    pass
