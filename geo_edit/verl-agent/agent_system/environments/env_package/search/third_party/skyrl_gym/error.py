"""Set of Error classes for skyrl_gym."""


class Error(Exception):
    """Error superclass."""


class RegistrationError(Error):
    """Raised when the user attempts to register an invalid env."""
