"""Error types raised by the interpreter."""


class MatlabError(Exception):
    """An error visible to user code (catchable with try/catch)."""

    def __init__(self, message: str, identifier: str = ""):
        super().__init__(message)
        self.message = message
        self.identifier = identifier
        self.stack: list = []


class BreakSignal(Exception):
    pass


class ContinueSignal(Exception):
    pass


class ReturnSignal(Exception):
    pass
