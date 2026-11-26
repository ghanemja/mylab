"""Error types raised by the interpreter."""


# Identifiers of the common runtime errors, by the start of the message. They apply to errors
# raised inside mylab without an identifier; error('text') from user code keeps an empty one.
_DEFAULT_IDENTIFIERS = (
    ("Too many input arguments", "MATLAB:TooManyInputs"),
    ("Not enough input arguments", "MATLAB:minrhs"),
    ("Too many output arguments", "MATLAB:TooManyOutputs"),
    ("Brace indexing is not supported", "MATLAB:cellRefFromNonCell"),
    ("Dot indexing is not supported", "MATLAB:structRefFromNonStruct"),
    ("Unable to perform assignment because dot indexing", "MATLAB:structRefFromNonStruct"),
    ("Unrecognized function or variable", "MATLAB:UndefinedFunction"),
    ("Undefined function", "MATLAB:UndefinedFunction"),
    ("Operator '", "MATLAB:UndefinedFunction"),
    ("Unrecognized field name", "MATLAB:nonExistentField"),
    ("Reference to non-existent field", "MATLAB:nonExistentField"),
    ("A null assignment can have only one non-colon index", "MATLAB:subsdeldimmismatch"),
    ("Operands to the logical AND", "MATLAB:nonLogicalConditional"),
    ("NaN's cannot be converted to logicals", "MATLAB:nologicalnan"),
    ("Conversion to ", "MATLAB:invalidConversion"),
    ("Matrix dimensions must agree", "MATLAB:dimagree"),
    ("Arrays have incompatible sizes", "MATLAB:sizeDimensionsMustMatch"),
    ("Incorrect dimensions for matrix multiplication", "MATLAB:innerdim"),
    ("Incorrect dimensions for raising a matrix to a power", "MATLAB:square"),
    ("Dimensions of arrays being concatenated are not consistent", "MATLAB:catenate:dimensionMismatch"),
    ("Index exceeds the number of array elements", "MATLAB:badsubscript"),
    ("Index in position", "MATLAB:badsubscript"),
    ("Array indices must be positive integers or logical values", "MATLAB:badsubscript"),
    ("The logical indices contain a true value outside of the array bounds", "MATLAB:badsubscript"),
    ("Number of elements must not change", "MATLAB:getReshapeDims:notSameNumel"),
    ("Maximum recursion limit", "MATLAB:recursionLimit"),
    ("Output argument \"", "MATLAB:unassignedOutputs"),
    ("The specified key is not present in this container", "MATLAB:Containers:Map:NoKey"),
    ("Attempt to execute SCRIPT", "MATLAB:scriptNotAFunction"),
    ("Integers can only be combined with integers of the same class", "MATLAB:mixedClasses"),
    ("SWITCH expression must be a scalar or a character vector", "MATLAB:badSwitchExpression"),
    ("Assertion failed", "MATLAB:assertion:failed"),
    ("Non-scalar in Uniform output", "MATLAB:cellfun:NotAScalarOutput"),
    ("A field named '", "MATLAB:rmfield:InvalidFieldname"),
    ("Parse error", "MATLAB:m_parse_error"),
)


def default_identifier(message: str) -> str:
    """The MATLAB identifier that goes with a message mylab raises itself ('' if none is known)."""
    for start, ident in _DEFAULT_IDENTIFIERS:
        if message.startswith(start):
            return ident
    return ""


class MatlabError(Exception):
    """An error visible to user code (catchable with try/catch).

    ``identifier`` None means "not given": the identifier MATLAB uses for that message is filled
    in when one is known. Pass "" for an error that must not have one (``error('text')``).
    """

    def __init__(self, message: str, identifier: str | None = None):
        super().__init__(message)
        self.message = message
        self.identifier = default_identifier(message) if identifier is None else identifier
        self.stack: list = []


class BreakSignal(Exception):
    pass


class ContinueSignal(Exception):
    pass


class ReturnSignal(Exception):
    pass
