"""Help text for the unit-testing framework."""

_C = "matlab.unittest.constraints."
_S = "matlab.unittest.selectors."
_F = "matlab.unittest.fixtures."
_P = "matlab.unittest.plugins."
_USE = (" Use it with verifyThat, assertThat, assumeThat or fatalAssertThat: "
        "testCase.verifyThat(actual, {call}). Constraints combine with &, | and ~.")
_SEL = (" Use it with selectIf: suite = selectIf(suite, {call}). Text may contain * and ? as wildcards, or "
        "be a text constraint such as ContainsSubstring('abc'). Selectors combine with &, | and ~.")
_FIX = (" Apply it with f = testCase.applyFixture(...); it is undone automatically when the test (or, in a "
        "TestClassSetup method, the test class) finishes.")
_PLUG = " Add it to a runner with addPlugin(runner, plugin) before calling run(runner, suite)."
_LEVELS = ("Verbosity levels for test output and logged messages, from least to most detail: None (0), "
           "Terse (1), Concise (2), Detailed (3), Verbose (4). Used with log(testCase, level, message), "
           "runtests(..., 'OutputDetail', level) and runtests(..., 'LoggingLevel', level).")


def _con(name: str, call: str, what: str) -> tuple:
    return (f"c = {_C}{call}", what + _USE.format(call=call))


def _sel(name: str, call: str, what: str) -> tuple:
    return (f"s = {_S}{call}", what + _SEL.format(call=call))


_KINDS = {
    "verify": "A failure is recorded and the test goes on (the test ends as Failed).",
    "assert": "A failure stops the current test at once (Failed and Incomplete); later tests still run.",
    "assume": "A failure stops the current test and marks it Incomplete (filtered) without failing it: "
              "use it for conditions the test needs but does not check.",
    "fatalAssert": "A failure stops the whole test run: use it when later tests cannot be trusted.",
}
_QUALS = {
    "Equal": ("(testCase, actual, expected)",
              "Checks that actual equals expected in class, size and value (NaN equals NaN; cell arrays and "
              "structs are compared element by element). 'AbsTol', t and 'RelTol', t allow numeric "
              "differences up to t, or t times the expected value."),
    "NotEqual": ("(testCase, actual, notExpected)", "Checks that the two values differ."),
    "True": ("(testCase, actual)", "Checks that actual is the logical scalar true."),
    "False": ("(testCase, actual)", "Checks that actual is the logical scalar false."),
    "Empty": ("(testCase, actual)", "Checks that actual has no elements."),
    "NotEmpty": ("(testCase, actual)", "Checks that actual has at least one element."),
    "Size": ("(testCase, actual, expectedSize)", "Checks that size(actual) equals expectedSize."),
    "Length": ("(testCase, actual, expectedLength)", "Checks that length(actual) equals expectedLength."),
    "NumElements": ("(testCase, actual, expectedCount)", "Checks that numel(actual) equals expectedCount."),
    "Class": ("(testCase, actual, className)", "Checks that the class of actual is exactly className."),
    "InstanceOf": ("(testCase, actual, className)",
                   "Checks that isa(actual, className) is true (subclasses count)."),
    "SameHandle": ("(testCase, actual, expectedHandle)", "Checks that both are the same handle object."),
    "NotSameHandle": ("(testCase, actual, otherHandle)", "Checks that the two are different handle objects."),
    "Substring": ("(testCase, actual, substring)", "Checks that the text actual contains substring."),
    "Matches": ("(testCase, actual, expression)", "Checks that the text actual matches the regular expression."),
    "GreaterThan": ("(testCase, actual, bound)", "Checks that every element of actual is greater than bound."),
    "GreaterThanOrEqual": ("(testCase, actual, bound)", "Checks that every element of actual is at least bound."),
    "LessThan": ("(testCase, actual, bound)", "Checks that every element of actual is less than bound."),
    "LessThanOrEqual": ("(testCase, actual, bound)", "Checks that every element of actual is at most bound."),
    "ReturnsTrue": ("(testCase, functionHandle)", "Checks that the function returns the logical scalar true."),
    "Error": ("(testCase, functionHandle, identifier)",
              "Checks that calling the function raises an error with that identifier, for example "
              "testCase.verifyError(@() myFun(-1), 'myFun:negative'). ?MException accepts any error."),
    "Warning": ("(testCase, functionHandle, identifier)",
                "Checks that calling the function issues a warning with that identifier. The function's "
                "outputs are returned."),
    "WarningFree": ("(testCase, functionHandle)",
                    "Checks that calling the function issues no warning. The function's outputs are returned."),
    "Fail": ("(testCase, diagnostic)", "Fails unconditionally, for code paths that must not be reached."),
    "That": ("(testCase, actual, constraint)",
             "Checks actual against a constraint object from matlab.unittest.constraints, for example "
             "IsEqualTo(5, 'Within', AbsoluteTolerance(0.1)) or IsGreaterThan(3) & IsLessThan(9)."),
}
_QUAL_HELP = {}
for _prefix, _note in _KINDS.items():
    for _suffix, (_sig, _what) in _QUALS.items():
        _usage = [f"{_prefix}{_suffix}{_sig}"]
        if _suffix not in ("Fail",):
            _usage.append(f"{_prefix}{_suffix}{_sig[:-1]}, diagnostic)")
        _QUAL_HELP[_prefix + _suffix] = (
            _usage, f"{_what} {_note} A method of the test case, also written testCase.{_prefix}{_suffix}(...). "
            "The optional diagnostic (text, or a function handle to run) is shown on failure together with "
            "the actual and expected values.")

HELP = {
    **_QUAL_HELP,
    "addTeardown": ("addTeardown(testCase, @fcn, arg1, ...)",
                    "Registers a function to run when the test finishes, whether it passed or not. Functions "
                    "run in the reverse order of registration. In a setupOnce function or TestClassSetup "
                    "method the function runs after the last test of the file or class."),
    "applyFixture": ("f = applyFixture(testCase, fixture)",
                     "Sets up a fixture such as matlab.unittest.fixtures.TemporaryFolderFixture and undoes "
                     "it when the test finishes. Returns the fixture, whose properties (for example Folder) "
                     "can then be read."),
    # -- running tests ---------------------------------------------------------------------------------
    "runtests": (["results = runtests", "results = runtests(tests)", "results = runtests(tests, Name, Value)"],
                 "Runs tests and returns an array of TestResult objects (Name, Passed, Failed, Incomplete, "
                 "Duration, Details; table(results) gives a table). tests is the name of a test file, test "
                 "class, folder or package, 'File/testName' for one test, or a cell array of these; with "
                 "no input every test file in the current folder runs. A test file is function-based "
                 "(its main function returns functiontests(localfunctions)), class-based (a classdef "
                 "derived from matlab.unittest.TestCase) or script-based (sections started by %%). "
                 "Selection options: 'Name', 'ProcedureName', 'Tag', 'ParameterProperty', 'ParameterName', "
                 "'BaseFolder', 'Superclass' (wildcards * and ? allowed) and 'IncludeSubfolders'. "
                 "'OutputDetail' (0 to 4 or None/Terse/Concise/Detailed/Verbose) sets how much is "
                 "printed and 'LoggingLevel' which log(testCase, ...) messages are shown. Progress is "
                 "printed as 'Running Name', one dot per test and 'Done Name'; a failing test prints its "
                 "diagnostics with the actual and expected values."),
    "testsuite": (["suite = testsuite", "suite = testsuite(tests)", "suite = testsuite(tests, Name, Value)"],
                  "Collects tests into a suite (an array of matlab.unittest.Test objects) without running "
                  "them. Takes the same inputs and selection options as runtests. Run the suite with "
                  "results = run(suite), narrow it with selectIf(suite, ...), and join suites with "
                  "[suite1, suite2]."),
    "functiontests": ("tests = functiontests(localfunctions)",
                      "Builds a test suite from the local functions of a test file. Functions whose names "
                      "start or end with 'test' are tests and take one input, the test case. Functions "
                      "named setupOnce and teardownOnce run once for the file, setup and teardown around "
                      "every test; they share data through testCase.TestData. Write the file as: "
                      "function tests = myTest; tests = functiontests(localfunctions); end"),
    "matlab.unittest.TestSuite.fromFile": ("suite = matlab.unittest.TestSuite.fromFile(file)",
                                           "The tests of one test file as a suite. Selection arguments as "
                                           "for selectIf may follow."),
    "matlab.unittest.TestSuite.fromFolder": (
        ["suite = matlab.unittest.TestSuite.fromFolder(folder)",
         "suite = matlab.unittest.TestSuite.fromFolder(folder, 'IncludingSubfolders', true)"],
        "The tests of every test file in a folder as a suite: test classes, function-based test files, "
        "and scripts whose names start or end with 'test'."),
    "matlab.unittest.TestSuite.fromClass": ("suite = matlab.unittest.TestSuite.fromClass(?ClassName)",
                                            "The Test methods of a test class as a suite (one element per "
                                            "method and parameter combination)."),
    "matlab.unittest.TestSuite.fromMethod": ("suite = matlab.unittest.TestSuite.fromMethod(?ClassName, 'method')",
                                             "The tests that one Test method of a test class produces."),
    "matlab.unittest.TestSuite.fromPackage": (
        ["suite = matlab.unittest.TestSuite.fromPackage(name)",
         "suite = matlab.unittest.TestSuite.fromPackage(name, 'IncludingInnerPackages', true)"],
        "The tests of the test classes and test files inside a package (+name folder)."),
    "matlab.unittest.TestSuite.fromName": ("suite = matlab.unittest.TestSuite.fromName('File/testName')",
                                           "The test with the given full name, or the tests of a cell "
                                           "array of names."),
    "matlab.unittest.TestRunner.withTextOutput": (
        ["runner = matlab.unittest.TestRunner.withTextOutput",
         "runner = matlab.unittest.TestRunner.withTextOutput('OutputDetail', level)"],
        "A test runner that prints progress and failure diagnostics. Run a suite with "
        "results = run(runner, suite) or results = runner.run(suite). 'LoggingLevel' sets which logged "
        "messages are shown."),
    "matlab.unittest.TestRunner.withDefaultPlugins": (
        "runner = matlab.unittest.TestRunner.withDefaultPlugins",
        "A test runner with the standard text output, the same as TestRunner.withTextOutput."),
    "matlab.unittest.TestRunner.withNoPlugins": ("runner = matlab.unittest.TestRunner.withNoPlugins",
                                                 "A silent test runner: results = run(runner, suite) "
                                                 "prints nothing and only returns the results."),
    "matlab.unittest.TestCase.forInteractiveUse": (
        ["testCase = matlab.unittest.TestCase.forInteractiveUse",
         "testCase = matlab.unittest.TestCase.forInteractiveUse(?TestClass)"],
        "A test case to try qualifications at the prompt: each verify, assert or assume call prints "
        "whether it passed, with diagnostics when it failed."),
    # -- verbosity -------------------------------------------------------------------------------------
    **{f"matlab.unittest.Verbosity.{n}": (f"matlab.unittest.Verbosity.{n}", _LEVELS)
       for n in ("None", "Terse", "Concise", "Detailed", "Verbose")},
    **{f"matlab.automation.Verbosity.{n}": (f"matlab.automation.Verbosity.{n}", _LEVELS)
       for n in ("None", "Terse", "Concise", "Detailed", "Verbose")},
    # -- diagnostics -----------------------------------------------------------------------------------
    "matlab.unittest.diagnostics.StringDiagnostic": (
        "d = matlab.unittest.diagnostics.StringDiagnostic(text)",
        "A test diagnostic made of text, shown when the qualification it is passed to fails. Passing the "
        "text itself as the last argument of verifyEqual and friends does the same."),
    "matlab.unittest.diagnostics.FunctionHandleDiagnostic": (
        "d = matlab.unittest.diagnostics.FunctionHandleDiagnostic(@fcn)",
        "A test diagnostic that runs a function when the qualification fails and shows what it prints."),
    "matlab.unittest.diagnostics.DisplayDiagnostic": (
        "d = matlab.unittest.diagnostics.DisplayDiagnostic(value)",
        "A test diagnostic that shows a value the way disp does when the qualification fails."),
    # -- tolerances and constraints --------------------------------------------------------------------
    _C + "AbsoluteTolerance": (f"t = {_C}AbsoluteTolerance(tol)",
                               "A tolerance for IsEqualTo: numbers are equal when abs(actual - expected) "
                               "<= tol. Use as IsEqualTo(expected, 'Within', AbsoluteTolerance(1e-6)). "
                               "Combine with RelativeTolerance using | (either may hold) or & (both)."),
    _C + "RelativeTolerance": (f"t = {_C}RelativeTolerance(tol)",
                               "A tolerance for IsEqualTo: numbers are equal when abs(actual - expected) "
                               "<= tol * abs(expected). Use as IsEqualTo(expected, 'Within', "
                               "RelativeTolerance(0.01))."),
    _C + "IsEqualTo": _con("IsEqualTo", "IsEqualTo(expected)",
                           "Satisfied when the value equals expected in class, size and content (NaN equals "
                           "NaN; cells and structs are compared element by element). Options: 'Within' with "
                           "a tolerance, 'IgnoringCase', 'IgnoringWhitespace' and 'IgnoringFields'."),
    _C + "IsTrue": _con("IsTrue", "IsTrue", "Satisfied by the logical scalar true only."),
    _C + "IsFalse": _con("IsFalse", "IsFalse", "Satisfied by the logical scalar false only."),
    _C + "IsEmpty": _con("IsEmpty", "IsEmpty", "Satisfied when the value has no elements."),
    _C + "IsScalar": _con("IsScalar", "IsScalar", "Satisfied when the value has exactly one element."),
    _C + "IsAnything": _con("IsAnything", "IsAnything", "Satisfied by every value."),
    _C + "IsReal": _con("IsReal", "IsReal", "Satisfied when the value is a real (not complex) array."),
    _C + "IsFinite": _con("IsFinite", "IsFinite", "Satisfied when every element is finite."),
    _C + "IsSparse": _con("IsSparse", "IsSparse", "Satisfied when the value is a sparse matrix."),
    _C + "HasNaN": _con("HasNaN", "HasNaN", "Satisfied when the value contains at least one NaN."),
    _C + "HasInf": _con("HasInf", "HasInf", "Satisfied when the value contains Inf or -Inf."),
    _C + "IsGreaterThan": _con("IsGreaterThan", "IsGreaterThan(bound)",
                               "Satisfied when every element of the value is greater than bound."),
    _C + "IsGreaterThanOrEqualTo": _con("IsGreaterThanOrEqualTo", "IsGreaterThanOrEqualTo(bound)",
                                        "Satisfied when every element of the value is at least bound."),
    _C + "IsLessThan": _con("IsLessThan", "IsLessThan(bound)",
                            "Satisfied when every element of the value is less than bound."),
    _C + "IsLessThanOrEqualTo": _con("IsLessThanOrEqualTo", "IsLessThanOrEqualTo(bound)",
                                     "Satisfied when every element of the value is at most bound."),
    _C + "HasSize": _con("HasSize", "HasSize(sz)", "Satisfied when size(value) equals sz."),
    _C + "HasLength": _con("HasLength", "HasLength(n)", "Satisfied when length(value) equals n."),
    _C + "HasElementCount": _con("HasElementCount", "HasElementCount(n)",
                                 "Satisfied when numel(value) equals n."),
    _C + "HasField": _con("HasField", "HasField(name)", "Satisfied when the value is a struct with that field."),
    _C + "IsOfClass": _con("IsOfClass", "IsOfClass(className)",
                           "Satisfied when the class of the value is exactly className."),
    _C + "IsInstanceOf": _con("IsInstanceOf", "IsInstanceOf(className)",
                              "Satisfied when isa(value, className) is true (subclasses and categories "
                              "such as 'numeric' count)."),
    _C + "IsSameHandleAs": _con("IsSameHandleAs", "IsSameHandleAs(h)",
                                "Satisfied when the value is the very same handle object as h."),
    _C + "ContainsSubstring": _con("ContainsSubstring", "ContainsSubstring(text)",
                                   "Satisfied when the value is text that contains the given text. Options: "
                                   "'IgnoringCase', 'IgnoringWhitespace'."),
    _C + "StartsWithSubstring": _con("StartsWithSubstring", "StartsWithSubstring(text)",
                                     "Satisfied when the value is text that starts with the given text."),
    _C + "EndsWithSubstring": _con("EndsWithSubstring", "EndsWithSubstring(text)",
                                   "Satisfied when the value is text that ends with the given text."),
    _C + "IsSubstringOf": _con("IsSubstringOf", "IsSubstringOf(text)",
                               "Satisfied when the value is text found inside the given text."),
    _C + "Matches": _con("Matches", "Matches(expression)",
                         "Satisfied when the value is text matched by the regular expression."),
    _C + "IsSubsetOf": _con("IsSubsetOf", "IsSubsetOf(set)",
                            "Satisfied when every element of the value is also an element of set."),
    _C + "IsSupersetOf": _con("IsSupersetOf", "IsSupersetOf(set)",
                              "Satisfied when every element of set is also an element of the value."),
    _C + "IsSameSetAs": _con("IsSameSetAs", "IsSameSetAs(set)",
                             "Satisfied when the value and set hold the same elements, in any order."),
    _C + "Throws": _con("Throws", "Throws(identifier)",
                        "Satisfied when the value is a function handle that, called with no inputs, raises "
                        "an error with that identifier."),
    _C + "IssuesNoWarnings": _con("IssuesNoWarnings", "IssuesNoWarnings",
                                  "Satisfied when the value is a function handle that runs without "
                                  "issuing a warning."),
    _C + "IssuesWarnings": _con("IssuesWarnings", "IssuesWarnings({'id1', 'id2'})",
                                "Satisfied when the value is a function handle that issues every listed "
                                "warning identifier."),
    _C + "IsFile": _con("IsFile", "IsFile", "Satisfied when the value is the name of an existing file."),
    _C + "IsFolder": _con("IsFolder", "IsFolder", "Satisfied when the value is the name of an existing folder."),
    # -- selectors -------------------------------------------------------------------------------------
    _S + "HasName": _sel("HasName", "HasName(name)", "Selects tests by full name, such as 'MyTest/testA'."),
    _S + "HasTag": _sel("HasTag", "HasTag(tag)", "Selects tests that carry the tag (the TestTags attribute)."),
    _S + "HasProcedureName": _sel("HasProcedureName", "HasProcedureName(name)",
                                  "Selects tests by the name of the test function or method."),
    _S + "HasBaseFolder": _sel("HasBaseFolder", "HasBaseFolder(folder)",
                               "Selects tests whose file lies in the given folder."),
    _S + "HasSuperclass": _sel("HasSuperclass", "HasSuperclass(className)",
                               "Selects tests of classes derived from the given class."),
    _S + "HasParameterProperty": _sel("HasParameterProperty", "HasParameterProperty(name)",
                                      "Selects parameterized tests that use the given TestParameter property."),
    _S + "HasParameterName": _sel("HasParameterName", "HasParameterName(name)",
                                  "Selects parameterized tests that use a parameter value with that name."),
    _S + "HasParameter": (f"s = {_S}HasParameter('Property', p, 'Name', n, 'Value', v)",
                          "Selects parameterized tests by parameter property, value name and value (each "
                          "option is optional). Use it with selectIf(suite, selector)."),
    # -- fixtures --------------------------------------------------------------------------------------
    _F + "TemporaryFolderFixture": (f"f = {_F}TemporaryFolderFixture",
                                    "Creates a new empty temporary folder, named by f.Folder, and deletes "
                                    "it with everything in it afterwards. Options: 'WithSuffix', text; "
                                    "'PreservingOnFailure', true keeps the folder." + _FIX),
    _F + "WorkingFolderFixture": (f"f = {_F}WorkingFolderFixture",
                                  "Creates a temporary folder (f.Folder) and makes it the current folder; "
                                  "afterwards the previous current folder is restored and the temporary "
                                  "one deleted." + _FIX),
    _F + "CurrentFolderFixture": (f"f = {_F}CurrentFolderFixture(folder)",
                                  "Changes the current folder to folder and changes back afterwards." + _FIX),
    _F + "PathFixture": ([f"f = {_F}PathFixture(folder)",
                          f"f = {_F}PathFixture(folder, 'IncludeSubfolders', true, 'Position', 'end')"],
                         "Adds a folder (or a cell array of folders) to the search path and restores the "
                         "previous path afterwards." + _FIX),
    _F + "SuppressedWarningsFixture": (f"f = {_F}SuppressedWarningsFixture(identifiers)",
                                       "Turns off the display of the given warning identifiers and "
                                       "restores the previous settings afterwards." + _FIX),
    _F + "EnvironmentVariableFixture": (f"f = {_F}EnvironmentVariableFixture(name, value)",
                                        "Sets an environment variable and restores its previous value "
                                        "afterwards." + _FIX),
    # -- plugins ---------------------------------------------------------------------------------------
    _P + "XMLPlugin.producingJUnitFormat": (f"p = {_P}XMLPlugin.producingJUnitFormat(file)",
                                            "A runner plugin that writes the results to an XML file in "
                                            "JUnit format, as continuous-integration systems read it."
                                            + _PLUG),
    _P + "StopOnFailuresPlugin": (f"p = {_P}StopOnFailuresPlugin",
                                  "A runner plugin that ends the run after the first test file or class "
                                  "with a failure (mylab does not open a debugger)." + _PLUG),
    _P + "TestRunProgressPlugin.withVerbosity": (f"p = {_P}TestRunProgressPlugin.withVerbosity(level)",
                                                 "A runner plugin that prints progress at the given "
                                                 "verbosity level (0 to 4)." + _PLUG),
    _P + "LoggingPlugin.withVerbosity": (f"p = {_P}LoggingPlugin.withVerbosity(level)",
                                         "A runner plugin that shows messages logged with "
                                         "log(testCase, ...) up to the given verbosity level." + _PLUG),
    _P + "DiagnosticsOutputPlugin": (f"p = {_P}DiagnosticsOutputPlugin",
                                     "A runner plugin that prints the diagnostics of failed tests." + _PLUG),
    _P + "FailureDiagnosticsPlugin": (f"p = {_P}FailureDiagnosticsPlugin",
                                      "A runner plugin that prints the diagnostics of failed tests." + _PLUG),
}
