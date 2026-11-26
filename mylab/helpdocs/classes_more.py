"""Help text for class introspection, the built-in mixin base classes and related helpers."""

_CD = "matlab.mixin.CustomDisplay"
_HEADER = ("Part of the display of a class derived from matlab.mixin.CustomDisplay. Call it from a getHeader "
           "or getFooter method that you write yourself.")
_CTX = ("One of the values a numArgumentsFromSubscript method receives as its third input. It says where "
        "the indexing expression is used: Statement (on its own line), Expression (inside a larger "
        "expression or a function call) or Assignment (on the left of =). Compare with ==.")
_OPTYPE = ("The Type of one level of an indexing operation handed to parenReference, dotReference or "
           "braceReference (classes derived from matlab.mixin.indexing.RedefinesParen and friends): "
           "Paren for (), Dot for .name and Brace for {}. Compare with ==.")

HELP = {
    # -- introspection ---------------------------------------------------------------------------------
    "metaclass": ("mc = metaclass(obj)",
                  "Returns the meta.class object that describes the class of obj. Its properties include "
                  "Name, Abstract, Sealed, Enumeration, HandleCompatible, ContainingPackage, PropertyList "
                  "(meta.property objects with Name, GetAccess, SetAccess, Dependent, Constant, Hidden, "
                  "Transient, HasDefault, DefaultValue, DefiningClass, ...), MethodList (Name, Access, "
                  "Static, Abstract, Sealed, Hidden, InputNames, OutputNames), EventList, "
                  "EnumerationMemberList and SuperclassList. The same object is written ?ClassName. "
                  "Meta classes compare with == and with < and > (is a subclass of / superclass of)."),
    "meta.class.fromName": ("mc = meta.class.fromName(name)",
                            "The meta.class object of the class called name, the same as ?name but with "
                            "the name given as text. Returns an empty meta.class array when there is no "
                            "such class."),
    "meta.package.fromName": ("mp = meta.package.fromName(name)",
                              "Describes the package name (a +name folder on the path): Name, ClassList, "
                              "FunctionList, PackageList and ContainingPackage. Empty when the package "
                              "does not exist."),
    "superclasses": (["superclasses(obj)", "names = superclasses(className)"],
                     "Lists the names of all superclasses of a class, nearest first, including built-in "
                     "bases such as handle. With an output it returns a cell column of names; without one "
                     "it prints them."),
    "events": (["events(obj)", "names = events(className)"],
               "Lists the events of a class (those declared in events blocks of the class and its "
               "superclasses, plus ObjectBeingDestroyed for handle classes)."),
    "event.hasListener": ("tf = event.hasListener(obj, eventName)",
                          "True when at least one enabled listener is attached to the event of the handle "
                          "object obj."),
    # -- dominance of old-style classes ----------------------------------------------------------------
    "superiorto": ("superiorto('class1', 'class2', ...)",
                   "Called in the constructor of a class kept in an @folder: when an operator or function "
                   "is given objects of this class and of one of the named classes, the method of this "
                   "class runs. In classdef files use the InferiorClasses attribute instead."),
    "inferiorto": ("inferiorto('class1', 'class2', ...)",
                   "Called in the constructor of a class kept in an @folder: the named classes take "
                   "precedence over this class when their objects appear together in a call."),
    # -- indexing --------------------------------------------------------------------------------------
    "numArgumentsFromSubscript": (
        "n = numArgumentsFromSubscript(obj, s, indexingContext)",
        "How many values an indexing expression on obj produces where a comma-separated list is expected, "
        "for example in [a, b] = obj{1:2}. A class that overloads subsref can define this method to state "
        "the count; calling builtin('numArgumentsFromSubscript', obj, s, ctx) gives the standard answer. "
        "s is the subscript struct array that subsref receives."),
    "matlab.mixin.util.IndexingContext.Statement": ("matlab.mixin.util.IndexingContext.Statement", _CTX),
    "matlab.mixin.util.IndexingContext.Expression": ("matlab.mixin.util.IndexingContext.Expression", _CTX),
    "matlab.mixin.util.IndexingContext.Assignment": ("matlab.mixin.util.IndexingContext.Assignment", _CTX),
    "matlab.mixin.util.IndexingContext.Unknown": ("matlab.mixin.util.IndexingContext.Unknown", _CTX),
    "matlab.indexing.IndexingOperationType.Paren": ("matlab.indexing.IndexingOperationType.Paren", _OPTYPE),
    "matlab.indexing.IndexingOperationType.Dot": ("matlab.indexing.IndexingOperationType.Dot", _OPTYPE),
    "matlab.indexing.IndexingOperationType.Brace": ("matlab.indexing.IndexingOperationType.Brace", _OPTYPE),
    # -- custom display --------------------------------------------------------------------------------
    "matlab.mixin.util.PropertyGroup": (
        ["g = matlab.mixin.util.PropertyGroup(propertyNames)",
         "g = matlab.mixin.util.PropertyGroup(propertyNames, title)",
         "g = matlab.mixin.util.PropertyGroup(structOfValues, title)"],
        "A group of properties for the display of a class derived from matlab.mixin.CustomDisplay. Return "
        "one or more groups from a getPropertyGroups method: a cell array of property names shows those "
        "properties of the object; a struct shows its own field names and values instead. The optional "
        "title is printed above the group."),
    f"{_CD}.getClassNameForHeader": (f"name = {_CD}.getClassNameForHeader(obj)",
                                     "The class name of obj as text, for use in a custom header. " + _HEADER),
    f"{_CD}.getSimpleHeader": (f"h = {_CD}.getSimpleHeader(obj)",
                               "The standard first line of an object display, such as 'Name with "
                               "properties:'. " + _HEADER),
    f"{_CD}.getDetailedHeader": (f"h = {_CD}.getDetailedHeader(obj)",
                                 "The standard header of an object display (in mylab the same text as "
                                 "getSimpleHeader). " + _HEADER),
    f"{_CD}.getSimpleFooter": (f"f = {_CD}.getSimpleFooter(obj)",
                               "The standard footer of an object display, which is empty text. " + _HEADER),
    f"{_CD}.getDetailedFooter": (f"f = {_CD}.getDetailedFooter(obj)",
                                 "The standard footer of an object display, which is empty text. " + _HEADER),
    f"{_CD}.getHandleText": (f"t = {_CD}.getHandleText()", "The word 'handle', for custom headers. " + _HEADER),
    f"{_CD}.getDeletedHandleText": (f"t = {_CD}.getDeletedHandleText()",
                                    "The words 'handle to deleted', for the header of a deleted handle "
                                    "object. " + _HEADER),
    f"{_CD}.convertDimensionsToString": (f"s = {_CD}.convertDimensionsToString(obj)",
                                         "The size of obj as text such as '2×3', for custom headers of "
                                         "object arrays. " + _HEADER),
    f"{_CD}.displayPropertyGroups": (f"{_CD}.displayPropertyGroups(obj, groups)",
                                     "Prints the property groups (matlab.mixin.util.PropertyGroup objects) "
                                     "of obj the way the standard display does. Use it in a "
                                     "displayScalarObject method that prints its own header."),
    # -- functions -------------------------------------------------------------------------------------
    "localfunctions": ("fcns = localfunctions",
                       "Returns a cell column of handles to the local functions of the file that calls it "
                       "(the main function of a function file is not included). Typical use: "
                       "tests = functiontests(localfunctions)."),
    "mlock": ("mlock",
              "Called inside a function: protects that function's persistent variables from clear "
              "functions, clear all and clear name until munlock is called."),
    "munlock": (["munlock", "munlock(functionName)"],
                "Removes the protection that mlock gave to the calling function (or to the named one), so "
                "clear can reset its persistent variables again."),
    "mislocked": (["tf = mislocked", "tf = mislocked(functionName)"],
                  "True when the calling function (or the named one) is protected by mlock."),
    # -- validators ------------------------------------------------------------------------------------
    "mustBeScalarOrEmpty": ("mustBeScalarOrEmpty(value)", "Raises an error unless value has one element or none."),
    "mustBeRow": ("mustBeRow(value)", "Raises an error unless value is a row vector (1-by-N, N may be 0)."),
    "mustBeColumn": ("mustBeColumn(value)", "Raises an error unless value is a column vector (N-by-1)."),
    "mustBeMatrix": ("mustBeMatrix(value)", "Raises an error unless value has at most two dimensions."),
    "mustBeFloat": ("mustBeFloat(value)", "Raises an error unless value is of class double or single."),
    "mustBeUnderlyingType": ("mustBeUnderlyingType(value, typeNames)",
                             "Raises an error unless the class of value (for an object derived from a "
                             "built-in type, the class of its data) is one of typeNames, a name or a cell "
                             "or string array of names."),
    "mustBeNonmissing": ("mustBeNonmissing(value)",
                         "Raises an error if value contains a missing value: NaN in numbers, <missing> in "
                         "strings, NaT in datetimes."),
    "mustBeNonsparse": ("mustBeNonsparse(value)", "Raises an error if value is a sparse matrix."),
    "mustBeSparse": ("mustBeSparse(value)", "Raises an error unless value is a sparse matrix."),
    "mustBeValidVariableName": ("mustBeValidVariableName(value)",
                                "Raises an error unless value is text that could be a variable name: a "
                                "letter followed by letters, digits or underscores, and not a keyword."),
    "mustBeBetween": (["mustBeBetween(value, lower, upper)", "mustBeBetween(value, lower, upper, intervalType)"],
                      "Raises an error unless every element of value lies between lower and upper. "
                      "intervalType is 'closed' (default, both ends allowed), 'open', 'openleft', "
                      "'openright', 'closedleft' or 'closedright'."),
}
