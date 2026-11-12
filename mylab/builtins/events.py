"""Class events (notify, addlistener, listener, event.EventData) and enumeration listing."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, Struct, char
from ._registry import builtin
from ._util import bool_scalar, is_text, text


class EventData(MObject):
    """The second argument a listener callback receives: which object and which event."""

    is_handle = True

    def __init__(self, source: Any = None, name: str = "", extra: dict | None = None):
        self.fields: dict[str, Any] = {"Source": source if source is not None else np.zeros((0, 0)),
                                       "EventName": char(name)}
        self.fields.update(extra or {})

    def class_name(self) -> str:
        return "event.EventData"

    def isa_class(self, name: str) -> bool:
        return name in ("event.EventData", "handle")

    def get_field(self, interp, name: str) -> Any:
        if name in self.fields:
            return self.fields[name]
        raise MatlabError(f"Unrecognized property '{name}' for class 'event.EventData'.")

    def set_field(self, interp, name: str, value: Any) -> "EventData":
        self.fields[name] = value
        return self

    def display_lines(self) -> list[str]:
        from ..display import summary
        w = max(len(n) for n in self.fields)
        return ["  EventData with properties:", ""] + \
            [f"    {n:>{w}}: {summary(v, False)}" for n, v in self.fields.items()]


class Listener(MObject):
    """What addlistener returns. Set Enabled to false, or delete it, to stop the callback."""

    is_handle = True

    def __init__(self, source: Any, event: str, callback: Any):
        self.source = source
        self.event = event
        self.callback = callback
        self.enabled = True

    def class_name(self) -> str:
        return "event.listener"

    def isa_class(self, name: str) -> bool:
        return name in ("event.listener", "event.proplistener", "handle")

    def active(self) -> bool:
        return self.enabled and not getattr(self, "_deleted", False)

    def get_field(self, interp, name: str) -> Any:
        if name == "Enabled":
            return bool_scalar(self.enabled)
        if name == "EventName":
            return char(self.event)
        if name == "Callback":
            return self.callback
        if name == "Source":
            return Cell.from_list([self.source])
        if name == "Recursive":
            return bool_scalar(False)
        raise MatlabError(f"Unrecognized property '{name}' for class 'event.listener'.")

    def set_field(self, interp, name: str, value: Any) -> "Listener":
        if name == "Enabled":
            self.enabled = bool(np.asarray(value).flat[0])
        elif name == "Callback":
            self.callback = value
        elif name != "Recursive":
            raise MatlabError(f"Cannot set property '{name}' of an event listener.")
        return self

    def display_lines(self) -> list[str]:
        return ["  listener with properties:", "",
                f"    EventName: '{self.event}'",
                f"      Enabled: {int(self.enabled)}"]


def _source_object(v: Any, what: str):
    from ..classdef import UserObject
    if not isinstance(v, UserObject) or not v.is_handle:
        raise MatlabError(f"{what}: the first argument must be an object of a handle class "
                          "(classdef Name < handle) that declares events.")
    return v


def _fire(interp, listeners: list, args: list) -> None:
    for lst in list(listeners):
        if not lst.active():
            continue
        cb = lst.callback
        if isinstance(cb, Cell):  # {@fn, extra1, extra2}
            items = cb.values_f()
            interp.call_function_value(items[0], args + items[1:], 0)
        else:
            interp.call_function_value(cb, args, 0)


def property_event(interp, obj, prop: str, when: str) -> None:
    """Run PreSet/PostSet listeners of a property (called when the property is assigned)."""
    listeners = (obj._listeners or {}).get(f"{prop}:{when}")
    if not listeners:
        return
    src = Struct.scalar({"Name": char(prop), "DefiningClass": char(obj.cls.name)})
    data = EventData(src, when, {"AffectedObject": obj})
    _fire(interp, listeners, [src, data])


@builtin("notify")
def _notify(interp, args, nargout):
    from ..classdef import UserObject
    if len(args) < 2:
        raise MatlabError("notify needs an object and an event name: notify(obj, 'EventName').")
    obj = _source_object(args[0], "notify")
    name = text(args[1], "event name")
    if name not in obj.cls.events and name != "ObjectBeingDestroyed":
        raise MatlabError(f"Event '{name}' is not defined for class '{obj.cls.name}'. Declare it in an "
                          "events block of the classdef.")
    listeners = (obj._listeners or {}).get(name)
    if not listeners:
        return None
    if len(args) > 2:
        data = args[2]
        if isinstance(data, UserObject):  # a subclass of event.EventData
            data.props["Source"] = obj
            data.props["EventName"] = char(name)
        elif isinstance(data, EventData):
            data.fields["Source"] = obj
            data.fields["EventName"] = char(name)
    else:
        data = EventData(obj, name)
    _fire(interp, listeners, [obj, data])
    return None


def _add_listener(interp, args, what: str) -> Listener:
    if len(args) < 3:
        raise MatlabError(f"{what} needs an object, an event name and a callback: "
                          f"{what}(obj, 'EventName', @callback).")
    obj = _source_object(args[0], what)
    name = text(args[1], "event name")
    if len(args) >= 4 and is_text(args[2]):  # property listener: (obj, 'Prop', 'PostSet', @cb)
        when = text(args[2])
        if when not in ("PreSet", "PostSet", "PreGet", "PostGet"):
            raise MatlabError(f"Unknown property event '{when}'; use 'PreSet' or 'PostSet'.")
        if name not in obj.cls.prop_defs:
            raise MatlabError(f"'{name}' is not a property of class '{obj.cls.name}'.")
        key, callback = f"{name}:{when}", args[3]
    else:
        if name not in obj.cls.events and name != "ObjectBeingDestroyed":
            raise MatlabError(f"Event '{name}' is not defined for class '{obj.cls.name}'. Declare it in an "
                              "events block of the classdef.")
        key, callback = name, args[2]
    if not isinstance(callback, (FunctionHandle, Cell)) and not is_text(callback):
        raise MatlabError(f"{what}: the callback must be a function handle, e.g. @(src, evt) disp('fired').")
    lst = Listener(obj, key, callback)
    if obj._listeners is None:
        obj._listeners = {}
    obj._listeners.setdefault(key, []).append(lst)
    return lst


@builtin("addlistener")
def _addlistener(interp, args, nargout):
    lst = _add_listener(interp, args, "addlistener")
    return lst if nargout else None


@builtin("listener", "event.listener", "event.proplistener")
def _listener(interp, args, nargout):
    return _add_listener(interp, args, "listener")


@builtin("event.EventData")
def _event_data(interp, args, nargout):
    return EventData()


@builtin("enumeration")
def _enumeration(interp, args, nargout):
    from ..classdef import ObjectArray, UserObject
    if not args:
        raise MatlabError("enumeration needs a class name or an object.")
    v = args[0]
    if isinstance(v, (UserObject, ObjectArray)):
        cls = v.cls
    else:
        cls = interp.find_class(text(v, "class name"))
        if cls is None:
            raise MatlabError(f"Class '{text(v)}' was not found.")
    if not cls.enums:
        if nargout:
            return ObjectArray(np.empty((0, 1), dtype=object), cls), Cell.empty(0, 1)
        interp.out(f"\nClass {cls.name} has no enumeration members.\n\n")
        return None
    members = cls.enum_members(interp)
    if not nargout:
        interp.out(f"\nEnumeration members for class '{cls.name}':\n\n" +
                   "".join(f"    {n}\n" for n in members) + "\n")
        return None
    data = np.empty((len(members), 1), dtype=object)
    for k, obj in enumerate(members.values()):
        data[k, 0] = obj
    arr = data.flat[0] if data.size == 1 else ObjectArray(data, cls)
    return arr, Cell.from_list([char(n) for n in members], column=True)


@builtin("isenum")
def _isenum(interp, args, nargout):
    from ..classdef import ObjectArray, UserObject
    v = args[0]
    return bool_scalar(isinstance(v, (UserObject, ObjectArray)) and v.cls is not None and bool(v.cls.enums))
