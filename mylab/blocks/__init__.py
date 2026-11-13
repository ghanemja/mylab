"""mylab Blocks: block-diagram models and their simulation.

A model is a set of blocks joined by signal lines. It is stored as readable
JSON (``name.mlb.json``). :mod:`mylab.blocks.model` holds the data structure,
:mod:`mylab.blocks.library` the block types, :mod:`mylab.blocks.engine` the
simulator and :mod:`mylab.blocks.slx` a reader for simple ``.slx`` files.
"""

from .model import Block, Line, Model, ModelError, System, build, load_model, save_model  # noqa: F401
from .library import TYPES, catalog, find_type  # noqa: F401
from .engine import SimError, SimResult, compile_model, diagnose, linearize, simulate, trim  # noqa: F401
