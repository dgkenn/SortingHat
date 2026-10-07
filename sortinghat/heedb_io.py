"""Compatibility shim: the HEEDB loader was generalised into ``sortinghat.data_io``."""

from .data_io import *  # noqa: F401,F403
from .data_io import _AWS_KEY_ID  # noqa: F401
