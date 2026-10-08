"""Frozen-encoder embeddings for the representation ladder (``emb.<family>.<j>`` columns).

Currently one family: public pretrained CBraMod (``cbramod``). ``torch`` is an OPTIONAL dependency
(``pip install torch --index-url https://download.pytorch.org/whl/cpu``); importing this package does not need it.
Developed and tested on synthetic EEG only (CLAUDE.md rule 1); real recordings are embedded by the human-run
``scripts/extract_embeddings.py``. See ``docs/embeddings.md``.
"""

from .cbramod import (CBRAMOD_CHANNELS, EMB_DIM, EmbedConfig, Embedder, embed_recording, emb_columns,  # noqa: F401
                      part_columns)
from .store import load_primary_embeddings  # noqa: F401
