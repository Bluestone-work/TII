# -*- coding: utf-8 -*-
"""Compatibility shim for un-pickling checkpoints written with an older `packaging`.

Some p0_core_v1 checkpoints (seed2/seed3) were trained on a different machine whose
`packaging` library pickled `packaging.version.Version` via a legacy tuple state
(``__getstate__`` returning the ``_version`` tuple).  Newer `packaging` (>=22)
stores state in ``__dict__`` and its ``BUILD`` step expects a dict, so loading the
old pickles raises ``UnpicklingError('state is not a dictionary')``.

Importing this module installs a ``Version.__setstate__`` that accepts both the new
dict form and the legacy tuple form, reconstructing the version from its release
tuple.  It is idempotent and only affects the current process — it does not touch
the installed environment.
"""
from __future__ import annotations

import packaging.version as _pv


def _install() -> None:
    if getattr(_pv.Version, "_dgta_pickle_compat", False):
        return

    def __setstate__(self, state):  # noqa: N807
        # New packaging: state is a plain dict (_version, _key, ...).
        if isinstance(state, dict):
            self.__dict__.update(state)
            return
        # Legacy packaging: state is the _version tuple, typically shaped like
        # ((epoch_or_release...), pre, post, dev, local). We only need a valid
        # Version equivalent for checkpoint metadata, so rebuild from the release
        # numbers when possible and fall back to stashing the raw state.
        try:
            release = state[0] if isinstance(state[0], tuple) else state
            version_str = ".".join(str(part) for part in release)
            rebuilt = _pv.Version(version_str)
            self.__dict__.update(rebuilt.__dict__)
        except Exception:
            # Last resort: keep the raw legacy state so unpickling still succeeds.
            self.__dict__["_legacy_state"] = state

    _pv.Version.__setstate__ = __setstate__
    _pv.Version._dgta_pickle_compat = True


_install()
