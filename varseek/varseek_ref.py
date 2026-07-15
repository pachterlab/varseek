"""varseek ref — a thin, backwards-compatible redirect to varseek build.

Historically, ``vk ref`` wrapped ``vk build`` (to create the VCRS fasta/t2g) and ``kb ref``
(to create the kallisto index). That index-creation step now lives inside ``vk build`` itself
(on by default; disable with the hidden ``dont_create_index`` flag), so ``vk ref`` is simply an
alias for ``vk build`` and is kept only for backwards compatibility. See ``varseek_build.py``.
"""

from .constants import varseek_ref_only_allowable_kb_ref_arguments  # re-exported for backwards compatibility (e.g. tests/check_pairwise_argument_names.py)
from .varseek_build import build, downloadable_references  # re-exported for backwards compatibility

__all__ = ["ref", "varseek_ref_only_allowable_kb_ref_arguments", "downloadable_references"]


def ref(*args, **kwargs):
    """Backwards-compatible alias for :func:`varseek.build`.

    ``vk ref`` used to wrap ``vk build`` and ``kb ref``; the index-creation step is now part of
    ``vk build`` itself (see the ``dont_create_index`` argument), so ``ref`` just forwards
    everything to ``build``. All arguments accepted by ``vk build`` are accepted here.
    """
    return build(*args, **kwargs)
