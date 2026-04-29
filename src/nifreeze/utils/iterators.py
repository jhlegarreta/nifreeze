# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Iterators to traverse the volumes in a 4D image."""

import random
from itertools import chain, zip_longest
from typing import Iterator, Sequence

DEFAULT_ROUND_DECIMALS = 2
"""Round decimals to use when comparing values to be sorted for iteration purposes."""

SIZE_KWARG = "size"
"""Size keyword argument name."""
BVALS_KWARG = "bvals"
"""b-vals keyword argument name."""
UPTAKE_KWARG = "uptake"
"""Uptake keyword argument name."""

MODALITY_SPECIFIC_SIZE_KEYS = (BVALS_KWARG, UPTAKE_KWARG)
"""Modality-specific keys used to infer the number of volumes in a dataset."""

SIZE_KEYS = (SIZE_KWARG, BVALS_KWARG, UPTAKE_KWARG)
"""Keys that may be used to infer the number of volumes in a dataset. When the
size of the structure to iterate over is not given explicitly, these keys
correspond to properties that distinguish one imaging modality from another, and
are part of the 4th axis (e.g. diffusion gradients in DWI or update in PET)."""

SIZE_KEYS_DOC = """
size : :obj:`int`, optional
    Size of the structure to iterate over.
bvals : :obj:`list`, optional
    List of b-values corresponding to all orientations of a DWI dataset.
uptake : :obj:`list`, optional
    List of uptake values corresponding to all volumes of the dataset.
"""

ITERATOR_NOTES = """
Only one of the size-related parameters (``size``, ``bvals``, or ``uptake``)
may be provided at a time. If ``bvals`` or ``uptake`` is given, it takes
precedence over ``size``. ``size`` is only used when no modality-specific
parameter is provided. If more than one modality-specific parameter (``bvals``
or ``uptake``) is provided at the same time, a :exc:`ValueError` will be
raised.
"""

ITERATOR_SIZE_ERROR_MSG = (
    f"None of {SIZE_KEYS} were provided to infer size: cannot build iterator without size."
)
"""Iterator size argument error message."""
ITERATOR_MULTIPLICITY_ERROR_MSG = (
    f"Only one of the modality-specific size parameters "
    f"({', '.join(MODALITY_SPECIFIC_SIZE_KEYS)}) may be provided at a time."
)
"""Iterator multiplicity error message."""
KWARG_ERROR_MSG = "Keyword argument {kwarg} is required."
"""Iterator keyword argument error message."""


def _resolve_feature(kwargs: dict, allowed_features: tuple = SIZE_KEYS) -> str:
    provided = [k for k in allowed_features if kwargs.get(k) is not None]
    if not provided:
        raise ValueError(ITERATOR_SIZE_ERROR_MSG)

    # Modality-specific keys take precedence over size
    modality_specific = [k for k in provided if k in MODALITY_SPECIFIC_SIZE_KEYS]
    if modality_specific:
        if len(modality_specific) > 1:
            raise ValueError(ITERATOR_MULTIPLICITY_ERROR_MSG)
        return modality_specific[0]

    return SIZE_KWARG


_resolve_feature.__doc__ = f"""
Determine which size-related feature to use from the provided kwargs.

Modality-specific keys (``{MODALITY_SPECIFIC_SIZE_KEYS}``) take precedence
over ``{SIZE_KWARG}``. If more than one modality-specific key is provided at
the same time, a :exc:`ValueError` is raised.

Parameters
----------
kwargs : :obj:`dict`
    The keyword arguments to inspect.
allowed_features : :obj:`tuple` of :obj:`str`, optional
    Keys accepted to define the domain length (e.g., ``("{SIZE_KWARG}",)`` or
    ``("{BVALS_KWARG}", "{UPTAKE_KWARG}")``). Exactly one of these keys must
    be present in ``kwargs`` with a non-:obj:`None` value.

Returns
-------
:obj:`str`
    The key of the selected feature.

Raises
------
:exc:`ValueError`
    If no size-related key is provided, or if more than one
    modality-specific key is provided at the same time.

Examples
--------
>>> _resolve_feature({{"size": 4}})
'{SIZE_KWARG}'
>>> _resolve_feature({{"bvals": [0, 1000, 2000, 3000]}})
'{BVALS_KWARG}'
>>> _resolve_feature({{"uptake": [0.1, 0.12, 0.3, 0.4]}})
'{UPTAKE_KWARG}'
>>> _resolve_feature({{"size": 4, "bvals": [10, 20, 30, 40]}})
'{BVALS_KWARG}'
>>> _resolve_feature({{"size": 4, "uptake": [0.1, 0.12, 0.3, 0.4]}})
'{UPTAKE_KWARG}'
"""


def _get_size_from_kwargs(kwargs: dict) -> int:
    """Extract the size from kwargs, ensuring only one key is used.

    Parameters
    ----------
    kwargs : :obj:`dict`
        The keyword arguments passed to the iterator function.

    Returns
    -------
    :obj:`int`
        The inferred size.

    Raises
    ------
    :exc:`ValueError`
        If size could not be extracted.
    """
    feature = _resolve_feature(kwargs)
    value = kwargs[feature]
    return value if isinstance(value, int) else len(value)


def linear_iterator(**kwargs) -> Iterator[int]:
    size = _get_size_from_kwargs(kwargs)
    return (s for s in range(size))


linear_iterator.__doc__ = f"""
Traverse the dataset volumes in ascending order.

Other Parameters
----------------
{SIZE_KEYS_DOC}

Notes
-----
{ITERATOR_NOTES}

Yields
------
:obj:`int`
    The next index.

Examples
--------
>>> list(linear_iterator(size=10))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

"""


def random_iterator(**kwargs) -> Iterator[int]:
    size = _get_size_from_kwargs(kwargs)

    _seed = kwargs.get("seed", None)
    _seed = 20210324 if _seed is True else _seed

    random.seed(None if _seed is False else _seed)

    index_order = list(range(size))
    random.shuffle(index_order)
    return (x for x in index_order)


random_iterator.__doc__ = f"""
Traverse the dataset volumes randomly.

If the ``seed`` key is present in the keyword arguments, initializes the seed
of Python's ``random`` pseudo-random number generator library with the given
value. Specifically, if :obj:`False`, :obj:`None` is used as the seed;
if :obj:`True`, a default seed value is used.

Other Parameters
----------------
seed : :obj:`int`, :obj:`bool`, :obj:`str`, or :obj:`None`
    If :obj:`int` or :obj:`str` or :obj:`None`, initializes the seed of Python's
    random generator with the given value. If :obj:`False`, the random generator
    is passed :obj:`None`. If :obj:`True`, a default seed value is set.

{SIZE_KEYS_DOC}

Notes
-----
{ITERATOR_NOTES}

Yields
------
:obj:`int`
    The next index.

Examples
--------
>>> list(random_iterator(size=15, seed=0))  # seed is 0
[1, 10, 9, 5, 11, 2, 3, 7, 8, 4, 0, 14, 12, 6, 13]
>>>  # seed is True -> the default value 20210324 is set
>>> list(random_iterator(size=15, seed=True))
[1, 12, 14, 5, 0, 11, 10, 9, 7, 8, 3, 13, 2, 6, 4]
>>> list(random_iterator(size=15, seed=20210324))
[1, 12, 14, 5, 0, 11, 10, 9, 7, 8, 3, 13, 2, 6, 4]
>>> list(random_iterator(size=15, seed=42))  # seed is 42
[8, 13, 7, 6, 14, 12, 5, 2, 9, 3, 4, 11, 0, 1, 10]

"""


def _value_iterator(
    values: Sequence[float], ascending: bool, round_decimals: int = DEFAULT_ROUND_DECIMALS
) -> Iterator[int]:
    """
    Traverse the given values in ascending or descenting order.

    Parameters
    ----------
    values : :obj:`Sequence`
        List of values to traverse.
    ascending : :obj:`bool`
        If :obj:`True`, traverse in ascending order; traverse in descending order
        otherwise.
    round_decimals : :obj:`int`, optional
        Number of decimals to round values for sorting.

    Yields
    ------
    :obj:`int`
        The next index.

    Examples
    --------
    >>> list(_value_iterator([0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0], True))
    [0, 1, 8, 4, 5, 2, 3, 6, 7]
    >>> list(_value_iterator([0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0], False))
    [7, 6, 3, 2, 5, 4, 8, 1, 0]
    >>> list(_value_iterator([-1.23, 1.06, 1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05], True))
    [4, 0, 6, 5, 2, 8, 1, 7, 3]
    >>> list(_value_iterator([-1.23, 1.06, 1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05], False))
    [3, 7, 1, 8, 2, 5, 6, 0, 4]

    """

    indexed_vals = sorted(
        ((round(v, round_decimals), i) for i, v in enumerate(values)), reverse=not ascending
    )
    return (index[1] for index in indexed_vals)


def monotonic_value_iterator(*_, **kwargs) -> Iterator[int]:
    try:
        feature = next(k for k in (BVALS_KWARG, UPTAKE_KWARG) if kwargs.get(k) is not None)
    except StopIteration:
        raise TypeError(KWARG_ERROR_MSG.format(kwarg=f"{BVALS_KWARG} or {UPTAKE_KWARG}"))

    ascending = feature == BVALS_KWARG
    values = kwargs[feature]
    return _value_iterator(
        values,
        ascending=ascending,
        round_decimals=kwargs.get("round_decimals", DEFAULT_ROUND_DECIMALS),
    )


monotonic_value_iterator.__doc__ = f"""
Traverse the volumes by increasing b-value in a DWI dataset or by decreasing
uptake value in a PET dataset.

This function requires ``bvals`` or ``uptake`` be a keyword argument to generate
the volume sequence. The b-values are assumed to all orientations in a DWI
dataset, and uptake uptake values correspond to all volumes in a PET dataset.

It is assumed that each uptake value corresponds to a single volume, and that
this value summarizes the uptake of the volume in a meaningful way, e.g. a mean
value across the entire volume.

Other Parameters
----------------
{SIZE_KEYS_DOC}

Notes
-----
{ITERATOR_NOTES}

Yields
------
:obj:`int`
    The next index.

Examples
--------
>>> list(monotonic_value_iterator(bvals=[0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0]))
[0, 1, 8, 4, 5, 2, 3, 6, 7]
>>> list(monotonic_value_iterator(uptake=[-1.23, 1.06, 1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05]))
[3, 7, 1, 8, 2, 5, 6, 0, 4]
"""


def centralsym_iterator(**kwargs) -> Iterator[int]:
    size = _get_size_from_kwargs(kwargs)

    linear = list(range(size))
    return (
        x
        for x in chain.from_iterable(
            zip_longest(
                linear[size // 2 :],
                reversed(linear[: size // 2]),
            )
        )
        if x is not None
    )


centralsym_iterator.__doc__ = f"""
Traverse the dataset starting from the center and alternatingly progressing to the sides.

Other Parameters
----------------
{SIZE_KEYS_DOC}

Notes
-----
{ITERATOR_NOTES}

Examples
--------
>>> list(centralsym_iterator(size=10))
[5, 4, 6, 3, 7, 2, 8, 1, 9, 0]
>>> list(centralsym_iterator(size=11))
[5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10]
"""
