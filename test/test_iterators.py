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

import re

import pytest

from nifreeze.utils.iterators import (
    BVALS_KWARG,
    ITERATOR_SIZE_ERROR_MSG,
    KWARG_ERROR_MSG,
    UPTAKE_KWARG,
    _value_iterator,
    centralsym_iterator,
    linear_iterator,
    monotonic_value_iterator,
    random_iterator,
)


@pytest.mark.parametrize(
    "values, ascending, round_decimals, expected",
    [
        # Simple integers
        ([1, 2, 3], True, 2, [0, 1, 2]),
        ([1, 2, 3], False, 2, [2, 1, 0]),
        # Repeated values
        ([2, 1, 2, 1], True, 2, [1, 3, 0, 2]),
        ([2, 1, 2, 1], False, 2, [2, 0, 3, 1]),  # Ties are reversed due to reverse=True
        # Floats
        ([1.01, 1.02, 0.99], True, 2, [2, 0, 1]),
        ([1.01, 1.02, 0.99], False, 2, [1, 0, 2]),
        # Floats with rounding
        (
            [1.001, 1.002, 0.999],
            True,
            2,
            [0, 1, 2],
        ),  # All round to 1.00 (round_decimals=2), so original order
        (
            [1.001, 1.002, 0.999],
            True,
            4,
            [2, 0, 1],
        ),
        (
            [1.001, 1.002, 0.999],
            False,
            2,
            [2, 1, 0],
        ),  # All round to 1.00 (round_decimals=2), ties are reversed due to reverse=True
        (
            [1.001, 1.002, 0.999],
            False,
            4,
            [1, 0, 2],
        ),
        # Negative and positive
        ([-1.2, 0.0, 3.4, -1.2], True, 2, [0, 3, 1, 2]),
        ([-1.2, 0.0, 3.4, -1.2], False, 2, [2, 1, 3, 0]),  # Ties are reversed due to reverse=True
    ],
)
def test_value_iterator(values, ascending, round_decimals, expected):
    result = list(_value_iterator(values, ascending=ascending, round_decimals=round_decimals))
    assert result == expected


def test_linear_iterator_error():
    with pytest.raises(ValueError, match=re.escape(ITERATOR_SIZE_ERROR_MSG)):
        list(linear_iterator())


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"size": 4}, [0, 1, 2, 3]),
        ({"bvals": [0, 1000, 2000, 3000]}, [0, 1, 2, 3]),
        ({"uptake": [-1.02, -0.56, 0.43, 1.16]}, [0, 1, 2, 3]),
    ],
)
def test_linear_iterator(kwargs, expected):
    assert list(linear_iterator(**kwargs)) == expected


def test_random_iterator_error():
    with pytest.raises(ValueError, match=re.escape(ITERATOR_SIZE_ERROR_MSG)):
        list(random_iterator())


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"size": 5, "seed": 1234}, [1, 2, 4, 0, 3]),
        ({"bvals": [0, 1000, 2000, 3000], "seed": 42}, [2, 1, 3, 0]),
        ({"uptake": [-1.02, -0.56, 0.43, 1.16], "seed": True}, [3, 0, 1, 2]),
    ],
)
def test_random_iterator(kwargs, expected):
    obtained = list(random_iterator(**kwargs))
    assert obtained == expected
    # Determinism check
    assert obtained == list(random_iterator(**kwargs))


def test_centralsym_iterator_error():
    with pytest.raises(ValueError, match=re.escape(ITERATOR_SIZE_ERROR_MSG)):
        list(random_iterator())


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"size": 6}, [3, 2, 4, 1, 5, 0]),
        ({"bvals": [1000] * 6}, [3, 2, 4, 1, 5, 0]),
        ({"bvals": [0, 700, 1000, 2000, 3000]}, [2, 1, 3, 0, 4]),
        ({"bvals": [0, 1000, 700, 2000, 3000]}, [2, 1, 3, 0, 4]),
        ({"uptake": [0.32, 0.27, -0.12]}, [1, 0, 2]),
        ({"uptake": [-1.02, -0.56, 0.43, 0.89, 1.16]}, [2, 1, 3, 0, 4]),
    ],
)
def test_centralsym_iterator(kwargs, expected):
    # The centralsym_iterator's output order depends only on the length
    assert list(centralsym_iterator(**kwargs)) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"bvals": None},
        {"uptake": None},
        {"bvals": None, "uptake": None},
    ],
)
def test_monotonic_value_iterator_error(kwargs):
    with pytest.raises(
        TypeError, match=KWARG_ERROR_MSG.format(kwarg=f"{BVALS_KWARG} or {UPTAKE_KWARG}")
    ):
        monotonic_value_iterator(**kwargs)


def test_monotonic_value_iterator_sorting_preference():
    result = list(monotonic_value_iterator(bvals=[700, 1000], uptake=[0.14, 0.23, 0.47]))
    assert result == [0, 1]

    result = list(monotonic_value_iterator(bvals=None, uptake=[0.14, 0.23, 0.47]))
    assert result == [2, 1, 0]


@pytest.mark.parametrize(
    "feature, values, expected",
    [
        ("bvals", [0, 700, 1200], [0, 1, 2]),
        ("bvals", [0, 0, 1000, 700], [0, 1, 3, 2]),
        ("bvals", [0, 1000, 1500, 700, 2000], [0, 3, 1, 2, 4]),
        ("uptake", [0.3, 0.2, 0.1], [0, 1, 2]),
        ("uptake", [0.2, 0.1, 0.3], [2, 1, 0]),
        ("uptake", [-1.02, 1.16, -0.56, 0.43], [1, 3, 2, 0]),
    ],
)
def test_monotonic_value_iterator(feature, values, expected):
    obtained = list(monotonic_value_iterator(**{feature: values}))
    assert set(obtained) == set(range(len(values)))
    # If b-values, should be ordered by increasing value; if uptake values,
    # should be ordered by decreasing uptake
    sorted_vals = [values[i] for i in obtained]
    reverse = True if feature == "uptake" else False
    assert sorted_vals == sorted(values, reverse=reverse)


from nifreeze.utils.iterators import _filter_indices


def test_filter_indices_basic():
    """Test basic index filtering."""
    indices = range(10)
    assert list(_filter_indices(indices, start_index=3, size=4)) == [3, 4, 5, 6]
    assert list(_filter_indices(indices, start_index=5)) == [5, 6, 7, 8, 9]
    assert list(_filter_indices(indices, start_index=0, size=3)) == [0, 1, 2]


def test_filter_indices_unordered():
    """Test filtering with unordered indices."""
    indices = [2, 5, 1, 8, 3, 7, 0, 9]
    assert list(_filter_indices(indices, start_index=2, size=4)) == [2, 5, 8, 3]
    assert list(_filter_indices(indices, start_index=5)) == [5, 8, 7, 9]


def test_linear_iterator_with_start_index():
    """Test linear iterator with start_index."""
    assert list(linear_iterator(size=10, start_index=3)) == [3, 4, 5, 6, 7, 8, 9]
    assert list(linear_iterator(size=10, start_index=0)) == list(range(10))


def test_linear_iterator_with_size():
    """Test linear iterator with start_index and iter_size."""
    assert list(linear_iterator(size=10, start_index=3, iter_size=4)) == [3, 4, 5, 6]
    assert list(linear_iterator(size=10, start_index=5, iter_size=2)) == [5, 6]


def test_random_iterator_with_start_index():
    """Test random iterator with start_index."""
    result = list(random_iterator(size=15, seed=0, start_index=5))
    # All results should be >= 5
    assert all(idx >= 5 for idx in result)
    # Should have 10 elements (15 - 5)
    assert len(result) == 10


def test_random_iterator_with_size():
    """Test random iterator with start_index and iter_size."""
    result = list(random_iterator(size=15, seed=0, start_index=5, iter_size=3))
    # All results should be >= 5
    assert all(idx >= 5 for idx in result)
    # Should have exactly 3 elements
    assert len(result) == 3


def test_centralsym_iterator_with_start_index():
    """Test centralsym iterator with start_index."""
    result = list(centralsym_iterator(size=10, start_index=3))
    assert all(idx >= 3 for idx in result)
    expected_length = 7  # indices 3-9
    assert len(result) == expected_length


def test_centralsym_iterator_with_size():
    """Test centralsym iterator with start_index and iter_size."""
    result = list(centralsym_iterator(size=11, start_index=3, iter_size=5))
    assert all(idx >= 3 for idx in result)
    assert len(result) == 5


def test_monotonic_value_iterator_with_start_index():
    """Test monotonic_value iterator with start_index."""
    bvals = [0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0]
    result = list(monotonic_value_iterator(bvals=bvals, start_index=4))
    assert all(idx >= 4 for idx in result)
    # Original full order: [0, 1, 8, 4, 5, 2, 3, 6, 7]
    # Filtered (>= 4): [8, 4, 5, 2, 3, 6, 7] -> [4, 5, 6, 7, 8]? No, order preserved
    expected = [8, 4, 5, 6, 7]  # indices >= 4 from original monotonic order
    assert result == expected


def test_monotonic_value_iterator_with_size():
    """Test monotonic_value iterator with start_index and iter_size."""
    uptake = [-1.23, 1.06, 1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05]
    result = list(monotonic_value_iterator(uptake=uptake, start_index=2, iter_size=4))
    assert all(idx >= 2 for idx in result)
    assert len(result) == 4


@pytest.mark.parametrize("start_index,iter_size,expected", [
    (0, None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (3, None, [3, 4, 5, 6, 7, 8, 9]),
    (0, 5, [0, 1, 2, 3, 4]),
    (3, 4, [3, 4, 5, 6]),
    (7, 2, [7, 8]),
])
def test_linear_iterator_parametrize(start_index, iter_size, expected):
    """Parametrized tests for linear iterator."""
    result = list(linear_iterator(size=10, start_index=start_index, iter_size=iter_size))
    assert result == expected
