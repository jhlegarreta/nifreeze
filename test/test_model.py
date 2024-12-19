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
"""Unit tests exercising models."""

import numpy as np
import pytest
from dipy.sims.voxel import single_tensor

from eddymotion import model
from eddymotion.data.dmri import DWI
from eddymotion.data.splitting import lovo_split
from eddymotion.exceptions import ModelNotFittedError
from eddymotion.model._dipy import GaussianProcessModel
from eddymotion.model.dmri import DEFAULT_MAX_S0, DEFAULT_MIN_S0
from eddymotion.testing import simulations as _sim


def test_trivial_model():
    """Check the implementation of the trivial B0 model."""

    rng = np.random.default_rng(1234)

    # Should not allow initialization without an oracle
    with pytest.raises(TypeError):
        model.TrivialModel()

    _S0 = rng.normal(size=(2, 2, 2))

    _clipped_S0 = np.clip(
        _S0.astype("float32") / _S0.max(),
        a_min=DEFAULT_MIN_S0,
        a_max=DEFAULT_MAX_S0,
    )

    tmodel = model.TrivialModel(predicted=_clipped_S0)

    data = None
    assert tmodel.fit(data) is None

    assert np.all(_clipped_S0 == tmodel.predict((1, 0, 0)))


def test_average_model():
    """Check the implementation of the average DW model."""

    data = np.ones((100, 100, 100, 6), dtype=float)

    gtab = np.array(
        [
            [0, 0, 0, 0],
            [-0.31, 0.933, 0.785, 25],
            [0.25, 0.565, 0.21, 500],
            [-0.861, -0.464, 0.564, 1000],
            [0.307, -0.766, 0.677, 1000],
            [0.736, 0.013, 0.774, 1300],
        ]
    )

    data *= gtab[:, -1]

    tmodel_mean = model.AverageDWModel(gtab=gtab, bias=False, stat="mean")
    tmodel_median = model.AverageDWModel(gtab=gtab, bias=False, stat="median")
    tmodel_1000 = model.AverageDWModel(gtab=gtab, bias=False, th_high=1000, th_low=900)
    tmodel_2000 = model.AverageDWModel(
        gtab=gtab,
        bias=False,
        th_high=2000,
        th_low=900,
        stat="mean",
    )

    with pytest.raises(ModelNotFittedError):
        tmodel_mean.predict([0, 0, 0])

    # Verify that fit function returns nothing
    assert tmodel_mean.fit(data[..., 1:], gtab=gtab[1:].T) is None

    tmodel_median.fit(data[..., 1:], gtab=gtab[1:].T)
    tmodel_1000.fit(data[..., 1:], gtab=gtab[1:].T)
    tmodel_2000.fit(data[..., 1:], gtab=gtab[1:].T)

    # Verify that the right statistics is applied and that the model discard b-values < 50
    assert np.all(tmodel_mean.predict([0, 0, 0]) == 950)
    assert np.all(tmodel_median.predict([0, 0, 0]) == 1000)

    # Verify that the threshold for b-value selection works as expected
    assert np.all(tmodel_1000.predict([0, 0, 0]) == 1000)
    assert np.all(tmodel_2000.predict([0, 0, 0]) == 1100)


@pytest.mark.parametrize(
    (
        "bval_shell",
        "S0",
        "evals",
    ),
    [
        (
            1000,
            100,
            (0.0015, 0.0003, 0.0003),
        )
    ],
)
@pytest.mark.parametrize("snr", (10, 20))
@pytest.mark.parametrize("hsph_dirs", (60, 30))
def test_gp_model(evals, S0, snr, hsph_dirs, bval_shell):
    # Simulate signal for a single tensor
    evecs = _sim.create_single_fiber_evecs()
    gtab = _sim.create_single_shell_gradient_table(hsph_dirs, bval_shell)
    signal = single_tensor(gtab, S0=S0, evals=evals, evecs=evecs, snr=snr)

    # Drop the initial b=0
    gtab = gtab[1:]
    data = signal[1:]

    gp = GaussianProcessModel(kernel_model="spherical")
    assert isinstance(gp, model._dipy.GaussianProcessModel)

    gpfit = gp.fit(data[:-2], gtab[:-2])
    prediction = gpfit.predict(gtab.bvecs[-2:])

    assert prediction.shape == (2,)


def test_two_initialisations(datadir):
    """Check that the two different initialisations result in the same models"""

    # Load test data
    dmri_dataset = DWI.from_filename(datadir / "dwi.h5")

    # Split data into test and train set
    data_train, data_test = lovo_split(dmri_dataset, 10)

    # Direct initialisation
    model1 = model.AverageDWModel(
        gtab=data_train[1],
        S0=dmri_dataset.bzero,
        th_low=100,
        th_high=1000,
        bias=False,
        stat="mean",
    )
    model1.fit(data_train[0], gtab=data_train[1])
    predicted1 = model1.predict(data_test[1])

    # Initialisation via ModelFactory
    model2 = model.ModelFactory.init(
        gtab=data_train[1],
        model="avgdwi",
        S0=dmri_dataset.bzero,
        th_low=100,
        th_high=1000,
        bias=False,
        stat="mean",
    )

    with pytest.raises(ModelNotFittedError):
        model2.predict(data_test[1])

    model2.fit(data_train[0], gtab=data_train[1])
    predicted2 = model2.predict(data_test[1])

    assert np.all(predicted1 == predicted2)
