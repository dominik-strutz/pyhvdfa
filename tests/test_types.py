"""
tests/test_types.py — Tests for _types.py (Layer, Model, HVResult).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyhvdfa._types import Layer, Model, HVResult


class TestLayer:
    def test_construct(self):
        l = Layer(10.0, 400.0, 200.0, 1800.0)
        assert l.thickness == 10.0
        assert l.vp == 400.0
        assert l.vs == 200.0
        assert l.density == 1800.0


class TestModel:
    def test_from_layers_basic(self, simple_2layer):
        m = simple_2layer
        assert m.n == 2
        assert len(m.alfa) == 2
        assert len(m.h) == 1
        assert m.h[0] == 20.0

    def test_mu_computed(self, simple_2layer):
        m = simple_2layer
        np.testing.assert_allclose(m.mu, m.rho * m.bta**2)

    def test_from_arrays(self):
        m = Model.from_arrays(
            vp=[500.0, 1800.0],
            vs=[200.0,  800.0],
            density=[1800.0, 2200.0],
            thickness=[20.0],
        )
        assert m.n == 2
        assert m.h[0] == 20.0

    def test_too_few_layers_raises(self):
        with pytest.raises(ValueError, match="at least one layer"):
            Model.from_layers([Layer(0.0, 1000.0, 500.0, 2000.0)])

    def test_wrong_h_length_raises(self):
        with pytest.raises(ValueError):
            Model(
                alfa=np.array([500.0, 1800.0]),
                bta =np.array([200.0,  800.0]),
                rho =np.array([1800.0, 2200.0]),
                h   =np.array([20.0, 10.0]),   # should be length 1
            )

    def test_from_file(self, tmp_path):
        model_text = "2\n20.0 500.0 200.0 1800.0\n0.0 1800.0 800.0 2200.0\n"
        p = tmp_path / "model.txt"
        p.write_text(model_text)
        m = Model.from_file(p)
        assert m.n == 2
        np.testing.assert_allclose(m.alfa, [500.0, 1800.0])


class TestHVResult:
    def test_construct(self):
        freq = np.array([1.0, 2.0, 3.0])
        hv   = np.array([1.5, 2.0, 1.8])
        r = HVResult(freq=freq, hv=hv)
        np.testing.assert_array_equal(r.freq, freq)
        np.testing.assert_array_equal(r.hv, hv)
