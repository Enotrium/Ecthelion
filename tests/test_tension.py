"""Tests for hap.tension — Tension Minimization (paper Section: Distributional Semantics)."""

import torch
from hap.tension import (
    tension_energy,
    minimize_tension,
    build_cooc_graph,
    build_masses,
    learn_distributional_hvs,
)
from hap.hdc_core import gen_hvs, hv_hamming_sim


class TestTensionEnergy:
    def test_zero_for_unconnected(self):
        hv = gen_hvs(3, 100, seed=42)
        graph = torch.zeros(3, 3)
        masses = torch.tensor([1.0, 1.0, 1.0])
        energy = tension_energy(hv, graph, masses)
        assert energy == 0.0

    def test_positive_for_connected(self):
        hv = gen_hvs(3, 100, seed=42)
        graph = torch.tensor([
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 0.],
        ])
        masses = torch.tensor([1.0, 1.0, 1.0])
        energy = tension_energy(hv, graph, masses)
        # Random vectors with co-occurrence force should have tension
        assert energy > 0

    def test_lower_for_similar_connected(self):
        dim = 200
        hv = gen_hvs(2, dim, seed=42)
        # Make them nearly identical
        hv[1] = hv[0].clone()
        graph = torch.tensor([[0., 1.], [1., 0.]])
        masses = torch.tensor([1.0, 1.0])
        energy_identical = tension_energy(hv, graph, masses)

        # Make them independent
        hv[1] = gen_hvs(1, dim, seed=99).squeeze(0)
        energy_different = tension_energy(hv, graph, masses)

        assert energy_identical < energy_different


class TestMinimizeTension:
    def test_energy_decreases(self):
        hv = gen_hvs(4, 100, seed=42)
        graph = torch.tensor([
            [0., 1., 0., 0.],
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
            [0., 0., 1., 0.],
        ])
        masses = torch.tensor([1.0, 1.0, 1.0, 1.0])

        initial = tension_energy(hv, graph, masses)
        optimized, history = minimize_tension(
            hv, graph, masses, n_iters=500, initial_temp=5.0, seed=42,
        )
        final = tension_energy(optimized, graph, masses)

        # Energy should decrease (or stay at 0)
        assert final <= initial + 0.01

    def test_converges_to_zero_with_conn_only(self):
        # With only connective force (proximal disabled), energy should
        # approach 0 for a simple chain.
        hv = gen_hvs(3, 50, seed=42)
        graph = torch.tensor([
            [0., 1., 0.],
            [1., 0., 1.],
            [0., 1., 0.],
        ])
        masses = torch.tensor([1.0, 1.0, 1.0])

        optimized, history = minimize_tension(
            hv, graph, masses, n_iters=2000, initial_temp=10.0,
            cooling_rate=0.99, disable_proximal=True, seed=42,
        )
        final_energy = history[-1]
        # Should be very low (approaching 0)
        assert final_energy < 1.0

    def test_returns_energy_history(self):
        hv = gen_hvs(3, 50, seed=42)
        graph = torch.ones(3, 3) * 0.5
        graph.fill_diagonal_(0)
        masses = torch.tensor([1.0, 1.0, 1.0])

        _, history = minimize_tension(
            hv, graph, masses, n_iters=100, seed=42,
        )
        assert len(history) >= 1
        assert all(isinstance(e, float) for e in history)


class TestBuildCoocGraph:
    def test_symmetric(self):
        cooc = {(0, 1): 5.0, (0, 2): 3.0}
        graph = build_cooc_graph(cooc, 3)
        assert graph[0, 1] == graph[1, 0]
        assert graph[0, 2] == graph[2, 0]

    def test_normalized(self):
        cooc = {(0, 1): 10.0}
        graph = build_cooc_graph(cooc, 3)
        assert graph.max() == 1.0


class TestBuildMasses:
    def test_normalized(self):
        counts = torch.tensor([10, 5, 1])
        masses = build_masses(counts)
        assert masses[0] == 1.0
        assert 0 < masses[1] < 1.0
        assert masses[2] < masses[1]

    def test_all_zero(self):
        counts = torch.zeros(5)
        masses = build_masses(counts)
        assert torch.allclose(masses, torch.tensor([0.2] * 5), atol=0.01)


class TestLearnDistributionalHVs:
    def test_basic_pipeline(self):
        # Simple co-occurrence: vertex 0 pairs with 1, vertex 2 pairs with 3
        cooc = {(0, 1): 10.0, (2, 3): 10.0}
        counts = torch.tensor([10, 10, 5, 5])
        hvs = learn_distributional_hvs(
            cooc, counts, dim=500, n_iters=1000, seed=42,
        )
        assert hvs.shape == (4, 500)

        # Co-occurring vertices (0,1) should be closer than non-co-occurring (0,2)
        sim_01 = hv_hamming_sim(hvs[0], hvs[1]).item()
        sim_02 = hv_hamming_sim(hvs[0], hvs[2]).item()
        assert sim_01 > sim_02

    def test_single_vertex(self):
        cooc = {}
        counts = torch.tensor([1])
        hvs = learn_distributional_hvs(cooc, counts, dim=100, n_iters=10, seed=42)
        assert hvs.shape == (1, 100)