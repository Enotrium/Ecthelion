"""
Tests for HDC Memory / Learning Module
========================================
Verifies: AssociativeMemory, ActionPerceptionMemory,
DataRecordMemory, HDCClassifier, and RefineHDLearner.
"""

import pytest
import torch

from hap.hdc_core import gen_hvs
from hap.memory import (
    ActionPerceptionMemory,
    AssociativeMemory,
    DataRecordMemory,
    HDCClassifier,
    RefineHDLearner,
)

# ═══════════════════════════════════════════════════════════════════════════════
# AssociativeMemory
# ═══════════════════════════════════════════════════════════════════════════════


class TestAssociativeMemory:
    @pytest.fixture
    def memory(self):
        return AssociativeMemory(dim=500, mode="binary")

    @pytest.fixture
    def percept_action(self):
        p = gen_hvs(1, 500, seed=1).squeeze(0)
        a = gen_hvs(1, 500, seed=10).squeeze(0)
        return p, a

    def test_init(self, memory):
        assert memory.n_samples == 0
        assert memory.memory.sum() == 0

    def test_train_increases_count(self, memory, percept_action):
        p, a = percept_action
        memory.train(p, a)
        assert memory.n_samples == 1
        assert memory.memory.sum() > 0

    def test_train_batch(self, memory):
        percepts = gen_hvs(10, 500, seed=1)
        actions = gen_hvs(10, 500, seed=10)
        memory.train_batch(percepts, actions)
        assert memory.n_samples == 10

    def test_infer_raises_on_empty(self, memory):
        p = gen_hvs(1, 500).squeeze(0)
        candidates = gen_hvs(3, 500)
        with pytest.raises(RuntimeError, match="empty"):
            memory.infer(p, candidates)

    def test_train_infer_roundtrip(self, memory, percept_action):
        """Train on one (percept, action) and verify inference recovers it."""
        p, a = percept_action
        memory.train(p, a)

        # Create candidates including the true action
        candidates = torch.stack(
            [
                gen_hvs(1, 500, seed=99).squeeze(0),
                gen_hvs(1, 500, seed=98).squeeze(0),
                a,
                gen_hvs(1, 500, seed=97).squeeze(0),
            ]
        )

        best_idx, sims = memory.infer(p, candidates)
        # The actual action should have highest similarity
        assert sims[2] > sims[0], f"True action should beat random, got {sims}"

    def test_clear(self, memory, percept_action):
        p, a = percept_action
        memory.train(p, a)
        assert memory.n_samples == 1
        memory.clear()
        assert memory.n_samples == 0
        assert memory.memory.sum() == 0

    def test_save_load(self, memory, percept_action, tmp_path):
        p, a = percept_action
        memory.train(p, a)

        path = str(tmp_path / "memory.pt")
        memory.save(path)

        m2 = AssociativeMemory(dim=500)
        m2.load(path)
        assert m2.n_samples == 1
        assert torch.equal(m2.memory, memory.memory)

    def test_threshold_binary(self, memory, percept_action):
        p, a = percept_action
        memory.train(p, a)
        thresh = memory._threshold_memory()
        assert thresh.shape == (500,)
        assert torch.all((thresh == 0) | (thresh == 1))

    def test_bipolar_mode(self):
        mem = AssociativeMemory(dim=100, mode="bipolar")
        p = gen_hvs(1, 100, "bipolar", seed=1).squeeze(0)
        a = gen_hvs(1, 100, "bipolar", seed=2).squeeze(0)
        mem.train(p, a)
        thresh = mem._threshold_memory()
        assert thresh.shape == (100,)
        assert ((thresh == -1) | (thresh == 1) | (thresh == 0)).all()


# ═══════════════════════════════════════════════════════════════════════════════
# ActionPerceptionMemory
# ═══════════════════════════════════════════════════════════════════════════════


class TestActionPerceptionMemory:
    @pytest.fixture
    def ap_mem(self):
        return ActionPerceptionMemory(n_classes=5, dim=300)

    def test_init(self, ap_mem):
        assert ap_mem.n_classes == 5
        assert ap_mem._class_memories.shape == (5, 300)
        assert ap_mem._class_counts.sum() == 0

    def test_train(self, ap_mem):
        hv = gen_hvs(1, 300, seed=1).squeeze(0)
        ap_mem.train(hv, class_idx=2)
        assert ap_mem._class_counts[2] == 1
        assert ap_mem._class_counts.sum() == 1

    def test_infer_returns_valid_idx(self, ap_mem):
        # Train a few classes
        for i in range(ap_mem.n_classes):
            hv = gen_hvs(1, 300, seed=i * 10).squeeze(0)
            ap_mem.train(hv, class_idx=i)

        query = gen_hvs(1, 300, seed=30).squeeze(0)  # near class 3
        best_idx, sims = ap_mem.infer(query)
        assert 0 <= best_idx < ap_mem.n_classes

    def test_infer_shape(self, ap_mem):
        # Train minimally
        hv = gen_hvs(1, 300, seed=1).squeeze(0)
        ap_mem.train(hv, class_idx=0)

        query = gen_hvs(1, 300, seed=2).squeeze(0)
        best_idx, sims = ap_mem.infer(query)
        assert sims.shape == (5,)
        assert torch.all((sims >= 0) & (sims <= 1))

    def test_clear(self, ap_mem):
        hv = gen_hvs(1, 300).squeeze(0)
        ap_mem.train(hv, class_idx=1)
        ap_mem.clear()
        assert ap_mem._class_counts.sum() == 0

    def test_save_load(self, ap_mem, tmp_path):
        hv = gen_hvs(1, 300, seed=1).squeeze(0)
        ap_mem.train(hv, class_idx=1)

        path = str(tmp_path / "ap_mem.pt")
        ap_mem.save(path)

        m2 = ActionPerceptionMemory(n_classes=5, dim=300)
        m2.load(path)
        assert m2._class_counts[1] == 1
        assert torch.equal(m2._class_memories, ap_mem._class_memories)

    def test_get_velocity_class(self, ap_mem):
        hv = gen_hvs(1, 300, seed=1).squeeze(0)
        ap_mem.train(hv, class_idx=1)

        query = hv.clone()
        v_keys = gen_hvs(5, 300, seed=42)
        idx, prob = ap_mem.get_velocity_class(query, v_keys)
        assert 0 <= idx < 5
        assert 0.0 <= prob <= 1.0

    def test_many_classes(self):
        mem = ActionPerceptionMemory(n_classes=100, dim=200)
        for i in range(100):
            mem.train(gen_hvs(1, 200, seed=i).squeeze(0), class_idx=i)
        assert mem._class_counts.sum() == 100


# ═══════════════════════════════════════════════════════════════════════════════
# DataRecordMemory (Sliding Window)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataRecordMemory:
    @pytest.fixture
    def drm(self):
        return DataRecordMemory(window_size=5, dim=200)

    def test_add_under_capacity(self, drm):
        p = gen_hvs(1, 200, seed=1).squeeze(0)
        a = gen_hvs(1, 200, seed=2).squeeze(0)
        drm.add(p, a)
        assert drm._n_records == 1
        assert not drm.is_full

    def test_add_at_capacity_evicts_oldest(self, drm):
        records = []
        for i in range(7):
            p = gen_hvs(1, 200, seed=i * 2).squeeze(0)
            a = gen_hvs(1, 200, seed=i * 2 + 1).squeeze(0)
            drm.add(p, a)
            records.append((p.clone(), a.clone()))

        assert drm._n_records == 5
        assert drm.is_full
        # Oldest 2 should be evicted
        assert len(drm._records) == 5

    def test_clear(self, drm):
        p = gen_hvs(1, 200).squeeze(0)
        a = gen_hvs(1, 200).squeeze(0)
        drm.add(p, a)
        drm.clear()
        assert drm._n_records == 0
        assert drm.memory.sum() == 0

    def test_infer(self, drm):
        p = gen_hvs(1, 200, seed=1).squeeze(0)
        a = gen_hvs(1, 200, seed=2).squeeze(0)
        drm.add(p, a)

        candidates = gen_hvs(3, 200)
        candidates[0] = a
        idx, sims = drm.infer(p, candidates)
        assert 0 <= idx < 3


# ═══════════════════════════════════════════════════════════════════════════════
# HDCClassifier
# ═══════════════════════════════════════════════════════════════════════════════


class TestHDCClassifier:
    @pytest.fixture
    def clf(self):
        return HDCClassifier(n_classes=4, dim=500, seed=42)

    @pytest.fixture
    def data(self):
        """Generate 4 simple class clusters."""
        n = 20
        class_hvs = gen_hvs(4, 500, seed=100)
        percepts = []
        labels = []
        for c in range(4):
            # Generate percepts biased toward class prototype
            for i in range(n):
                noise = gen_hvs(1, 500, seed=c * 1000 + i).squeeze(0)
                # Mix class prototype with noise
                p = (class_hvs[c] * 0.8 + noise * 0.2).round().clamp(0, 1)
                percepts.append(p)
                labels.append(c)

        return torch.stack(percepts), labels

    def test_init(self, clf):
        assert clf.n_classes == 4
        assert clf.class_hvs.shape == (4, 500)

    def test_fit(self, clf, data):
        percepts, labels = data
        clf.fit(percepts, labels)
        assert clf.memory.n_samples == len(labels)

    def test_predict(self, clf, data):
        percepts, labels = data
        clf.fit(percepts, labels)
        pred = clf.predict(percepts[0])
        assert 0 <= pred < 4

    def test_predict_batch(self, clf, data):
        percepts, labels = data
        clf.fit(percepts, labels)
        preds = clf.predict_batch(percepts)
        assert len(preds) == len(percepts)
        assert all(0 <= p < 4 for p in preds)

    def test_accuracy(self, clf, data):
        percepts, labels = data
        clf.fit(percepts, labels)
        acc = clf.accuracy(percepts, labels)
        assert 0.0 <= acc <= 1.0
        # With biased data, should be above chance (25%)
        assert acc > 0.25, f"Accuracy {acc} should exceed random chance"

    def test_save_load(self, clf, data, tmp_path):
        percepts, labels = data
        clf.fit(percepts, labels)

        path = str(tmp_path / "classifier.pt")
        clf.save(path)

        clf2 = HDCClassifier(n_classes=4, dim=500)
        clf2.load(path)
        assert clf2.n_classes == 4
        pred1 = clf.predict(percepts[0])
        pred2 = clf2.predict(percepts[0])
        assert pred1 == pred2

    def test_clear(self, clf, data):
        percepts, labels = data
        clf.fit(percepts, labels)
        clf.clear()
        assert clf.memory.n_samples == 0


# ═══════════════════════════════════════════════════════════════════════════════
# RefineHDLearner
# ═══════════════════════════════════════════════════════════════════════════════


class TestRefineHDLearner:
    @pytest.fixture
    def data(self):
        n = 15
        class_hvs = gen_hvs(3, 400, seed=100)
        percepts = []
        labels = []
        for c in range(3):
            for i in range(n):
                noise = gen_hvs(1, 400, seed=c * 1000 + i).squeeze(0)
                p = (class_hvs[c] * 0.75 + noise * 0.25).round().clamp(0, 1)
                percepts.append(p)
                labels.append(c)
        return torch.stack(percepts), labels

    def test_refinement_improves(self, data):
        percepts, labels = data

        clf = HDCClassifier(n_classes=3, dim=400, seed=42)
        learner = RefineHDLearner(clf, n_refinement_rounds=3, refinement_weight=2.0)

        result = learner.fit(percepts, labels)

        assert "initial_accuracy" in result
        assert "final_accuracy" in result
        assert len(result["history"]) == 4  # round 0 + 3 refinement rounds
        assert result["final_accuracy"] >= result["initial_accuracy"], (
            "Refinement should not decrease accuracy"
        )

    def test_history_structure(self, data):
        percepts, labels = data
        clf = HDCClassifier(n_classes=3, dim=400, seed=42)
        learner = RefineHDLearner(clf, n_refinement_rounds=1)

        result = learner.fit(percepts, labels)
        history = result["history"]
        assert history[0]["round"] == 0
        assert history[1]["round"] == 1
        assert "accuracy" in history[0]

    def test_zero_rounds(self, data):
        percepts, labels = data
        clf = HDCClassifier(n_classes=3, dim=400, seed=42)
        learner = RefineHDLearner(clf, n_refinement_rounds=0)

        result = learner.fit(percepts, labels)
        assert len(result["history"]) == 1  # only initial round


# ═══════════════════════════════════════════════════════════════════════════════
# Integration / End-to-End
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end tests combining encoding, memory, and inference."""

    def test_encoding_to_classification(self):
        """Encode images → classify via HDCClassifier."""
        from hap.encoding import PositionalIntensityEncoder
        from hap.memory import HDCClassifier

        # Encode simple images
        encoder = PositionalIntensityEncoder(height=8, width=8, dim=300, seed=1)
        clf = HDCClassifier(n_classes=3, dim=300, seed=42)

        percepts = []
        labels = []
        for c in range(3):
            for _ in range(10):
                img = torch.rand(8, 8) * 0.3 + c * 0.2
                hv = encoder.encode(img)
                percepts.append(hv)
                labels.append(c)

        percepts = torch.stack(percepts)
        clf.fit(percepts, labels)
        acc = clf.accuracy(percepts, labels)
        assert acc > 0.3, f"Accuracy {acc} should exceed chance (33%)"

    def test_ego_motion_end_to_end(self):
        """Minimal ego-motion pipeline: encode → train → infer."""
        from hap.hap import EgoMotionEstimator

        est = EgoMotionEstimator(
            width=8,
            height=6,
            dim=300,
            n_angular_bins=10,
            n_linear_x_bins=5,
            n_linear_z_bins=5,
            velocity_step=0.1,
            seed=42,
        )

        # Train on a few samples
        for i in range(20):
            img = torch.rand(6, 8)
            ang = (i % 10) * 0.1
            lx = (i % 5) * 0.1
            lz = (i % 5) * 0.1
            est.train(img, ang, lx, lz)

        # Infer
        result = est.infer(torch.rand(6, 8))
        assert "angular" in result
        assert "linear_x" in result
        assert "linear_z" in result
        for key in ["angular_prob", "linear_x_prob", "linear_z_prob"]:
            assert 0.0 <= result[key] <= 1.0

    def test_HAP_full_pipeline(self):
        """Test the full HyperdimensionalActivePerception pipeline."""
        from hap.encoding import VelocityEncoder
        from hap.hap import HyperdimensionalActivePerception
        from hap.memory import AssociativeMemory

        encoder = VelocityEncoder(
            min_val=0.0,
            max_val=1.0,
            step=0.1,
            dim=300,
            seed=42,
        )
        memory = AssociativeMemory(dim=300, mode="binary")

        hap = HyperdimensionalActivePerception(
            dim=300,
            encoder=encoder,
            memory=memory,
        )

        # Observe, memorize, decide
        vel_hv = hap.observe(torch.tensor(0.3))
        action = gen_hvs(1, 300, seed=1).squeeze(0)

        hap.memorize(vel_hv, action)

        candidates = torch.stack([action, gen_hvs(1, 300, seed=99).squeeze(0)])
        idx, sims = hap.decide(vel_hv, candidates)
        assert idx == 0  # Should recover the bound action

        stats = hap.stats
        assert stats["n_train"] == 1
        assert stats["n_infer"] == 1

    def test_HAP_no_encoder_raises(self):
        from hap.hap import HyperdimensionalActivePerception

        hap = HyperdimensionalActivePerception(dim=100)
        with pytest.raises(ValueError, match="encoder"):
            hap.observe(torch.rand(10))

    def test_deterministic_with_seed(self):
        """Same seed → same results for full pipeline."""
        from hap.hap import EgoMotionEstimator

        e1 = EgoMotionEstimator(
            width=8,
            height=6,
            dim=200,
            n_angular_bins=5,
            n_linear_x_bins=3,
            n_linear_z_bins=3,
            velocity_step=0.1,
            seed=42,
        )

        torch.manual_seed(123)
        for i in range(10):
            img = torch.rand(6, 8)
            e1.train(img, (i % 5) * 0.1, (i % 3) * 0.1, (i % 3) * 0.1)

        e2 = EgoMotionEstimator(
            width=8,
            height=6,
            dim=200,
            n_angular_bins=5,
            n_linear_x_bins=3,
            n_linear_z_bins=3,
            velocity_step=0.1,
            seed=42,
        )

        torch.manual_seed(123)
        for i in range(10):
            img = torch.rand(6, 8)
            e2.train(img, (i % 5) * 0.1, (i % 3) * 0.1, (i % 3) * 0.1)

        # Check memories match
        assert torch.equal(
            e1.angular_mem._class_memories,
            e2.angular_mem._class_memories,
        )

    def test_ego_motion_save_load(self, tmp_path):
        from hap.hap import EgoMotionEstimator

        est = EgoMotionEstimator(
            width=8,
            height=6,
            dim=200,
            n_angular_bins=5,
            n_linear_x_bins=3,
            n_linear_z_bins=3,
            velocity_step=0.1,
            seed=42,
        )

        est.train(torch.rand(6, 8), 0.1, 0.0, 0.0)

        path = str(tmp_path / "estimator.pt")
        est.save(path)

        est2 = EgoMotionEstimator(
            width=8,
            height=6,
            dim=200,
            n_angular_bins=5,
            n_linear_x_bins=3,
            n_linear_z_bins=3,
            velocity_step=0.1,
            seed=42,
        )
        est2.load(path)
        assert est2._total_train_samples == 1
        assert torch.equal(
            est.angular_mem._class_memories,
            est2.angular_mem._class_memories,
        )
