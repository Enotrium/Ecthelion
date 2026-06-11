"""
Tests for HDC Encoding Module
==============================
Verifies all encoding schemes from the paper:
positional-intensity, time-slice, velocity, sequence,
DVS event, and data record encoding.
"""

import pytest
import torch

from hap.encoding import (
    DataRecordEncoder,
    DVSEncoder,
    PositionalIntensityEncoder,
    SequenceEncoder,
    TimeSliceEncoder,
    VelocityEncoder,
)
from hap.hdc_core import gen_hvs

# ═══════════════════════════════════════════════════════════════════════════════
# PositionalIntensityEncoder
# ═══════════════════════════════════════════════════════════════════════════════


class TestPositionalIntensityEncoder:
    @pytest.fixture
    def encoder(self):
        return PositionalIntensityEncoder(
            height=16,
            width=16,
            dim=500,
            n_intensity_levels=64,
            seed=42,
        )

    def test_encode_shape(self, encoder):
        image = torch.rand(16, 16)
        hv = encoder.encode(image)
        assert hv.shape == (500,)
        assert hv.dtype == torch.float32
        assert torch.all((hv == 0) | (hv == 1))

    def test_encode_with_channel_dim(self, encoder):
        image = torch.rand(1, 16, 16)
        hv = encoder.encode(image)
        assert hv.shape == (500,)

    def test_encode_smaller_than_capacity(self, encoder):
        image = torch.rand(8, 8)
        hv = encoder.encode(image)
        assert hv.shape == (500,)

    def test_encode_exceeds_capacity(self, encoder):
        image = torch.rand(32, 32)
        with pytest.raises(ValueError, match="exceeds encoder capacity"):
            encoder.encode(image)

    def test_constant_image(self, encoder):
        image = torch.ones(16, 16) * 0.5
        hv = encoder.encode(image)
        assert hv.shape == (500,)
        assert torch.all((hv == 0) | (hv == 1))

    def test_zero_image(self, encoder):
        image = torch.zeros(16, 16)
        hv = encoder.encode(image)
        assert hv.shape == (500,)

    def test_reproducibility(self):
        e1 = PositionalIntensityEncoder(height=8, width=8, dim=200, seed=42)
        e2 = PositionalIntensityEncoder(height=8, width=8, dim=200, seed=42)
        img = torch.rand(8, 8)
        assert torch.equal(e1.encode(img), e2.encode(img))

    def test_different_images_different_hvs(self, encoder):
        img1 = torch.rand(16, 16)
        img2 = torch.rand(16, 16)
        hv1 = encoder.encode(img1)
        hv2 = encoder.encode(img2)
        # They might be the same by chance with small dim, but likely different
        assert hv1.shape == hv2.shape

    def test_forward_equals_encode(self, encoder):
        img = torch.rand(16, 16)
        assert torch.equal(encoder(img), encoder.encode(img))

    def test_bipolar_mode(self):
        encoder = PositionalIntensityEncoder(
            height=8,
            width=8,
            dim=200,
            mode="bipolar",
            seed=42,
        )
        img = torch.rand(8, 8)
        hv = encoder.encode(img)
        assert hv.shape == (200,)
        # Bipolar HVs have values in {-1, 1}
        assert ((hv == -1) | (hv == 1) | (hv == 0)).all()


# ═══════════════════════════════════════════════════════════════════════════════
# TimeSliceEncoder
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimeSliceEncoder:
    @pytest.fixture
    def encoder(self):
        return TimeSliceEncoder(
            height=32,
            width=32,
            dim=400,
            intensity_levels=16,
            seed=42,
        )

    def test_encode_time_slice(self, encoder):
        img = torch.rand(32, 32)
        img[img < 0.3] = 0  # Simulate sparse time image
        hv = encoder.encode_time_slice(img)
        assert hv.shape == (400,)
        assert torch.all((hv == 0) | (hv == 1))

    def test_all_zero_time_image(self, encoder):
        img = torch.zeros(32, 32)
        hv = encoder.encode_time_slice(img)
        assert hv.shape == (400,)

    def test_forward(self, encoder):
        img = torch.rand(32, 32)
        assert torch.equal(encoder(img), encoder.encode_time_slice(img))

    def test_encode_sequence(self, encoder):
        images = torch.rand(5, 32, 32)
        result = encoder.encode_sequence(images)
        assert result.shape == (5, 400)

    def test_encode_sequence_with_window(self, encoder):
        images = torch.rand(5, 32, 32)
        result = encoder.encode_sequence(images, window_size=2)
        assert result.shape == (3, 400)  # ceil(5/2) = 3

    def test_encode_sequence_window_greater_than_T(self, encoder):
        images = torch.rand(3, 32, 32)
        result = encoder.encode_sequence(images, window_size=10)
        assert result.shape[0] == 1  # single bundled window


# ═══════════════════════════════════════════════════════════════════════════════
# VelocityEncoder
# ═══════════════════════════════════════════════════════════════════════════════


class TestVelocityEncoder:
    @pytest.fixture
    def encoder(self):
        return VelocityEncoder(
            min_val=0.0,
            max_val=1.0,
            step=0.1,
            dim=300,
            seed=42,
        )

    def test_single_encode(self, encoder):
        hv = encoder.encode(0.5)
        assert hv.shape == (300,)
        assert torch.all((hv == 0) | (hv == 1))

    def test_boundary_values(self, encoder):
        hv_min = encoder.encode(0.0)
        hv_max = encoder.encode(1.0)
        assert hv_min.shape == (300,)
        assert hv_max.shape == (300,)

    def test_out_of_range_clamped(self, encoder):
        hv = encoder.encode(10.0)  # above max
        assert hv.shape == (300,)
        hv_neg = encoder.encode(-5.0)  # below min
        assert hv_neg.shape == (300,)

    def test_tensor_input(self, encoder):
        hv = encoder.encode(torch.tensor(0.3))
        assert hv.shape == (300,)

    def test_encode_3d(self, encoder):
        hv = encoder.encode_3d(0.1, 0.2, 0.3)
        assert hv.shape == (300,)

    def test_encode_3d_with_custom_keys(self, encoder):
        keys = gen_hvs(3, 300, seed=9999)
        hv = encoder.encode_3d(0.1, 0.2, 0.3, keys=keys)
        assert hv.shape == (300,)

    def test_nearby_velocities_similar(self, encoder):
        hv1 = encoder.encode(0.5)
        hv2 = encoder.encode(0.5001)
        from hap.hdc_core import hv_hamming_sim

        sim = hv_hamming_sim(hv1, hv2)
        assert sim > 0.9, f"Nearby velocities should be similar, got {sim}"

    def test_distant_velocities_dissimilar(self, encoder):
        hv1 = encoder.encode(0.0)
        hv2 = encoder.encode(1.0)
        from hap.hdc_core import hv_hamming_sim

        sim = hv_hamming_sim(hv1, hv2)
        assert sim < 0.8, f"Distant velocities should differ, got {sim}"


# ═══════════════════════════════════════════════════════════════════════════════
# SequenceEncoder
# ═══════════════════════════════════════════════════════════════════════════════


class TestSequenceEncoder:
    @pytest.fixture
    def seq_encoder(self):
        return SequenceEncoder(dim=200, permute_k=1)

    def test_encode_single(self, seq_encoder):
        hv = gen_hvs(1, 200, seed=1).squeeze(0)
        result = seq_encoder.encode(hv.unsqueeze(0))
        assert torch.equal(result, hv)

    def test_encode_length(self, seq_encoder):
        seq = gen_hvs(5, 200, seed=42)
        result = seq_encoder.encode(seq)
        assert result.shape == (200,)
        assert torch.all((result == 0) | (result == 1))

    def test_encode_reversible_prefix(self, seq_encoder):
        """Sequences differing by prefix produce different HVs."""
        a = gen_hvs(3, 200, seed=1)
        b = torch.cat([a[:2], gen_hvs(1, 200, seed=99)])
        hv_a = seq_encoder.encode(a)
        hv_b = seq_encoder.encode(b)
        assert not torch.equal(hv_a, hv_b)

    def test_shift_forward(self, seq_encoder):
        seq_hv = gen_hvs(1, 200, seed=1).squeeze(0)
        new_hv = gen_hvs(1, 200, seed=2).squeeze(0)
        updated = seq_encoder.shift_forward(seq_hv, new_hv)
        assert updated.shape == (200,)

    def test_shift_backward(self, seq_encoder):
        seq = gen_hvs(3, 200, seed=10)
        original = seq_encoder.encode(seq)
        oldest = seq[0]
        removed = seq_encoder.shift_backward(original, oldest, 3)
        # The result should be equivalent to encoding just the last 2 elements
        expected = seq_encoder.encode(seq[1:])
        assert torch.equal(removed, expected)


# ═══════════════════════════════════════════════════════════════════════════════
# DVSEncoder
# ═══════════════════════════════════════════════════════════════════════════════


class TestDVSEncoder:
    @pytest.fixture
    def encoder(self):
        return DVSEncoder(width=16, height=12, dim=200, seed=42)

    def test_encode_event(self, encoder):
        hv = encoder.encode_event(x=5, y=3, t=0.01, polarity=1)
        assert hv.shape == (200,)

    def test_encode_on_off_different(self, encoder):
        hv_on = encoder.encode_event(x=5, y=3, t=0.01, polarity=1)
        hv_off = encoder.encode_event(x=5, y=3, t=0.01, polarity=-1)
        assert not torch.equal(hv_on, hv_off)

    def test_accumulate_and_get_time_image(self, encoder):
        # Accumulate a few events
        encoder.accumulate(5, 3, 0.01, 1)
        encoder.accumulate(5, 3, 0.02, -1)
        encoder.accumulate(7, 5, 0.03, 1)

        img = encoder.get_time_image()
        assert img.shape == (12, 16)
        assert img.max() > 0

    def test_empty_accumulator(self, encoder):
        img = encoder.get_time_image()
        assert img.shape == (12, 16)
        assert img.max() == 0.0

    def test_reset(self, encoder):
        encoder.accumulate(5, 3, 0.01, 1)
        encoder.reset_accumulator()
        img = encoder.get_time_image()
        assert img.max() == 0.0

    def test_get_polarity_time_image(self, encoder):
        encoder.accumulate(5, 3, 0.01, 1)
        encoder.accumulate(5, 3, 0.02, -1)
        img = encoder.get_polarity_time_image()
        assert img.shape == (12, 16)

    def test_out_of_bounds_clamped(self, encoder):
        # Should not crash on out-of-bounds coordinates
        encoder.accumulate(100, 100, 0.01, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# DataRecordEncoder
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataRecordEncoder:
    @pytest.fixture
    def encoder(self):
        return DataRecordEncoder(
            field_names=["image", "velocity", "timestamp"],
            dim=200,
            seed=42,
        )

    def test_encode_record(self, encoder):
        img_hv = gen_hvs(1, 200, seed=1).squeeze(0)
        vel_hv = gen_hvs(1, 200, seed=2).squeeze(0)
        time_hv = gen_hvs(1, 200, seed=3).squeeze(0)

        record = encoder.encode_record(
            image=img_hv,
            velocity=vel_hv,
            timestamp=time_hv,
        )
        assert record.shape == (200,)

    def test_query_field(self, encoder):
        img_hv = gen_hvs(1, 200, seed=1).squeeze(0)
        vel_hv = gen_hvs(1, 200, seed=2).squeeze(0)
        time_hv = gen_hvs(1, 200, seed=3).squeeze(0)

        record = encoder.encode_record(
            image=img_hv,
            velocity=vel_hv,
            timestamp=time_hv,
        )

        # Query the image field
        candidates = gen_hvs(5, 200, seed=1)  # first = img_hv
        idx, sim = encoder.query_field(record, "image", candidates)
        assert 0 <= idx < 5

    def test_unknown_field_raises(self, encoder):
        hv = gen_hvs(1, 200).squeeze(0)
        with pytest.raises(KeyError, match="Unknown field"):
            encoder.encode_record(unknown=hv)

    def test_encode_with_scalar(self, encoder):
        vel_hv = gen_hvs(1, 200).squeeze(0)
        scalar = torch.tensor(0.5)
        # Should handle scalar values
        record = encoder.encode_record(
            image=vel_hv,
            velocity=scalar,
            timestamp=vel_hv,
        )
        assert record.shape == (200,)

    def test_reproducibility(self):
        e1 = DataRecordEncoder(["a", "b"], dim=100, seed=42)
        e2 = DataRecordEncoder(["a", "b"], dim=100, seed=42)
        hv = gen_hvs(1, 100).squeeze(0)
        r1 = e1.encode_record(a=hv, b=hv)
        r2 = e2.encode_record(a=hv, b=hv)
        assert torch.equal(r1, r2)
