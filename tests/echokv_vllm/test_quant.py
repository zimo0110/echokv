import pytest
import torch

from echokv_vllm.quant import dequantize, fake_quant, fake_quant_channels, pack, quantize, unpack


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_unpack_roundtrip(bits):
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 1 << bits, (5, 3, 64), generator=g, dtype=torch.int64).to(torch.uint8)
    packed = pack(codes, bits)
    assert packed.shape == (5, 3, 64 * bits // 8)
    assert torch.equal(unpack(packed, bits), codes)


def _max_err_in_steps(x, bits, gs):
    codes, scale, mn = quantize(x, bits, gs)
    err = (dequantize(codes, scale, mn, gs) - x).abs().reshape(*x.shape[:-1], -1, gs)
    return (err / scale.float().unsqueeze(-1)).max().item()


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("shift", [0.0, 40.0, -40.0])
def test_error_within_half_step_for_any_sign(bits, shift):
    # shift=±40 makes every group one-signed: the case an integer zero-point
    # clamped to [0, 2**b - 1] gets wrong by the magnitude of the values.
    x = torch.randn(7, 4, 128, generator=torch.Generator().manual_seed(1)) + shift
    # Half a step, plus fp16 rounding of min/scale (relative 2**-11 of |x| / scale).
    slack = 2**-11 * (abs(shift) + 4) / (8 / ((1 << bits) - 1))
    assert _max_err_in_steps(x, bits, 32) <= 0.5 + slack + 1e-3


def test_constant_group_is_exact_to_fp16():
    for val in (-3.5, 0.0, 7.25):
        x = torch.full((2, 1, 64), val)
        for bits in (2, 4, 8):
            torch.testing.assert_close(fake_quant(x, bits, 32), x, rtol=1e-3, atol=0)


def test_channel_quant_groups_tokens_and_keeps_tail_exact():
    k = torch.randn(70, 2, 64, generator=torch.Generator().manual_seed(3))
    k[:, :, 5] += 25.0  # outlier channel
    out = fake_quant_channels(k, 4, 32)
    torch.testing.assert_close(out[64:], k[64:], rtol=0, atol=0)  # partial group untouched
    # Per-channel groups absorb the outlier: error stays relative to each channel's own range.
    span = k[:64].unflatten(0, (2, 32)).amax(1) - k[:64].unflatten(0, (2, 32)).amin(1)
    err = (out[:64] - k[:64]).abs().unflatten(0, (2, 32)).amax(1)
    assert (err <= span / 15 * 0.5 + 1e-2).all()
