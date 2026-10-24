import math

import torch

from echokv_vllm.canary import KRefRing, canary_kl
from echokv_vllm.controller import ControllerParams, LayerController


def test_kref_ring_keeps_last_window():
    ring = KRefRing(4)
    assert ring.view() is None
    ring.push(torch.arange(3).float().view(3, 1, 1))
    assert ring.view().shape[0] == 3
    ring.push(torch.arange(3, 9).float().view(6, 1, 1))
    assert ring.view().flatten().tolist() == [5.0, 6.0, 7.0, 8.0]
    ring.push(torch.tensor([9.0]).view(1, 1, 1))
    assert ring.view().flatten().tolist() == [6.0, 7.0, 8.0, 9.0]  # oldest first


def test_canary_kl_is_monotone_in_bits():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(4, 8, 64, generator=g) * 3
    kref = torch.randn(64, 2, 64, generator=g) * 3
    d2, d4, d8 = canary_kl(q, kref, (2, 4, 8), 32, 64**-0.5).tolist()
    assert d2 > d4 > d8 >= 0.0


def test_canary_matches_cache_k_format():
    """δ̂ at b bits is exactly the KL induced by the cache's stored K for those keys."""
    from echokv_vllm.quant import fake_quant_channels

    g = torch.Generator().manual_seed(1)
    q = torch.randn(2, 4, 64, generator=g)
    kref = torch.randn(40, 2, 64, generator=g)  # 40 keys: one whole group of 32 scored
    (d4,) = canary_kl(q, kref, (4,), 32, 0.125).tolist()
    k = kref[8:]
    kq = fake_quant_channels(k, 4, 32)
    qh = q.transpose(0, 1)
    lr = torch.log_softmax(qh @ k.transpose(0, 1).repeat_interleave(2, 0).transpose(1, 2) * 0.125, -1)
    lq = torch.log_softmax(qh @ kq.transpose(0, 1).repeat_interleave(2, 0).transpose(1, 2) * 0.125, -1)
    assert abs(d4 - (lq.exp() * (lq - lr)).sum(-1).mean().item()) < 1e-6


def make(**kw):
    return LayerController(ControllerParams(**kw))


def test_controller_picks_lowest_tier_under_budget():
    c = make(tau0=0.01, l0=4096, alpha=1.0, init_bits=8)
    c.observe({2: 0.5, 4: 0.005, 8: 0.0001}, ctx_len=4096)
    assert c.bits == 4
    c.observe({2: 0.001, 4: 0.0005, 8: 0.0001}, ctx_len=4096)
    assert c.bits == 2
    c.observe({2: 0.5, 4: 0.5, 8: 0.5}, ctx_len=4096)
    assert c.bits == 8  # nothing fits: highest tier
    assert c.transitions == 3


def test_tau_sqrt_scaling_only_loosens():
    c = make(tau0=0.022, l0=4096, tau_scaling="sqrt")
    assert math.isclose(c.tau(4096 * 32), 0.022 * math.sqrt(32))
    assert c.tau(100) == 0.022  # short context: never tighter than tau0
    assert make(tau0=0.022).tau(4096 * 32) == 0.022  # default: no scaling
    c = make(tau0=0.01, l0=4096, alpha=1.0, tau_scaling="sqrt")
    c.observe({2: 0.03, 4: 0.001, 8: 0.0}, ctx_len=4096)
    assert c.bits == 4
    c.observe({2: 0.03, 4: 0.001, 8: 0.0}, ctx_len=4096 * 16)  # τ = 0.04
    assert c.bits == 2


def test_frozen_and_fixed_hold_tier_but_track_ema():
    c = make(alpha=0.5, init_bits=8, frozen=True)
    c.observe({2: 0.0, 4: 0.0, 8: 0.0}, ctx_len=100)
    c.observe({2: 1.0, 4: 1.0, 8: 1.0}, ctx_len=100)
    assert c.bits == 8 and c.ema[2] == 0.5
    c = make(fixed_bits=2)
    c.observe({2: 9.0, 4: 9.0, 8: 9.0}, ctx_len=100)
    assert c.bits == 2
