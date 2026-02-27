"""Quick verification: RiboformerPyTorch shape + param count (no weights needed)."""

import sys
import torch
sys.path.insert(0, "/Users/gongchen/Code/ProteinQC")

from proteinqc.tools.riboformer import RiboformerPyTorch, RiboformerConfig


def test_shapes_and_params() -> None:
    cfg   = RiboformerConfig()
    model = RiboformerPyTorch(cfg)
    model.train(False)

    B = 4
    seq = torch.randint(0, cfg.vocab_size, (B, cfg.wsize))
    exp = torch.rand(B, cfg.wsize)

    with torch.no_grad():
        out, _ = model(seq, exp)

    assert out.shape == (B, 1), f"bad output shape: {out.shape}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape : {out.shape}  OK")
    print(f"Total params : {n_params:,}")
    assert 100_000 < n_params < 500_000, f"param count out of expected range: {n_params}"
    print("All checks passed")


if __name__ == "__main__":
    test_shapes_and_params()
