import numpy as np
import torch
from pyctcdecode import build_ctcdecoder

def ctc_beam_decode_with_lm(
    logits: torch.Tensor,          # shape [T, C] or [B, T, C]
    labels: list,                  # length C, example ["<blk>", "AA", "AE", ...]
    blank_id: int,
    arpa_path: str,
    alpha: float = 0.6,            # LM weight
    beta: float = 1.0,             # word insertion bonus
    beam_width: int = 50,
):
    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=arpa_path,
        alpha=alpha,
        beta=beta,
    )

    if logits.dim() == 2:
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        text = decoder.decode(log_probs, beam_width=beam_width)
        return text

    if logits.dim() == 3:
        outs = []
        for b in range(logits.size(0)):
            log_probs = torch.log_softmax(logits[b], dim=-1).cpu().numpy()
            outs.append(decoder.decode(log_probs, beam_width=beam_width))
        return outs

    raise ValueError("logits must be [T, C] or [B, T, C]")