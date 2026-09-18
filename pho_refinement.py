import torch
from pho_refinement_tools import PhonemeDenoiseDataset, collate_fn, train_denoiser, PAD_ID, id_to_phoneme,phoneme_to_id

def denoise_phoneme_sequence(model, phoneme_str: str, device=None):
    if device is None:
        device = next(model.parameters()).device

    tokens = phoneme_str.strip().split()
    ids = torch.tensor(
        [phoneme_to_id[t] for t in tokens if t in phoneme_to_id],
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    attention_mask = (ids != PAD_ID).long()

    model.eval()
    with torch.no_grad():
        logits, _ = model(ids, attention_mask=attention_mask)
        pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()

    pred_tokens = [id_to_phoneme[i] for i in pred_ids if i != PAD_ID]
    return " ".join(pred_tokens)


# ===== 7. Entry point example =====

if __name__ == "__main__":
    # if you have a confusion based replacement table, pass it here
    # otherwise leave replacement_table=None for simple noise
    model = train_denoiser(
        jsonl_path="./data/phoneme_sb.jsonl",
        replacement_table=None,
        batch_size=32,
        num_epochs=50,
        lr=2e-4,
    )
    torch.save(model.state_dict(), "model_saving/phoneme_denoise_transformer.pt")

    # quick test with a noisy example
    test_seq = "K AH M P Y UW T ER | IH Z |"
    print("Noisy input: ", test_seq)
    print("Denoised:   ", denoise_phoneme_sequence(model, test_seq))

