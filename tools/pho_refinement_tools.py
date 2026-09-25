import json
import math
import random
from pathlib import Path

import numpy as np
import torch,sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

import random

LOGIT_TO_PHONEME = [
    'BLANK',
    'AA','AE','AH','AO','AW',
    'AY','B','CH','D','DH',
    'EH','ER','EY','F','G',
    'HH','IH','IY','JH','K',
    'L','M','N','NG','OW',
    'OY','P','R','S','SH',
    'T','TH','UH','UW','V',
    'W','Y','Z','ZH',
    '|', 'SLOT'
]

# special tokens we do not want to corrupt
SPECIAL_TOKENS = {"BLANK", "|"}


phoneme_to_id = {p: i for i, p in enumerate(LOGIT_TO_PHONEME)}
id_to_phoneme = {i: p for p, i in phoneme_to_id.items()}

PAD_TOKEN = "BLANK"
PAD_ID = phoneme_to_id[PAD_TOKEN]     # this is now 0

# phonemes that can be used for substitutions or insertions
NOISE_PHONEMES = [p for p in LOGIT_TO_PHONEME if p not in SPECIAL_TOKENS]

def noise_phoneme_tokens(
    tokens,
    keep_clean_prob: float = 0.1,
    delete_prob: float = 0.06,
    switch_prob: float = 0.10,
    insert_prob: float = 0.02,
    replacement_table=None,
):
    """
    tokens: list of clean phoneme strings, e g ["K","AH","M","P","Y","UW","T","ER","|"]

    Returns a new list of phoneme strings which is a noisy version
    of the input sequence. This will be used as model input, while
    the original tokens are the target.
    """

    # maybe keep the sequence completely clean
    if random.random() < keep_clean_prob:
        modified_idx = list(range(len(tokens)))
        return tokens[:], modified_idx   # return a shallow copy, not the same list

    new_tokens = []
    modified_idx = []
    original_i = 0
    delete_prob = random.randint(4,8)*0.01
    switch_prob = random.randint(4,10)*0.01
    for t in tokens:
        ts = t.strip()
        # do not corrupt special tokens like BLANK or |
        if ts in SPECIAL_TOKENS:
            new_tokens.append(ts)
            original_i += 1
            continue

        r = random.random()
        
        # delete: skip this token
        if r < delete_prob:
            continue

        # substitute (switch) the phoneme

        if r < switch_prob:
            modified_idx.append(original_i)
            original_i += 1
            if replacement_table is not None and ts in replacement_table:
                targets, probs = replacement_table[ts]
                # pick a replacement based on confusion probabilities
                new_t = random.choices(targets, weights=probs, k=1)[0]
            else:
                # fallback: uniform among NOISE_PHONEMES, avoid identity if possible
                if NOISE_PHONEMES:
                    new_t = ts
                    if len(NOISE_PHONEMES) > 1:
                        while new_t == ts:
                            new_t = random.choice(NOISE_PHONEMES)
                    else:
                        new_t = NOISE_PHONEMES[0]
                else:
                    new_t = ts
            new_tokens.append(new_t)
        
        # keep token as is
        else:
            original_i += 1
            new_tokens.append(ts)
    return new_tokens,modified_idx


class PhonemeDenoiseDataset(Dataset):
    def __init__(self, jsonl_path, noise_func, replacement_table=None):
        self.data = []
        self.noise_func = noise_func
        self.replacement_table = replacement_table
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                ph = obj["phonemes"].strip().split()  # pure phoneme
                if len(ph) > 0:
                    self.data.append(ph)

        print("Loaded", len(self.data), "phoneme sequences")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clean_tokens = self.data[idx]             # list of clean phonemes
        noisy_tokens, modified_idx = self.noise_func(           # list of noisy phonemes
            clean_tokens,
            replacement_table=self.replacement_table
        )
        
        clean_ids = torch.tensor([phoneme_to_id[t] for t in clean_tokens], dtype=torch.long)
        noisy_ids = torch.tensor([phoneme_to_id[t] for t in noisy_tokens if t in phoneme_to_id],
                                 dtype=torch.long)
        noisy_ids, len2 = interleave_slot(noisy_ids, len(noisy_tokens), slot_id=41, pad_id=PAD_ID)
        return noisy_ids, clean_ids, torch.tensor(modified_idx, dtype=torch.long), len2, len(clean_ids)
    

def collate_fn(batch):
    noisy_list, clean_list, mod_idx_list,len_s,len_t = zip(*batch)

    noisy_padded = pad_sequence(noisy_list, batch_first=True, padding_value=PAD_ID)
    clean_padded = pad_sequence(clean_list, batch_first=True, padding_value=PAD_ID)

    B, T = clean_padded.size()

    # Build a weight mask (default = 1)
    weight_mask = torch.zeros((B, T), dtype=torch.float)

    # Mark modified positions as weight 5
    for i, (clean_ids, mod_idx) in enumerate(zip(clean_list, mod_idx_list)):
        for idx_modified in mod_idx.tolist():
            if idx_modified < len(clean_ids):  # safe check
                weight_mask[i, idx_modified] = 1

    attention_mask = (noisy_padded != PAD_ID).long()

    return {
        "input_ids": noisy_padded,
        "labels": clean_padded,
        "attention_mask": attention_mask,
        "weight_mask": weight_mask,
        'len_s': torch.tensor(len_s, dtype=torch.long),
        'len_t': torch.tensor(len_t, dtype=torch.long),
    }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class PhonemeDenoiseTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=PAD_ID
        )
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.conv1 = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=9, stride=1, padding=4, bias=False),nn.ReLU(),
                                   nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=2, bias=False),nn.ReLU(),
                                      nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=2, bias=False),nn.ReLU(),
                                   nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False),nn.ReLU())
        self.act=nn.ReLU()
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, len_s, len_t, attention_mask=None, labels=None, weight_mask=None):
        x = self.embedding(input_ids)  # [B, T, D]
        x0 = self.conv1(x.transpose(1,2)).transpose(1,2)
        x0 = self.act(x0)
        x1 = self.conv2(x.transpose(1,2)).transpose(1,2)
        x1 = self.act(x1)
        x = 0.5*x0 + 0.3*x1 + 0.2*x
        x = self.pos_encoding(x)

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # True on pad
        else:
            src_key_padding_mask = None

        encoded = self.encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # [B, T, D]

        logits = self.output_proj(encoded)  # [B, T, V]
        loss = None
        if labels is not None:
            loss_fct = nn.CTCLoss(blank=0, zero_infinity=True)
            loss = loss_fct(
                logits.permute(1, 0, 2),
                labels,
                len_s.squeeze(1),
                len_t.squeeze(1),
            )  # [B*T]
            # avoid divide-by-zero if all weights are zero
        return logits, loss


# ===== 5. Training loop =====

def train_denoiser(
    jsonl_path: str,
    replacement_table=None,
    batch_size: int = 32,
    num_epochs: int = 5,
    lr: float = 1e-4,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = PhonemeDenoiseDataset("./data/phoneme_sb.jsonl", noise_phoneme_tokens,
                                     replacement_table=None)  # or your table

    N = len(full_dataset)
    val_size = min(8120, N)  # in case dataset is smaller

    val_indices = list(range(val_size))
    train_indices = list(range(val_size, N))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = PhonemeDenoiseTransformer(
        vocab_size=len(LOGIT_TO_PHONEME),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            weight_mask = batch["weight_mask"].to(device)
            len_s = batch['len_s'].to(device)
            len_t = batch['len_t'].to(device)
            optimizer.zero_grad()
            _, loss = model(
                input_ids=input_ids,
                len_s=len_s[:,None],
                len_t=len_t[:,None],
                attention_mask=attention_mask,
                labels=labels,
                weight_mask=weight_mask,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % 1000 == 0:
                avg_loss = total_loss / 1000
                print(f"step {global_step}  loss {avg_loss:.4f}")
                total_loss = 0.0

        print(f"--- Epoch {epoch+1} finished. Train loss: {total_loss/len(train_loader):.4f} ---")

        # Evaluate after every epoch
        evaluate(model, val_loader, device)
    return model


def phoneme_error_rate(pred, target):
    import numpy as np
    dp = np.zeros((len(target)+1, len(pred)+1), dtype=int)

    for i in range(len(target)+1):
        dp[i][0] = i
    for j in range(len(pred)+1):
        dp[0][j] = j

    for i in range(1, len(target)+1):
        for j in range(1, len(pred)+1):
            cost = 0 if target[i-1] == pred[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete
                dp[i][j-1] + 1,      # insert
                dp[i-1][j-1] + cost  # substitute
            )

    dist = dp[len(target)][len(pred)]
    norm = max(len(target), len(pred))
    return dist / norm
 
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_per = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            weight_mask = batch["weight_mask"].to(device)
            len_s = batch['len_s'].to(device)
            len_t = batch['len_t'].to(device)

            logits, loss = model(input_ids, len_s, len_t, attention_mask=attention_mask, labels=None,weight_mask=weight_mask)

            # total_loss += loss.item()

            # ---- Compute PER ----
            preds = logits.argmax(dim=-1).cpu().tolist()
            trues = labels.cpu().tolist()
            len_s_cpu = len_s.cpu().tolist()
            len_t_cpu = len_t.cpu().tolist()

            for p, t, ls, lt in zip(preds, trues, len_s_cpu, len_t_cpu):
                ls = int(ls)
                lt = int(lt)

                p_tokens = [id_to_phoneme[idx] for idx in p[:ls]]
                t_tokens = [id_to_phoneme[idx] for idx in t[:lt]]

                p_tokens = remove_consecutive_duplicates(" ".join(p_tokens)).split()

                p_tokens = [ph for ph in p_tokens if ph != PAD_TOKEN]
                t_tokens = [ph for ph in t_tokens if ph != PAD_TOKEN]

                per = phoneme_error_rate(p_tokens, t_tokens)
                total_per += per
                count += 1

    avg_loss = total_loss / len(val_loader)
    avg_per = total_per / count

    print(f"VALIDATION  loss={avg_loss:.4f}   PER={avg_per:.4f}")
    return avg_loss, avg_per


def interleave_slot(x, lens: int, slot_id: int, pad_id: int):
    T = x.shape[0]
    T2 = 2 * T

    out = torch.full((T2,), pad_id, dtype=x.dtype, device=x.device)

    out[0::2] = x
    out[1::2] = slot_id

    lens2 = lens * 2
    return out, lens2


def remove_consecutive_duplicates(s: str) -> str:
    words = s.split()
    if not words:
        return s

    result = [words[0]]
    for w in words[1:]:
        if w != result[-1]:
            result.append(w)

    return " ".join(result)