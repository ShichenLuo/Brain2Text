import numpy as np
import torch
from pyctcdecode import build_ctcdecoder

LOGIT_TO_PHONEME = [
    "BLANK",
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH",
    "|",
]

BLANK_ID = 0
BLANK_TOKEN = "BLANK"
SILENCE_TOKEN = "|"


def build_phoneme_decoder(lm_path, labels=LOGIT_TO_PHONEME):
    unigrams = [t for t in labels if t != BLANK_TOKEN]
    return build_ctcdecoder(labels=labels, kenlm_model_path=lm_path, unigrams=unigrams)


def _to_numpy_ctc(x, is_log_probs):
    if x.dim() == 2:
        lp = x if is_log_probs else torch.log_softmax(x, dim=-1)
        return lp.detach().cpu().numpy(), False
    if x.dim() == 3:
        lp = x if is_log_probs else torch.log_softmax(x, dim=-1)
        return lp.detach().cpu().numpy(), True
    raise ValueError("Expected [T,C] or [B,T,C]")


def _ctc_collapse(best_ids, blank_id):
    out_ids = []
    out_ts = []
    prev = None
    for t, idx in enumerate(best_ids.tolist()):
        if idx == prev:
            continue
        prev = idx
        if idx == blank_id:
            continue
        out_ids.append(int(idx))
        out_ts.append(int(t))
    return out_ids, out_ts


def phoneme_decode_greedy_split(x, labels=LOGIT_TO_PHONEME, blank_id=BLANK_ID, is_log_probs=False, show_silence=True):
    lp_np, batched = _to_numpy_ctc(x, is_log_probs=is_log_probs)

    def one(lp_tc):
        best = np.argmax(lp_tc, axis=-1)
        ids, ts = _ctc_collapse(best, blank_id)
        tokens = [labels[i] for i in ids]

        if show_silence:
            text = " ".join(" | " if t == SILENCE_TOKEN else t for t in tokens)
            text = " ".join(text.split())
        else:
            text = " ".join(t for t in tokens if t != SILENCE_TOKEN)

        return {"text": text, "tokens": tokens, "ids": ids, "timesteps": ts}

    if not batched:
        return one(lp_np)
    return [one(lp_np[b]) for b in range(lp_np.shape[0])]


def _split_concat_phonemes(decoded_text, labels):
    if " " in decoded_text:
        return [t for t in decoded_text.strip().split() if t]

    vocab = sorted((set(labels) - {BLANK_TOKEN}), key=len, reverse=True)

    out = []
    i = 0
    while i < len(decoded_text):
        matched = None
        for tok in vocab:
            if decoded_text.startswith(tok, i):
                matched = tok
                break
        if matched is None:
            out.append(decoded_text[i])
            i += 1
        else:
            out.append(matched)
            i += len(matched)
    return out

def move_blank_to_last(logits):
    # logits: [T,C] or [B,T,C], where blank is at index 0
    # pyctcdecode expects blank as its own internal token (usually last index)
    return torch.cat([logits[..., 1:], logits[..., :1]], dim=-1)


def phoneme_beam_decode_with_lm(logits, decoder, labels=LOGIT_TO_PHONEME, beam_width=200, show_silence=True):
    logits = move_blank_to_last(logits)
    lp_np, batched = _to_numpy_ctc(logits, is_log_probs=False)

    def one(lp_tc):
        raw = decoder.decode(lp_tc, beam_width=beam_width)
        tokens = _split_concat_phonemes(raw, labels)

        if show_silence:
            text = " ".join(" | " if t == SILENCE_TOKEN else t for t in tokens)
            text = " ".join(text.split())
        else:
            text = " ".join(t for t in tokens if t != SILENCE_TOKEN)

        return {"text": text, "tokens": tokens, "raw_text": raw}

    if not batched:
        return one(lp_np)
    return [one(lp_np[b]) for b in range(lp_np.shape[0])]

PHONEME_VOCAB = sorted(set(LOGIT_TO_PHONEME[1:]) - {"|"}, key=len, reverse=True)

def split_to_phonemes(text):
    # remove spaces, because pyctcdecode may insert them in strange places
    s = text.replace(" ", "")
    out = []
    i = 0
    while i < len(s):
        m = None
        for tok in PHONEME_VOCAB:
            if s.startswith(tok, i):
                m = tok
                break
        if m is None:
            # unknown char, keep it so you can see issues
            out.append(s[i])
            i += 1
        else:
            out.append(m)
            i += len(m)
    return out

def format_phonemes(tokens, add_word_bars=True):
    # Simple heuristic: put a bar between “chunks”
    # Default: bar between every chunk separated by original decoder spaces
    # If you want better word boundaries, you need a pronunciation lexicon or word LM.
    if not add_word_bars:
        return " ".join(tokens)

    # If you pass in tokens per chunk, you can add bars between chunks.
    # Here we just add bars between all phonemes as a placeholder: you can adjust.
    return " ".join(tokens)

def decode_and_resplit(pred_dict, add_bars_between_decoder_words=True):
    raw = pred_dict["raw_text"]

    if add_bars_between_decoder_words:
        # Treat each whitespace separated segment as a chunk, then resplit inside
        chunks = [c for c in raw.strip().split() if c]
        all_tokens = []
        for k, ch in enumerate(chunks):
            phs = split_to_phonemes(ch)
            all_tokens.extend(phs)
            if k != len(chunks) - 1:
                all_tokens.append("|")
        text = " ".join(all_tokens)
        return {"text": text, "tokens": all_tokens, "raw_text": raw}

    # No bars, just split everything
    tokens = split_to_phonemes(raw)
    text = " ".join(tokens)
    return {"text": text, "tokens": tokens, "raw_text": raw}



def edit_distance(a, b):
    # a, b are lists of phoneme tokens
    n = len(a)
    m = len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,        # delete
                dp[j - 1] + 1,    # insert
                prev + cost       # substitute
            )
            prev = cur
    return dp[m]




def load_lexicon_prons(lexicon_path):
    # returns list of (word, pron_list)
    prons = []
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            w = parts[0]
            pron = parts[1:]
            prons.append((w, pron))
    return prons


def build_pron_index(prons):
    # index by (length, first_phone) for quick filtering
    idx = {}
    for w, pron in prons:
        if not pron:
            continue
        key = (len(pron), pron[0])
        idx.setdefault(key, []).append((w, pron))
    return idx

def split_by_bar(phonemes, bar="|"):
    chunks = []
    cur = []
    for p in phonemes:
        if p == bar:
            if cur:
                chunks.append(cur)
                cur = []
        else:
            cur.append(p)
    if cur:
        chunks.append(cur)
    return chunks

def candidates_for_chunk(chunk, pron_index, max_edits=2, len_window=2, max_cands=50):
    # chunk is list of phonemes for one word like ["D","AH","Z"]
    L = len(chunk)
    first = chunk[0] if chunk else None
    pool = []

    # search pronunciations with similar length and same first phone first
    for dL in range(-len_window, len_window + 1):
        key = (L + dL, first)
        for item in pron_index.get(key, []):
            pool.append(item)

    # if none found, relax first phone constraint
    if not pool:
        for dL in range(-len_window, len_window + 1):
            key_len = L + dL
            for (klen, kfirst), items in pron_index.items():
                if klen == key_len:
                    pool.extend(items)

    scored = []
    for w, pron in pool:
        dist = edit_distance(chunk, pron)
        if dist <= max_edits:
            scored.append((dist, w))

    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[:max_cands]

def decode_chunks_with_word_lm(chunks, lexicon_path, word_lm_bin, beam_size=30,
                              max_edits=2, edit_penalty=2.0,
                              word_bonus=2.0, unk_word="<unk>", unk_penalty=6.0):
    """Decode with the legacy KenLM word model when explicitly requested."""
    try:
        import kenlm
    except ImportError as exc:
        raise ImportError(
            "The legacy word-LM decoder requires KenLM. "
            "The primary Brain2Text pipeline uses an LLM instead."
        ) from exc

    prons = load_lexicon_prons(lexicon_path)
    pron_index = build_pron_index(prons)
    lm = kenlm.Model(word_lm_bin)

    init_state = kenlm.State()
    lm.BeginSentenceWrite(init_state)

    hyps = [{"score": 0.0, "words": [], "state": init_state}]

    def prune(hs):
        hs.sort(key=lambda x: x["score"], reverse=True)
        return hs[:beam_size]

    for chunk in chunks:
        cands = candidates_for_chunk(chunk, pron_index, max_edits=max_edits)

        new_hyps = []
        for h in hyps:
            # normal candidates
            for dist, w in cands:
                st2 = kenlm.State()
                add = lm.BaseScore(h["state"], w, st2)
                sc = h["score"] + add + word_bonus - edit_penalty * dist
                new_hyps.append({"score": sc, "words": h["words"] + [w], "state": st2})

            # fallback unknown if no candidates
            if not cands:
                st2 = kenlm.State()
                add = lm.BaseScore(h["state"], unk_word, st2)
                sc = h["score"] + add + word_bonus - unk_penalty
                new_hyps.append({"score": sc, "words": h["words"] + [unk_word], "state": st2})

        hyps = prune(new_hyps)

    finals = []
    for h in hyps:
        st2 = kenlm.State()
        end_add = lm.BaseScore(h["state"], "</s>", st2)
        finals.append({"score": h["score"] + end_add, "words": h["words"]})

    finals.sort(key=lambda x: x["score"], reverse=True)
    best = finals[0] if finals else {"score": -1e9, "words": []}
    return " ".join(best["words"]), best