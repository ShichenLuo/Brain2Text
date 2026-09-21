import numpy as np
import pandas as pd

# Your phoneme label list: index -> label
LABELS = [
    'BLANK',    # 0
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',      # silence token
    '<DEL>',    # pseudo phoneme for deletion
    '<INS>',    # pseudo phoneme for insertion
]

DEL_IDX = LABELS.index('<DEL>')
INS_IDX = LABELS.index('<INS>')


def parse_idx_seq(seq_str: str):
    """
    seq_str: e.g. "0 10 12 40"
    returns: [0, 10, 12, 40]
    """
    seq_str = seq_str.strip()
    if not seq_str:
        return []
    return [int(x) for x in seq_str.split()]


def align_levenshtein_idx(ref_idxs, hyp_idxs):
    """
    Levenshtein alignment on index sequences.
    Returns list of (ref_idx_or_DEL_IDX, hyp_idx_or_INS_IDX).
    """
    n = len(ref_idxs)
    m = len(hyp_idxs)

    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        dp[i, 0] = i
    for j in range(1, m + 1):
        dp[0, j] = j

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = 0 if ref_idxs[i - 1] == hyp_idxs[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,          # deletion
                dp[i, j - 1] + 1,          # insertion
                dp[i - 1, j - 1] + cost_sub  # match / substitution
            )

    # Backtrace to recover alignment
    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost_sub = 0 if ref_idxs[i - 1] == hyp_idxs[j - 1] else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost_sub:
                aligned.append((ref_idxs[i - 1], hyp_idxs[j - 1]))
                i -= 1
                j -= 1
                continue

        # deletion
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            aligned.append((ref_idxs[i - 1], DEL_IDX))
            i -= 1
        else:
            # insertion
            aligned.append((INS_IDX, hyp_idxs[j - 1]))
            j -= 1

    aligned.reverse()
    return aligned


def build_phoneme_confusion_from_str_indices(all_refs, all_preds, labels=LABELS):
    """
    all_refs, all_preds: lists of strings
        e.g. ["0 10 12 40", "5 7 9", ...]

    Returns:
        confusion_df: DataFrame with row/col index = phoneme labels
                      (including <DEL> and <INS>).
    """
    assert len(all_refs) == len(all_preds), "refs and preds must have same length"

    K = len(labels)
    cm = np.zeros((K, K), dtype=int)

    for ref_str, hyp_str in zip(all_refs, all_preds):
        ref_idxs = parse_idx_seq(ref_str)
        hyp_idxs = parse_idx_seq(hyp_str)

        aligned_pairs = align_levenshtein_idx(ref_idxs, hyp_idxs)
        for r_idx, h_idx in aligned_pairs:
            if 0 <= r_idx < K and 0 <= h_idx < K:
                cm[r_idx, h_idx] += 1

    confusion_df = pd.DataFrame(cm, index=labels, columns=labels)
    confusion_df.to_csv("phoneme_confusion_matrix.csv")
    return confusion_df


# all_refs = [
#     "7 21 17 39",       # e.g. B R IH NG
#     "10 3 7 26"        # e.g. DH AH B OY
# ]
# all_preds = [
#     "7 21 17 39",       # perfect
#     "10 3 7"            # missing OY (deletion)
# ]

# cm_df = build_phoneme_confusion_from_str_indices(all_refs, all_preds)
# print(cm_df)

# # Save as CSV
# cm_df.to_csv("phoneme_confusion_matrix.csv")