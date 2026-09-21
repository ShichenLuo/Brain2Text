import re
import json
from g2p_en import G2p

LOGIT_TO_PHONEME = [
    'BLANK',    # CTC blank symbol
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    '|',    # "|" = silence token
]

SIL_DEF = ['|']

def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join(sentence.split())

    return sentence

def sentence_to_phonemes(thisTranscription, g2p_instance=None):
    if g2p_instance is None:
        g2p_instance = G2p()

    original_text = thisTranscription  # keep original for "target"

    # Remove punctuation for g2p input
    thisTranscription = remove_punctuation(thisTranscription)

    # Convert to phonemes
    phonemes = []
    if len(thisTranscription) == 0:
        phonemes = SIL_DEF
    else:
        for p in g2p_instance(thisTranscription):
            if p == ' ':
                phonemes.append('|')
                continue

            # remove stress numbers
            p = re.sub(r'[0-9]', '', p)

            # keep only phoneme tokens (capital letters)
            if re.fullmatch(r'[A-Z]+', p):
                phonemes.append(p)

        # add one SIL symbol at the end so there is one at the end of each sentence
        phonemes.append('|')

    return phonemes, thisTranscription

def build_phoneme_jsonl(
    input_txt_path="corpus_clean.txt",
    output_jsonl_path="phoneme_sb.jsonl"
):
    g2p_instance = G2p()

    with open(input_txt_path, "r", encoding="utf-8") as fin, \
         open(output_jsonl_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            phoneme_list, target_text = sentence_to_phonemes(
                line, g2p_instance=g2p_instance
            )

            # join phonemes into a single string with spaces
            phoneme_str = " ".join(phoneme_list)

            record = {
                "phonemes": phoneme_str,
                "target": target_text
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # import nltk

    # nltk.download('averaged_perceptron_tagger_eng')
    build_phoneme_jsonl("corpus_clean.txt", "phoneme_sb.jsonl")