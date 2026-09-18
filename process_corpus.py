import os
import re

CORPUS_DIR = "corpus"
OUTPUT_FILE = "corpus_clean.txt"

# filler words to remove (can extend)
FILLERS = {"ooh", "uh", "uh-huh", "um", "hmm", "uhh"}
UTT_PATTERN = re.compile(r"utt\d+:\s*(.*)", re.IGNORECASE)

def clean_sentence(text: str):
    match = UTT_PATTERN.search(text)
    if not match:
        return None

    sent = match.group(1)

    # 1) remove {} <> [] (3 passes for nested cases)
    for _ in range(3):
        sent = re.sub(r"\{[^}]*\}", " ", sent)
        sent = re.sub(r"<[^>]*>", " ", sent)
        sent = re.sub(r"\[[^\]]*\]", " ", sent)

    # 2) remove leftover brackets just in case
    sent = re.sub(r"[\[\]\{\}\<\>]", " ", sent)

    # 3) remove / and -/
    sent = sent.replace("/"," ").replace("-/"," ")

    # 4) remove annotation + , - , #
    # patterns such as "+ word", ", - ,", "# something"
    sent = re.sub(r"\s[+\-#]\s", " ", sent)
    sent = re.sub(r"\s[+\-#](?=\W)", " ", sent)     # space + annotation before punctuation
    sent = re.sub(r"(?<=\W)[+\-#]\s", " ", sent)     # punctuation before annotation + space
    sent = re.sub(r"[+\-#]", " ", sent)              # final fallback

    # 5) normalize whitespace
    sent = re.sub(r"\s+", " ", sent).strip()
    if not sent:
        return None

    # 6) split into words and remove fillers & leftover symbols
    words = sent.split()
    cleaned_words = []
    for w in words:
        base = re.sub(r"[.,!?;:]+$", "", w).lower()
        if base in FILLERS:
            continue
        if base in {"+", "-", "#"}:
            continue
        cleaned_words.append(w)

    if len(cleaned_words) < 2:
        return None

    return " ".join(cleaned_words)

def process_corpus(root_dir: str):
    all_sentences = []

    for current_dir, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".utt"):
                file_path = os.path.join(current_dir, fname)

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        cleaned = clean_sentence(line.strip())
                        if cleaned:
                            all_sentences.append(cleaned)

    # write to one output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(all_sentences))

    print(f"Saved {len(all_sentences)} cleaned sentences to {OUTPUT_FILE}")


if __name__ == "__main__":
    process_corpus(CORPUS_DIR)