# Research Paper Parser

Extracts text from research-paper PDFs, isolates the Introduction section, and includes a Python port of the SIMetrix input-based summary evaluation features (KL/JS divergence, TF-IDF cosine, topic-signature coverage, unigram/multinomial likelihood).

## What and why

Evaluating a summary of a paper usually needs a human-written reference. SIMetrix (Louis & Nenkova) instead scores a summary directly against its *input* document using distributional similarity, which makes it useful for comparing summaries of papers where no gold summary exists. This repo is the groundwork for that pipeline:

1. Convert each PDF in `papers/` to plain text in `output/`.
2. Locate the Introduction (the text between the `1 I...` and `2 ...` section headings) and print it.
3. Compute input-vs-summary features with the classes in `main.py` (wired up but currently commented out in the driver loop; see Limitations).

## Approach

`main.py` contains:

| Component | Role |
|---|---|
| `vocabDist` | Vocabulary distribution: words, frequencies, total token count |
| `CorpusBasedUtilities` | Stopword removal, Porter stemming, background IDF and corpus counts, TF-IDF cosine, log-likelihood-ratio topic signatures |
| `EvalFeatures` | KL divergence (smoothed, both directions), JS divergence (smoothed and unsmoothed), % tokens that are topic words, fraction of topic words covered, unigram and multinomial log-probability |
| `InputBasedEvaluation` | Assembles the selected features into a feature string; holds the default config |
| Driver (bottom of file) | PDF to text with PyPDF2, then section extraction |

Background resources used by the features:

- `smart_common_words.txt`: SMART stopword list
- `bgFreqCounts.*`: background corpus word frequencies (for topic signatures)
- `bgIdfValues.*`: background IDF values, stemmed and unstemmed (for TF-IDF cosine)
- `config.example`: reference configuration in the original SIMetrix format
- `sampleMappings.txt*`: the sample input/summary mapping and reference output format from the original SIMetrix distribution

## Tech stack

Python 3, PyPDF2 (1.x API), NLTK (Porter stemmer), standard library (`re`, `math`, `os`).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "PyPDF2<3" nltk
python main.py
```

The script reads every PDF in `papers/`, writes `output/<name>.pdf.txt`, and prints the path and Introduction section of each extracted file.

## Usage

- Drop new PDFs into `papers/` and rerun `python main.py`.
- To compute evaluation features, uncomment the block at the end of `main.py` and point `inputVocabDist` / `summaryVocabDist` at the source text and the summary text respectively. Features are selected by name, e.g. `KLInputSummary`, `cosineAllWords`, `percentTopicTokens`, `unigramProb`.

## Limitations and next steps

- The feature-computation step is not yet enabled in the driver; the current run only does extraction and Introduction isolation.
- Section detection relies on the `\n1 I` / `\n2 ` heading pattern, which fits IEEE-style numbered papers but not every layout.
- `PdfFileReader` / `extractText` is the PyPDF2 1.x API; newer `pypdf` releases renamed these.
- Directory-mode vocabulary building joins paths with `\\` (Windows).
- Next: generate a candidate summary per paper (extractive or LLM-based), score it against the full text with the SIMetrix features, and report averages per system.

## Credits

The evaluation features follow SIMetrix by Annie Louis and Ani Nenkova (*Automatically Evaluating Content Selection in Summarization without Human Models*, EMNLP 2009); the background resources and sample mapping files come from that distribution. The sample papers in `papers/` are IEEE publications included for testing only.
