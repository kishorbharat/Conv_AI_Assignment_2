\# Pre-training a GPT-Style (Decoder-Only) LLM from Scratch

This repository implements an end-to-end pre-training pipeline for a small Decoder-only Transformer model using domain PDFs.

The core implementation is in [src/run_rag_terminal.py](src/run_rag_terminal.py). A compatibility launcher is kept at [run_rag_terminal.py](run_rag_terminal.py), so existing commands continue to work.

It follows the assignment's 6 required phases:

1. Data collection, PDF extraction, and cleaning
2. Dataset generation with custom tokenizer and CLM shift labels
3. Input embeddings (token + positional)
4. Decoder-only Transformer forward pass with causal mask
5. Loss computation and optimization
6. Inference via autoregressive generation

## Project Structure (Production-Ready)

```text
Conv_AI_Group_Assignment/
├── src/
│   ├── __init__.py
│   └── run_rag_terminal.py
├── data/
│   ├── pdfs/
│   │   └── AIS_197-1_BNCAP.pdf
│   └── processed/
│       ├── ais197_raw.txt
│       └── ais197_clean.txt
├── tokenizer/
│   ├── vocab.json
│   └── merges.txt
├── artifacts_group1/
│   └── decoder_only_15m.pt
├── evaluation/
├── submission/
├── legacy/
├── run_rag_terminal.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment Setup

Install required packages:

```bash
pip install torch pymupdf
```

Alternative PDF parser (if needed):

```bash
pip install pdfplumber
```

## Assignment Pipeline Mapping

### Step 1: Data Collection, Extraction, and Curation

Goal:
Build a clean domain corpus from multiple PDF files.

Code components:

- `extract_pdfs_to_raw_text(...)`
- `clean_raw_text(...)`

Inputs:

- PDF directory: [data/pdfs](data/pdfs)

Outputs:

- Raw extracted text: [data/processed/ais197_raw.txt](data/processed/ais197_raw.txt)
- Cleaned corpus text: [data/processed/ais197_clean.txt](data/processed/ais197_clean.txt)

How cleaning works:

- Removes page markers and file separators
- Removes page-number artifacts (for example: `Page 4 of 22`, standalone numbers)
- Repairs hyphenated line breaks from PDFs
- Normalizes whitespace and punctuation spacing

Run:

```bash
python run_rag_terminal.py extract
python run_rag_terminal.py clean
```

### Step 2: Dataset Generation (Text to Training Instances)

Goal:
Convert clean text to token IDs and create CLM training pairs.

Code components:

- `train_bpe_tokenizer_from_scratch(...)`
- `save_tokenizer(...)`
- `encode_text(...)`
- `CLMDataset`

Inputs:

- Clean corpus: [data/processed/ais197_clean.txt](data/processed/ais197_clean.txt)

Outputs:

- Tokenizer vocabulary: [tokenizer/vocab.json](tokenizer/vocab.json)
- Tokenizer merges: [tokenizer/merges.txt](tokenizer/merges.txt)
- Training sequences from `CLMDataset`

CLM objective in data loader:

- Input sequence X: `[t1, t2, t3, ..., t(n-1)]`
- Label sequence Y: `[t2, t3, t4, ..., t(n)]`

Run:

```bash
python run_rag_terminal.py train-tokenizer
```

### Step 3: Model Architecture and Input Embeddings

Goal:
Map token IDs to dense vectors and inject token position information.

Code components:

- `DecoderOnlyTransformer.token_embedding` (token embeddings)
- `SinusoidalPositionalEncoding` (sine/cosine positional encoding)

Inputs:

- Batch token IDs from `CLMDataset`

Outputs:

- Final input embedding = token embedding + positional encoding

### Step 4: Forward Pass (Decoder-only Transformer)

Goal:
Process sequence through masked self-attention and predict next-token logits.

Code components:

- `DecoderBlock`
- `DecoderOnlyTransformer.forward(...)`
- `lm_head` linear projection to vocabulary size

Inputs:

- Embedded token sequences
- Causal attention mask

Outputs:

- Logits tensor with shape `[batch_size, seq_len, vocab_size]`

Masking behavior:

- Position `i` can only attend to positions `<= i`
- Future tokens are blocked to prevent information leakage

### Step 5: Loss Calculation and Backpropagation

Goal:
Train model with cross-entropy next-token loss and update parameters.

Code components:

- `F.cross_entropy(...)` in `DecoderOnlyTransformer.forward(...)`
- `train_model(...)` training loop
- `torch.optim.AdamW`

Inputs:

- Logits from model forward pass
- Shifted CLM labels from data loader

Outputs:

- Scalar training loss
- Updated model weights
- Saved model checkpoint: [artifacts_group1/decoder_only_15m.pt](artifacts_group1/decoder_only_15m.pt)

Run:

```bash
python run_rag_terminal.py train-model
```

### Step 6: Inference and Prediction (Autoregressive Generation)

Goal:
Generate text continuation token by token from a prompt.

Code components:

- `DecoderOnlyTransformer.generate(...)`
- `generate_text(...)`
- `run_five_prompts(...)`

Inputs:

- Prompt text
- Trained tokenizer
- Trained model checkpoint

Outputs:

- Generated continuation text
- At least five generation samples for analysis

Run one prompt:

```bash
python run_rag_terminal.py generate --prompt "The company policy states"
```

Run five prompts:

```bash
python run_rag_terminal.py demo-5
```

## Full Pipeline Command

Run all phases in one command:

```bash
python run_rag_terminal.py run-all
```

This executes:

1. PDF extraction
2. Cleaning
3. Tokenizer training
4. Model training
5. One generation example

## Main CLI Commands

- `extract`
- `clean`
- `train-tokenizer`
- `train-model`
- `generate`
- `demo-5`
- `run-all`

You can inspect all arguments with:

```bash
python run_rag_terminal.py --help
python run_rag_terminal.py train-model --help
```

## Deliverables Checklist (Assignment-aligned)

1. Source Code (complete pipeline)
	- Core: [src/run_rag_terminal.py](src/run_rag_terminal.py)
	- Launcher: [run_rag_terminal.py](run_rag_terminal.py)
2. Trained Tokenizer Files
	- [tokenizer/vocab.json](tokenizer/vocab.json)
	- [tokenizer/merges.txt](tokenizer/merges.txt)
3. Extraction and Cleaning Description
	- Document noise patterns and cleaning decisions in notebook/report
4. Pipeline Explanation (all 6 steps)
	- This README and notebook markdown sections
5. Prediction Examples (minimum 5)
	- `demo-5` outputs, plus analysis of coherence and domain specificity

## Mandatory Final Submission Details (4-Member Group)

As required, include both items below in the final submission.

### 1. Individual Contribution Percentage (Must total 100%)

Fill this table with the actual contribution of each member:

| Team Member | Student ID | Contribution (%) | Key Work Completed |
| --- | --- | ---: | --- |
| Kishor Bharat | 2024TM05030 | 30 | Data collection from PDFs, extraction pipeline setup, raw corpus validation, and cleaning rule definition |
| Saurabh Mani | 2024TM05067 | 25 | Tokenizer training (BPE), dataset preparation with CLM shift labels, and data pipeline checks |
| Saranjit Singh | 2024TM05027 | 25 | Decoder-only Transformer implementation, training loop configuration, and checkpoint generation |
| Aniruddha Madan Patil | 2024TM05041 | 20 | Inference/demo prompts, output analysis, README/report consolidation, and final submission packaging |
| **Total** |  | **100** |  |

Note: Update percentages and descriptions if your actual contribution split differs.

### 2. Group Demonstration Video (All members must speak)

Record one group demonstration video that:

- Clearly explains every task in the assignment activity
- Justifies the final results and inferences
- Includes contribution from all team members in the explanation

Upload the video to Microsoft Stream (on SharePoint) and submit the shared link.

Use this submission format:

- Microsoft Stream Link: `<paste-share-link-here>`
- Access: `Anyone with university access can view`
- Duration: `<video-length>`

## Notes for Final Report / Notebook Write-up

Include these sections in your notebook:

1. Domain choice and PDF corpus size (target 10-50 MB)
2. Extraction issues found in PDFs
3. Cleaning rules and rationale
4. Tokenizer design choices (vocab size, min frequency)
5. Model design choices (layers, heads, embedding size, block size)
6. Training metrics (loss, perplexity trend)
7. Five generated examples and qualitative analysis

## Predictions (Latest AIS-197 / BNCAP Q&A)

The following predictions were generated from the updated AIS-197 corpus (AIS_197-1_BNCAP.pdf) after retraining.

1. Prompt: What are minimum qualifying points for star ratings for AOP?
	Generated Answer: Table 2A lists minimum AOP scores as 27, 22, 16, 10, and 4 (in descending star order), with Safety Assist Technologies as a qualifier (refer Annexure VI).

2. Prompt: What tests are proposed for Adult Occupant Protection?
	Generated Answer: The proposed AOP tests are ODB Frontal Impact Test, MDB Side Impact Test, and Pole Side Impact Test.

3. Prompt: How many total assessment points are for Adult Occupant Protection?
	Generated Answer: Total AOP points are 32, split as 16 points for ODB Frontal Impact Test and 16 points for MDB Side Impact Test.

4. Prompt: What is the process to apply for BNCAP?
	Generated Answer: OEM submits application details (Form 70-A) after nomination/selection, the Designated Agency evaluates and issues fee/sample-selection instructions, test samples are submitted to allocated test agency, tests are conducted and reported (Form 70-B), and final rating is published by the Designated Agency.

5. Prompt: What is the intrusion measurement limits before and after test?
	Generated Answer: The protocol defines pre- and post-test coordinate measurement and comparison; the explicit numeric limit in this section is the 3D measurement system tolerance of +/- 1 mm (no separate fixed pass/fail intrusion threshold is stated in that subsection).
