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

## Predictions (Latest AIS-175 WLTP Q&A)

The following predictions were generated from the AIS-175 WLTP corpus after training.

1. **Prompt:** Summarise AIS 175 Standard in 100 words

   **Generated Answer:** AIS 175 is an Indian Automotive Industry Standard that establishes the test method, testing equipment, and procedures for type approval, Conformity of Production (COP), and In-Service Conformity (ISC) testing under the Worldwide harmonized Light vehicle Test Procedure (WLTP). It applies to M1, M2, and N1 category vehicles with a GVW not exceeding 3,500 kg. The standard covers Type I (emissions), Type II (idle), Type III (crankcase), Type IV (evaporative), and Type V (durability) tests, along with OBD, Real Driving Emissions (RDE), and gear-shift procedures. It is aligned with UN GTR 15 Amendment 6 and adopted by India under CMV Rules 115, 116, and 126.

---

2. **Prompt:** What is Roadload Family?

   **Generated Answer:** A Road Load Family (defined in paragraph 6.3.3) is a group of vehicles that are identical with respect to: (a) transmission type and model (e.g. manual, automatic, CVT); (b) n/v ratios (engine speed divided by vehicle speed), with a tolerance within 25 per cent of the most commonly installed type; and (c) number of powered axles. Vehicles sharing the same road load family can use a single coastdown measurement to derive road load coefficients for all members, avoiding the need to test every variant individually on the road.

---

3. **Prompt:** What is Interpolation Family?

   **Generated Answer:** An Interpolation Family (defined in paragraph 6.3.2) is a group of vehicles between which CO2 emissions and fuel consumption can be mathematically interpolated using two boundary vehicles — Vehicle High (H) and Vehicle Low (L). For pure ICE vehicles, all members must be identical in: (a) ICE fuel type, combustion process, engine capacity, and charging system; (b) CO2-influencing powertrain operation strategy; (c) transmission type and model; (d) n/v ratios within 8 per cent; and (e) number of powered axles. Separate sub-families apply for NOVC-HEVs, OVC-HEVs, PEVs, and FCHVs, each with additional criteria covering electric machine type, REESS, and energy converter characteristics.

---

4. **Prompt:** Which type of passenger cars are covered in this standard?

   **Generated Answer:** AIS 175 covers all vehicles of category M1 (passenger cars), and also M2 and N1 category vehicles whose technically permissible maximum laden mass does not exceed 3,500 kg. The standard applies to these vehicles for the WLTP Type I test (gaseous and particulate emissions, CO2, fuel efficiency, electric energy consumption, and electric range) and the Type IV evaporative emissions test. In addition, it lays down rules for Type II, III, V, OBD, RDE, COP, and ISC tests for the covered vehicle categories.

---

5. **Prompt:** What are criteria for including new vehicle in interpolation family?

   **Generated Answer:** Under paragraph 6.3.2.1.2, a new vehicle may join an existing interpolation family only if it is identical to all current members with respect to: (a) ICE type — fuel type, combustion process, engine capacity, full-load characteristics, engine technology, and charging system; (b) operation strategy of all CO2-influencing powertrain components; (c) transmission type and model; (d) n/v ratios within 8 per cent of the most commonly installed transmission; and (e) number of powered axles. Additionally, any alternative parameter (e.g. higher nmin_drive or ASM) used within the family must remain the same for the new vehicle.

---

6. **Prompt:** Do this standard cover Flex Fuel vehicles? explain in short

   **Generated Answer:** Yes. AIS 175 explicitly covers Flex Fuel vehicles. Paragraph 3.3.25 defines a Flex Fuel Vehicle as one with a single fuel storage system capable of running on different mixtures of two or more fuels, and paragraph 3.3.26 defines the Flex Fuel Ethanol Vehicle sub-type (petrol/ethanol blends up to E85). Section 5.8 contains dedicated additional requirements for type approval of flex fuel ethanol and biodiesel vehicles, including the requirement that the vehicle can adapt to any fuel mixture available in the market, and that no manual engine re-adjustment is needed during fuel transition between reference fuels (paragraph 5.8.2). The interpolation family criteria in 6.3.2.1.2(a) also explicitly reference flex-fuel and bi-fuel fuel types.
