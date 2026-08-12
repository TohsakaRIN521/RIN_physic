---
name: paper-completion-arxiv-en
description: "Complete the missing sections of a finished paper draft (abstract, introduction, supporting body paragraphs, conclusion, acknowledgments, references) without altering its reasoning or final results. Workflow: extract key theories/formulas/conclusions from the draft → search arXiv for related papers from the past year (top 40) → verify journals online, group by journal tier, and let the user pick 3/5/10 papers → download TeX sources → rewrite and integrate similar theory/method passages with proper citations. Triggers: requests to complete, fill in, polish the missing parts of, or add references to a paper whose main body is already written. Not for revising reasoning, changing conclusions, or writing from scratch."
---

# arXiv-Based Paper Completion

## Applicability (verify first)

- The paper body (theoretical derivation + numerical analysis) is already complete; do not change the reasoning or final conclusions.
- The task is to fill in: abstract, introduction, citable supporting paragraphs in the body, conclusion, acknowledgments, references.
- If the user actually wants to revise reasoning/conclusions or write from scratch, state that this skill does not apply and stop; do not force it.

## Workflow

### Step 1: Dissect the draft

Read the whole draft and extract:

- Theory/method keywords (e.g., Floquet theory, tight-binding model, density functional theory, Monte Carlo)
- Key formulas (record symbols and equation numbers)
- Core conclusions and numerical results

Produce a keyword list of theories / formulas / conclusions; base all subsequent searches on it.

### Step 2: Search arXiv (past year, top 40)

Run `scripts/search_arxiv.py` with the keyword list:

```bash
python scripts/search_arxiv.py "floquet theory" --limit 40 --years 1
```

The script prints a relevance-ranked list (arXiv ID, title, authors, publication date, journal_ref if any). Add `--json <file>` to save results for later steps.

If no results: broaden the keywords, drop journal-specific terms or obscure abbreviations, retry; extend `--years` as a fallback. Record the queries actually used.

### Step 3: Classify journals by tier, then let the user choose

1. For each result, use web search to confirm the final journal and its tier (tier refers to journal standing, not field — e.g., Phys. Rev. A/B ≈ second tier, Phys. Rev. Lett. ≈ first tier).
2. Papers with no confirmed journal go under "Unpublished / arXiv preprint".
3. Group results by tier into a table (arXiv ID / Title / Journal / Tier), send it to the user, and ask them to choose 3/5/10 papers for source download (default 5 if unspecified).
4. This is a human-confirmation gate: do not proceed to Step 4 until the user confirms; never decide the downloads yourself.

### Step 4: Download TeX sources

For the selected papers, run:

```bash
python scripts/download_arxiv_sources.py 2401.00001 2401.00002 --out work/tex_sources
```

The script downloads from `https://arxiv.org/e-print/<id>` and unpacks tar.gz / single-file gz / plain TeX, one subdirectory per paper. If a paper has no TeX source (PDF only), note it and fall back to extracting text from the PDF in Step 5.

### Step 5: Match and integrate similar passages (core)

For each paper:

1. Open its `.tex` source and search for theories/methods shared with the draft (e.g., the published paper also uses Floquet theory and cites the same original reference, and the draft also uses Floquet theory).
2. Locate complementary passages (intro background, derivation details, method explanations) and assess how similar they are to the draft content.
3. Rewrite and adapt them into the corresponding place in the draft, adding citations (`\cite`) and reference entries; never copy large passages verbatim.
4. Repeat until all selected papers are integrated. Then report to the user: the paper is ~95% complete; the remaining plagiarism check, polishing, and gap-filling need human collaboration (offer to continue section by section).

## Undergraduate variant

- If the author is an undergraduate, replace Step 2 with a keyword search on CNKI (https://www.cnki.net). Classification is still by journal tier; extract full-text content from PDFs (no TeX source). Everything else stays the same.

## Rules and boundaries

- Academic integrity: all integrated content must be rewritten and properly cited; the final paper must pass plagiarism checking; prefer citing the original sources.
- Intermediate files (search JSON, downloaded sources) go in the working directory, not the paper's directory.
- Step 3 is a human-confirmation gate: do not download or integrate before the user chooses.
