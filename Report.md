# Developing an Outreach Priority Scoring (OPS) for Potential Funders of Iterlight: Given Small Data, No Labels, and Limited Time
## A quantitative sensitivity analysis of different OPS design choices
### Full Technical Report

---

# 1. Objective

Our goal is to develop an **Outreach Priority Scoring (OPS)** framework that ranks potential funders by how promising they are for IterLight. For each organization \(x\), we want a score \(f(x)\) that accounts for the probaility of a given org to fund Iterlight, and the amount they would receive given funding. The main objective will be to estimate this probability function p(x).

We are under **small-data and limited-time constraints**:

- Only ~10 proxy-labeled positive EdTech funders,  
- Limited/no hard negative labels yet,  
- One main dataset of ~150 candidates.

So the OPS framework will:

1. Rely heavily on **unsupervised representation learning**,  
2. Use **semi-supervised / proxy labels** to orient the space,  
3. Support **multiple heuristic scoring functions** that we can compare (Ranking 1–6),  
4. Consider additional **LLM-based rankings** as a complementary perspective and use as an additional evaluation of our OPS rankings.

---

# 2. Data Source

Our analysis begins with an **LLM-generated funder dataset** provided by IterLight’s founder. This file was created through a structured prompt that asked a language model to enumerate potential funders and generate standardized metadata for each. The intention was to construct a high-recall universe of possible supporters before filtering down to true financial funders relevant for modeling.

---

## 2.1 The Dataset (LLM-Generated)

The original dataset contained **~200 organizations**, spanning:

- Philanthropic foundations  
- Corporate CSR programs  
- Venture capital / EdTech investors  
- Government agencies  
- Community sports nonprofits  
- Nonprofits providing equipment or in-kind services (not always financial funding)

The LLM generated a standardized schema for each organization, including:

- **Organization Name**  
- **Category** (foundation, CSR, VC, agency, nonprofit, etc.)  
- **Geographic Scope** (focus )  
- **DEI Priorities** (explicit DEI focus)  
- **Mission / Stated Pillars**  
  - A short free-text summary used later for text embeddings  
- **Typical Grant Size**  
- **Tier**  
- **Partnership Potential**  
- **Best Pitch Angle / Strategic Notes / Enhanced Notes**  
- **Application Process & Contact Information**  
- **Funder Information**  

This dataset was intentionally broad — designed to capture many plausible funders before narrowing.

---

## 2.2 Cleaning & Narrowing to Relevant Financial Funders

Before any modeling could occur, the LLM-generated dataset required extensive **cleaning, verification, and feature engineering**. The raw file was extremely messy and incomplete: funding amounts were categorical or inconsistent, mission statements contained irrelevant information, geographic descriptors were vague, and many organizations were repeated or misidentified. The majority of our early pipeline work was dedicated to producing a reliable, analysis-ready dataset.

### 2.2.1 Identifying Actual Financial Funders  
Because the OPS model is designed to prioritize **real monetary funders**, the first and most important filtering step was removing all organizations that do **not** provide financial grants. The LLM-generated list incorrectly included many groups that offered only equipment, programming, volunteers, or sports-related services.  

As part of feature engineering, we created a corrected binary variable:

- **`Financial Funder = 1`** if the organization actually awards monetary grants  
- **`Financial Funder = 0`** otherwise i.e., school supplies, merch, etc.

This manual verification required reviewing program guidelines, grant histories, and philanthropic disclosures for nearly every organization.

We kept only organizations where:

- `Financial Funder = 1`  
- Grant size information could be verified and expressed numerically  
- A usable mission statement existed (or could be sourced externally)  
- All essential OPS modeling fields were present

### 2.2.2 Correcting & Standardizing Grant Size Information  
Funding information was the messiest part of the LLM-generated dataset. Typical grant size fields were:

- missing,  
- contradictory across entries,  
- or categorical (“medium-sized grant,” “significant support”).  

To use grant size as a quantitative modeling feature, we manually **researched and verified actual funding ranges** for each organization. This required searching public databases, CSR reports, and 990 forms. We replaced the LLM’s vague categories with verified numerical values for:

- `Typical Min. Grant Size`  
- `Typical Max. Grant Size`

This step alone corrected dozens of inaccurate or unusable entries.

### 2.2.3 Engineering Geographic Focus  
The LLM-provided geographic descriptors were inconsistent (“regional,” “national,” “local impact,” etc.). Because IterLight benefits from identifying funders with direct or adjacent geographic relevance, we engineered a binary feature:

- **`geo_focus = 1`** if the organization operates in, funds, or is programmatically aligned with NYC or the surrounding region  
- **`geo_focus = 0`** otherwise  

We manually reviewed all ~200 original entries and assigned `geo_focus = 1` for organizations connected to the following regions, else `geo_focus = 0`:

- **New York (NY)**  
- **Vermont (VT)**  
- **Massachusetts (MA)**  
- **New Jersey (NJ)**  
- **Pennsylvania (PA)**  
- **Connecticut (CT)**  
- **Washington, D.C.**  
- **Ontario, Canada**

These areas either **border New York State** or have direct geographic proximity to NYC. Each organization’s regional eligibility, chapter locations, and funding footprint were manually inspected to assign this feature accurately.

### 2.2.4 Cleaning Mission Statements for Embeddings  
Mission statements are later encoded via SentenceTransformer and inform the structure of the UMAP + HDBSCAN clustering. To prevent bias or leakage, we cleaned the mission text extensively:

- **Removed monetary values** (to avoid embedding grant amounts)  
- **Removed geographic references** (to avoid geography dominating embedding structure)  
- **Filled in missing missions** by searching websites and public descriptions  
- **Deduplicated** repeated or near-duplicate organizations  
- **Standardized formatting** for consistency

These steps ensured that the embedding space was shaped by **true organizational mission**, not by superficial cues.

### 2.2.5 Core Fields Retained for Modeling  
After cleaning and verification, the fields required for OPS modeling — and therefore **retained during filtering** — were:

- `Organization Name`  
- `Geographic focus`  
- `clean mission statements` (engineered from the raw mission text)  
- `Typical Min. Grant Size`  
- `Typical Max. Grant Size`

Only organizations with complete values for these essential columns were kept.

Fields that were excluded from the analysis and kept for more interior use were things like:

- `DEI Priorities`  
- `Partnership Potential` 
- `Tier`
- `Best Pitch Angle / Strategic Notes`
- `Category`  

---

After all cleaning, standardization, research, and filtering, the dataset narrowed from the original **~200 organizations** to approximately **150 true, verified financial funders** with complete, model-ready information.

---

# 3. OPS Building Blocks and Global Experimental Design

All rankings reuse the same **cleaned dataset** and **representation pipeline** introduced in Section 2. We treat this pipeline as fixed background machinery: every ranking method simply plugs these shared ingredients into a different scoring function \(f_k(x)\).

## 3.1 Shared OPS Building Blocks

The common ingredients available to every ranking are:

- **mission_clean** — cleaned mission text  
- **numeric grant ranges** — min/max typical grant  
- **geo_focus** — engineered geographic alignment  
- **transformer embeddings** of `mission_clean`  
- **UMAP + HDBSCAN clusters** on those embeddings  

Key quantities referred to throughout:

- **zₓ** — embedding vector for organization x  
- **μ(x)** — expected grant capacity  
- **geo(x)** — geographic weight  
- **w_cluster(x)** — prior weight for the cluster of x  
- **soft_fit(x)** — confidence of cluster membership  
- **sim_iter(x)** — similarity to IterLight  
- **p(x)** — composite “probability-like” score derived from these components  

Each ranking \(R_k\) constructs a different scoring function \(f_k(x)\) using these shared pieces.

---

## 3.2 Global Experimental Design

To understand how different modeling choices affect final outreach rankings, we run a **structured set of controlled comparisons**.

### 3.2.1 Two heuristics (H1, H2)

We define two scoring heuristics:

- **H1:** Cluster-aware expected value (baseline)  
- **H2:** Risk-adjusted or alternative weighting heuristic  

### 3.2.2 Three representation variations per heuristic

For each heuristic, we evaluate three controlled variations:

1. **Sim A + Emb A**  
2. **Sim A + Emb B**  
3. **Sim B + Emb A**

This gives **three rankings per heuristic**, for a total of 6 numeric rankings.

We intentionally avoid further rankings because this smaller design is:

- easier to interpret,  
- still isolates the effect of each modeling choice,  
- and avoids noisy interactions that are hard to explain.

### 3.2.3 LLM-Based Comparisons

To complement the numeric rankings, we also generate **three LLM rankings**:

- **LLM–1:** Mission-only assessment  
- **LLM–2:** With proxy-success anchors  
- **LLM–3:** With proxy anchors + OPS-derived context  

These help evaluate whether OPS rankings align with qualitative judgment.

---

## 3.3 What We Compare

Across the 6 numeric rankings and 3 LLM baselines, we analyze:

- **Stability of top / middle / bottom groups** (we will consistently sample top 15, middle 10 and bottom 5)
- **Movements caused by changing embeddings**  
- **Movements caused by changing similarity metrics**  
- **Movements caused by changing heuristics**  
- **Consensus funders** (appear consistently at the top) i.e. orgs that are ranked high are ranked high consistently across variations 
- **Disagreement cases** that reveal sensitivity to modeling choices

This design produces a clear, interpretable map of which modeling decisions matter most for IterLight’s outreach strategy.

---

# 4. The Nine Rankings (Baseline + Variants + LLM)

All nine rankings reuse the shared OPS pipeline introduced earlier:
- cleaned mission text (`mission_clean`)
- numeric grant ranges
- engineered geographic focus (`geo_focus`)
- transformer embeddings of missions
- UMAP + HDBSCAN structure
- proxy-success funders to identify the “success neighborhood”

Each ranking defines its own scoring function \(f_k(x)\) by combining these shared components differently.

We build:
- **6 numeric rankings** (two heuristics × two embeddings × two similarity metrics)
- **3 LLM-based rankings**

---

## 4.1 Ranking 1 (R1) — Cluster-Aware Expected Value  
### *(H1 + Sim A + Emb A — Baseline Implementation)*

**Heuristic H1:**

\[
f_{R1}(x) = p(x)\cdot \mu(x)
\]

where:

\[
p(x) = \text{sim\_Iter}(x)\cdot w_{\text{cluster}}(x)\cdot \text{soft\_fit}(x)\cdot w_{\text{geo}}(x).
\]

**Embedding (Emb A):** BGE-large  
**Similarity (Sim A):** cosine (normalized)

This is the fully implemented baseline ranking used for all comparisons.

---

## 4.2 Ranking 2 (R2) — Same Heuristic, New Embedding  
### *(H1 + Sim A + Emb B)*

Everything identical to Ranking 1 **except** the embedding model.

- Embedding replaced with `gte-large` (Emb B)  
- Same heuristic H1  
- Same cosine similarity  

Goal: measure sensitivity to representation choice.

---

## 4.3 Ranking 3 (R3) — Same Heuristic, New Similarity  
### *(H1 + Sim B + Emb A)*

- Embedding stays BGE-large  
- Similarity changed (e.g., dot-product or unnormalized cosine)  

Goal: measure sensitivity to the similarity function.

---

## 4.4 Ranking 4 (R4) — Alternative Heuristic  
### *(H2 + Sim A + Emb A)*

Heuristic H2 introduces risk adjustment or exploration:

\[
f_{R4}(x) = p(x)\cdot \mu(x) - \lambda \cdot r(x)
\]

where \(r(x)\) is local grant-size variability.

Goal: de-prioritize noisy or inconsistent funders.

---

## 4.5 Ranking 5 (R5) — New Heuristic + New Embedding  
### *(H2 + Sim A + Emb B)*

Same risk-aware heuristic as R4, but with Emb B (gte-large).

Goal: test joint sensitivity to heuristic + embedding.

---

## 4.6 Ranking 6 (R6) — New Heuristic + New Similarity  
### *(H2 + Sim B + Emb A)*

Alternative heuristic + alternative similarity metric.

Goal: see if a different functional form or metric dramatically reshapes the ranking landscape.

---

## 4.7 LLM Ranking 1 (LLM-1) — Data-Only Judgement

LLM receives: name, mission text, geography, grant sizes.

No proxy funders, no OPS signals.

Task:  
“Rank funders by best fit for IterLight’s mission.”

---

## 4.8 LLM Ranking 2 (LLM-2) — With Proxy Anchors

LLM receives:
- full funder data  
- list of ~10 known EdTech success funders

Task:  
Infer the “successful EdTech funder concept” and rank all funders by similarity to that concept.

---

## 4.9 LLM Ranking 3 (LLM-3) — With OPS Top-30 Labels

LLM receives:
- proxy funders  
- OPS top/middle/bottom 30  
- short explanations for why certain orgs are promising

Task:  
Produce a refined global ranking that balances:  
- similarity to proxies  
- similarity to OPS top-30  
- diversity across funder types  

---

# 5. Comparative Analysis Across Rankings

This section will:

- compute rank correlations across R1–R6  
- analyze stability of top-15 funders  
- identify consensus funders  
- visualize rank-shift heatmaps  
- compare numeric vs LLM disagreements  
- isolate why certain orgs move dramatically when embedding or heuristic changes  

Outputs include:
- Rank trajectories  
- Cluster-level summary statistics  
- Sensitivity diagnostics  

---

# 6. Transition to Supervised Modeling (Logistic Regression)

After outreach, we will have ~30 labeled outcomes  
(**interested** vs **not interested**).

Steps:

1. Construct a feature matrix \(X\) from OPS components.  
2. Fit a logistic model:

\[
\Pr(\text{success}\mid x) = \sigma(\beta^\top X_x).
\]

3. Produce a **supervised ranking** using predicted probabilities.  
4. Compare the supervised ranking against all 9 earlier rankings.  
5. Use the learned coefficients to refine OPS (feature reweighting, cluster-prior tuning, etc.).

This will create a **learning loop** in which:
- OPS gives initial rankings
- Outreach produces labels
- Labels produce a better model
- Model updates the next OPS cycle
