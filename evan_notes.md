# Outreach Priority Scoring (OPS) Framework

## 1 Our Objective

Our goal is to develop an **Outreach Priority Scoring (OPS)** function  
that ranks potential funders by their likelihood of providing meaningful support to IterLight.  
Each organization $x$ receives a continuous score $f(x)$ that guides **which funders we contact first**.

---

## 2 Our Data Sources

| Dataset | Description | Role in Pipeline |
|----------|--------------|------------------|
| **Proxy-Labeled Funder Set (10 orgs)** | Confirmed EdTech funders (NewSchools Venture Fund, AT&T Aspire, Schmidt Futures, etc.) with known missions, categories, and grant sizes. | Provides **positive anchors** — examples of “successful” funders used for calibration. |
| **Unlabeled Funder Corpus (~60 orgs)** | All remaining potential funders curated from mission statements and foundation records. | Used for **representation and unsupervised learning**. |
| **IterLight Outreach Log (future)** | Records of real outreach attempts and responses. | Becomes new labels for **supervised refinement** of the OPS model. |

---

## 3 The Learning Strategy

### 3.1 Representation Learning — *(foundation for unsupervised 3.2)*

- Encode each organization’s **mission statement**, **category**, **geography**, and **grant info** into a dense vector $z_x \in \mathbb{R}^d$.  
- Use models such as **Sentence-BERT** or **OpenAI text-embedding-3-large**.  
- Output: one embedding vector per org.
Note : The way in which we embed, represent, and preprocess our data should be aligned with the specific unsupervised model we choose.
Different clustering and dim reduction methods assume different distance metrics etc, so our encoding and normalization strategy may need to change accordingly.

**Purpose of the Rep Learning Above:**  
We want to transform the unstructured mission statement text, geography scope, grant info, etc into numeric representations that capture meaning.  
Once we have these embeddings, we will
1. Compare funders semantically (nearby vectors indicates similarity between funders).  
2. Quantify “thematic distance” between IterLight and each funder. (We can quantify how aligned each funder is with iterlight) 
3. Provide the base features for both unsupervised discovery and label-driven calibration.

---

### 3.2 Unsupervised Learning — *(structure discovery; no labels used)*

Using only the embeddings $z_x$ (text + structured features):

1. **Dimensionality reduction and clustering**  
   - **Goal:** uncover the underlying structure of the funder space — which organizations resemble each other thematically, and along what axes they differ.  
   - **Method:**  
     - Apply **PCA (Principal Component Analysis)** or **UMAP (Uniform Manifold Approximation and Projection)** to project the high-dimensional feature matrix $X$ into a lower-dimensional space $Z_k$.  
     - **PCA** provides interpretable linear components (directions of greatest variance), allowing us to inspect **explained-variance ratios** to see how much information each principal component captures.  
       - The **top PCs** reveal the dominant axes of variation (e.g., “STEM ↔ Arts,” “Youth ↔ Adult,” “Corporate ↔ Philanthropic”).  
       - **Loadings** on each PC indicate which original features (e.g., grant size, org type, DEI focus) drive those distinctions.  
     - **UMAP**, when used, preserves nonlinear local structure and neighborhood relationships — helpful for visualization and cluster formation even if its dimensions aren’t directly interpretable.  
   - After reduction, run **K-Means** or **HDBSCAN** on $Z_k$ to identify latent funder groups such as “K-12 STEM,” “Workforce Tech,” or “Youth Equity.”  
     - **Diagnostic measures for clustering quality:**  
       - **Silhouette score:** cohesion vs separation of clusters.  
       - **Davies–Bouldin index:** lower values = better separation.  
       - **Calinski–Harabasz index:** higher = better cluster definition.  
       - **Stability checks:** rerun clustering with small perturbations; compare with **Adjusted Rand Index (ARI)** or **Normalized Mutual Information (NMI)**.  
     - **Diagnostic measures for reduction quality:**  
       - **Explained-variance curve** (PCA) → choose $k$ by “elbow” or cumulative ≥ 80–95 %.  
       - **Trustworthiness** and **continuity** (UMAP) → quantify how well local neighborhoods are preserved.  
       - **Average kNN distance** → checks density consistency.

2. **With this we will compute local statistics for each organization** This is to empirically estimate the function values within the OPS socring fomulas. 
   Using neighborhoods in the reduced space (k-nearest neighbors or clusters), compute:  
   - **Capacity** $\mu(x)$: median or mean grant size of similar funders → typical scale of giving.  
   - **Risk** $r(x)$: variability (IQR or standard deviation) of grant sizes among those neighbors → how volatile their giving is.  
   - **Uncertainty** $u(x)$: how isolated the funder is in the embedding space (inverse local density or average kNN distance) → how atypical or niche its mission is.  

   These statistics are **fully data-driven** — no success labels required.  
   They summarize how each funder behaves *relative to its peers* in the discovered landscape.

3. **Interpretation and use**  
   - **PC directions** and **cluster compositions** reveal *which features most differentiate organizations.*  
     For example, if PC1 strongly correlates with grant size and PC2 with DEI focus, the primary structure of the landscape separates funders by **scale** and **equity orientation**.  
   - The reduced coordinates $Z_k$, clusters, and local statistics ($\mu(x)$, $r(x)$, $u(x)$) become new interpretable descriptors of each funder’s position and behavior in the ecosystem.  
   - These outputs will later feed into the label-driven calibration step to refine the **Outreach Priority Scoring (OPS)** function.

These quantities come **entirely from unsupervised analysis** — they describe what a funder *typically does* (how big, how variable, how typical) rather than how successful any outreach has been.

**Why this matters:**  
Unsupervised learning exposes the hidden geometry of the funder landscape — the main axes of thematic variation, the natural groupings of funders, and the typical behaviors within each region of that space.  
It gives IterLight a **map of similarity and diversity** across organizations, validated by internal diagnostics (variance explained, silhouette, trustworthiness), forming the backbone for later alignment scoring and prioritization.

---

### 3.3 Semi-Supervised / Proxy Label Integration — *(orientation using labels)*

Once the unsupervised structure is built, we introduce the **10 known EdTech funders** as labeled positives.  
These labeled organizations serve two purposes:  
1. They **go through the same unsupervised pipeline** (they already have embeddings, clusters, and local statistics just like every other funder).  
2. They provide **reference points** to orient the rest of the funder landscape and estimate the probability of success.

---

#### 1 Apply unsupervised descriptors to labeled funders

Each labeled funder already has:
- **Capacity** $\mu(x)$ — median or mean grant size of its nearest neighbors  
- **Risk** $r(x)$ — variability of those grant sizes  
- **Uncertainty** $u(x)$ — local density or isolation  
- **Thematic distance** $d_{\\text{thematic}}(x)$ — cosine distance to IterLight’s embedding  
- **Cluster ID** — which thematic group it belongs to  

So the labeled funders sit inside the *same feature space* as all other funders.  
We simply tag them with a binary label $y=1$ (confirmed EdTech supporters).  
All remaining funders are **unlabeled** ($y=?$), not 0.

| funder_id | μ(x) | r(x) | u(x) | d_thematic(x) | cluster_id | y |
|------------|------|------|------|----------------|-------------|---|
| NewSchools | 280000 | 40000 | 0.10 | 0.08 | 0 | 1 |
| AT&T Aspire | 300000 | 45000 | 0.09 | 0.07 | 0 | 1 |
| Schmidt Futures | 260000 | 38000 | 0.12 | 0.09 | 0 | 1 |
| Ford Foundation | 700000 |150000 | 0.25 | 0.60 | 3 | ? |
| RAVE Foundation | 120000 | 25000 | 0.22 | 0.30 | 2 | ? |

The labeled rows ($y=1$) anchor the concept of “successful EdTech funders,” while the unlabeled ones form the background population used for comparison.

---

#### 2 Estimate the probability of funding $p(x)$

Two equivalent approaches can be used:

- **Centroid similarity:**  
- **Positive–Unlabeled (PU) model:**  

In both cases, $p(x)$ is a **soft probability** reflecting *how similar a funder is to proven EdTech supporters*, not a hard classification.

---

#### 3 Calibrate the scale of expected amounts

We also use the labeled funders’ known **grant amounts** to anchor the magnitude of  
E[amount | x] ≈ μ(x).  
This converts relative “capacity” scores into realistic dollar ranges so that when we later compute $p(x),\mu(x)$, the values correspond to real funding potential rather than normalized units.

---

#### 4 Outputs of this stage

| Variable | Description | Derived From |
|-----------|--------------|--------------|
| $p(x)$ | Probability of IterLight funding | Similarity / PU model using labeled positives |
| $\mu(x)$ | Typical grant size | Cluster / neighborhood median (scaled by labeled funders) |
| $r(x)$ | Variability of giving | Local standard deviation of grant sizes |
| $u(x)$ | Uncertainty / exploration term | Local density in embedding space |

---

> The labeled data are used **only to orient the space**, not to force negatives.  
> They tell the model *where in the landscape successful funders live* and how that region relates to typical grant size and risk.

This produces the calibrated ingredients — $p(x)$, $\\mu(x)$, $r(x)$, $u(x)$ — that feed directly into the Outreach Priority Scoring formulas that follow.

---

## Putting It Together — Source of Each Term

| OPS Variable | Meaning | Derived From | Type of Learning |
|---------------|----------|---------------|------------------|
| $p(x)$ | Probability of funding success | Similarity / PU model using 10 labeled funders | **Semi-supervised (labels used)** |
| $\mu(x)$ | Expected grant amount | Median grant size of similar funders (cluster-based) | **Unsupervised** |
| $r(x)$ | Risk / variability of funding | Variance of grant sizes among similar funders | **Unsupervised** |
| $u(x)$ | Uncertainty / exploration term | Local density or distance from known funders | **Unsupervised** |

---

## OPS Models to Explore

We will prototype three progressively richer formulations of the Outreach Priority Scoring function.

### (a) Risk-Neutral Expected Value OPS
$$
f_{\text{EV}}(x) = p(x)\,\mu(x)
$$
**Goal:** maximize expected funding yield.  
**Uses:** $p(x)$ (semi-supervised) × $\mu(x)$ (unsupervised).

---

### (b) Risk-Adjusted Expected Value OPS
$$
f_{\text{RiskAdj}}(x) = p(x)\,\mu(x) - \lambda\,r(x)
$$
**Goal:** favor consistent, lower-variance funding.  
**Uses:** $p(x)$ (semi-supervised) + $r(x)$ (unsupervised).

---

### (c) Explore–Exploit OPS
$$
f_{\text{EE}}(x) = f_{\text{RiskAdj}}(x) + \beta\,u(x)
$$
**Goal:** balance exploiting high-probability funders and exploring uncertain but promising ones.  
**Uses:** all terms; $u(x)$ from unsupervised density.

---

## Outreach Phase Plan

### Step 1 — Scoring & Ranking
Compute OPS scores for all funders and rank them.  
Top N per category form the initial outreach batch. 

### Step 2 — Contact & Record
Track outreach date, response type, and qualitative notes.

### Step 3 — Measure Success
We define a proxy for success as showing willingness to engage, share materials, or explore alignment. If they are willing to move forward and continue conversation, we consider that success.

Formally,
$$
\text{Success Rate} = \frac{\#\{\text{positive responses}\}}{|C|}
$$

### Step 4 — Feedback Loop
Each outreach cycle adds new confirmed positives (and negatives).  
These new labels expand the training set, turning the model progressively more **supervised** over time.

---

## Expected Deliverables

| Phase | Deliverable |
|--------|--------------|
| Representation Learning | Embedding model + latent feature matrix |
| Unsupervised Analysis | Cluster map + capacity/risk/uncertainty stats |
| Proxy Calibration | Semi-supervised classifier estimating $p(x)$ |
| OPS Scoring | Implementation of EV, Risk-Adjusted, and Explore–Exploit functions |
| Outreach Dashboard | Ranked list + measured success rate |
| Iterative Updates | Re-trained OPS after each outreach wave |

---

## Summary

- **Unsupervised learning (3.1–3.2):** builds the *map* — embeddings, clusters, and local grant statistics (capacity, risk, uncertainty).  
- **Proxy-labeled learning (3.3):** points the *compass* — identifies where “success” lies by using the 10 EdTech funders as anchors to estimate $p(x)$.  
- **OPS functions:** combine both sets of signals into an interpretable score for outreach prioritization.  
- **Outreach loop:** turns future engagement outcomes into new labels, gradually shifting the model from semi-supervised to fully supervised.

The result is a **data-driven, interpretable, and continuously improving outreach engine** that ranks funders by both their similarity to proven EdTech supporters and their practical funding potential.
