<h1 align="center"><strong>PrimatePostOpGuard™</strong></h1>
<h3 align="center">Synthetic Data & 15-Day Time-Series for Post-Surgical Complication Prediction in NHPs (Notebook)</h3>

---

## Phase 1 — Introduction & Mission

### 🔹 Why NHPs? Translational & Ethical Significance
- **High Translational Relevance:** Complex physiological & immune responses mirror humans.  
- **Ethical Responsibility:** Cognitively advanced species require predictive strategies to **minimize pain and stress**.  
- **Complex Recovery Dynamics:** Rich longitudinal data captures **non-linear interactions** between vitals, behavior, and environment.  
- **Colony Health Management:** Early identification of high-risk NHPs supports **resource allocation** and **alignment with 3Rs principles**.

> Combines **scientific rigor** with **ethical refinement**, laying groundwork for translational research.

### 🔹 Mission & Scientific Goal
*PrimatePostOpGuard™* is designed to **predict and prevent post-operative complications** in **non-human primates (NHPs)** by combining:
- **Synthetic, biologically plausible veterinary datasets** (0–14 days post-op)  
- **Longitudinal recovery trajectory simulations** with stochastic rare-event exposure  
- **Advanced machine learning ensembles** (LightGBM, Gradient Boosting, VotingRegressors)  
- **Explainable AI:** SHAP values, Partial Dependence Plots, per-subject feature contributions  
- **Interactive dashboards** for real-time, vet-focused risk visualization  

**Key Objectives:**
- Derive **subject-level pseudo-labels** via **unsupervised learning** to guide **supervised prediction** of post-op outcomes  
- Identify **high-risk individuals** and **latent recovery clusters**  
- Provide **actionable, interpretable guidance** for individualized post-op care and colony-level management  
- Uphold **3Rs principles** and maximize **ethical data use**

### 🔹 Deliverables & Predictive Insights

| Output                            | Description                                                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| **ComplicationRiskScore (0–100)** | Continuous, calibrated risk probability for post-surgical complications                      |
| **PredictedComplicationType**     | Most likely complication (Infection / Delayed Healing / Organ Failure / Multi-factor)        |
| **ComplicationCategory**          | Risk stratification (Low / Medium / High / Critical)                                         |
| **RecoveryTimePrediction**        | Estimated days until full recovery from longitudinal trends                                  |
| **ComplicationSeverity**          | Continuous severity score for prioritization                                                 |
| **ActionRecommendation**          | Vet-guided interventions: monitoring, analgesia adjustment, immediate attention              |
| **TopContributingFeatures**       | SHAP/PDP-driven features explaining risk drivers                                             |
| **ConfidenceInterval**            | Bootstrapped uncertainty estimates                                                           |
| **VisualizationSuite**            | Dashboards: per-subject risk timelines, heatmaps, colony summaries, top-feature trajectories |

### 🔹 Analytical Horizon & Data Boundaries
- **Data Level:** Individual NHP surgical records (synthetic but biologically plausible)  
- **Temporal Window:** Post-op recovery 0–14 days  
- **Primary Species:** *Macaca mulatta*, *Macaca fascicularis* (scalable to others)  
- **Features Captured:** Vitals, PainScore, WBC, CRP, SPORI, Behavioral metrics, Environmental metadata, Surgical details, Rare-event indices  

> Supports **high-resolution temporal modeling**, **trajectory clustering**, and **risk stratification**.

### 🔹 Ethical Framework & Compliance
- **Fully Synthetic Data:** All experiments use simulated datasets; no live animals were involved or exposed to risk.  
- **Vet-Informed Modeling:** Stochastic simulations reflect **biological plausibility**  
- **Regulatory Alignment:** Project design is modeled after **AAALAC, IACUC, ACLAM** principles  
- **3Rs Integration:** Replacement via synthetic datasets, Reduction via targeted identification, Refinement via actionable dashboards  
- **Transparency:** Rare-event simulations and pseudo-label trajectories enable **robust evaluation and reproducibility**

### 🔹 Project Vision & Takeaways
*PrimatePostOpGuard™* establishes a **clinically interpretable, ethically responsible AI framework**:  
- **Actionable subject-level insights:** Risk scores, top contributing features, interventions  
- **Colony-level monitoring:** Recovery clusters, high-risk summaries, resource planning  
- **Explainable AI:** SHAP, PDPs, interactive dashboards for vets  
- **Scalable & reproducible framework:** Ready for multi-center NHP datasets and translational applications  

**Phase 1 Summary:**  
*Establishes the strategic foundation for ethically responsible, clinically interpretable AI modeling of NHP post-operative outcomes.*

---

## Phase 2 — Libraries & Tools

### Core Data Handling & Utilities
- **numpy** – Efficient numerical computations  
- **pandas** – Structured data management and transformation  
- **os, sys** – File system and environment handling  
- **joblib** – Efficient save/load of ML models  
- **tqdm** – Progress bars  
- **warnings** – Suppress unnecessary output  

> Ensures smooth handling of **subject-level and time-series datasets** with reproducibility.

### Visualization & Interactive Dashboards
- **matplotlib, seaborn** – Static plots  
- **plotly.express & plotly.graph_objects** – Interactive dashboards and trend visualization  
- **plotly.subplots** – Multi-plot layouts  
- **matplotlib.cm.get_cmap** – Custom color maps  

> Delivers **high-quality visualizations** for recovery trajectories, risk clusters, and trends over time.

### Machine Learning & Modeling
- **Supervised Learning:** RandomForest, GradientBoosting, XGB, LGBM, Voting models  
- **Regression / Risk Scoring:** RandomForestRegressor, GradientBoostingRegressor  
- **Preprocessing:** Scaling (Standard, MinMax, Robust), OneHot/Label Encoding  
- **Model Evaluation:** train_test_split, StratifiedKFold; metrics: accuracy, precision, recall, f1, ROC-AUC, MAE, MSE, R2  
- **Unsupervised Learning:** KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, PCA  
- **Probability Calibration:** CalibratedClassifierCV  
- **Imbalanced Learning:** SMOTE for rare classes  

> Supports **robust predictive modeling, ensemble learning, rare-event risk assessment, and scalable pipelines**.

### Explainability & Interpretability
- **shap** – Feature importance and per-subject contributions  
- **PartialDependenceDisplay** – Visualizes **individual feature effects**

> Provides **actionable, interpretable insights** for veterinary decision-making.

**Phase 2 Summary:**  
*Establishes a robust, reproducible ecosystem for data handling, predictive modeling, visualization, and explainable AI.*

---

## Phase 3 — Synthetic Dataset Design & Exploratory Analysis

### Phase 3.1 — Synthetic Dataset Design
This phase builds a **mechanistically grounded, high-fidelity synthetic dataset** to simulate **post-operative recovery trajectories** in non-human primates (NHPs).  
It enables **predictive modeling**, **risk stratification**, and **hypothesis generation**, while **minimizing animal use** and maintaining **3Rs compliance**.

#### 🔹 Overview
- **Species:** *Macaca mulatta*, *Macaca fascicularis*, *Callithrix jacchus*  
- **Purpose:** Model individualized recovery patterns, evaluate risk, and refine post-surgical veterinary strategies  

**Scientific Objectives:**
- Characterize individualized recovery dynamics and physiological responses  
- Quantify daily risk using the **Synthetic Post-Operative Risk Index (SPORI_daily)**  
- Examine effects of **surgical complexity**, **species physiology**, and **veterinary expertise**  
- Generate translational insights for **early complication detection** and **perioperative optimization**  

> ✅ Fully adheres to the **3Rs (Replacement, Reduction, Refinement)** framework — offering a **humane, reproducible alternative** to live animal experimentation.

#### 🔹 Dataset Architecture

**1️⃣ Demographics & Baseline Physiology**

| Feature             | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| **Species**         | *50% Macaca mulatta*, *40% Macaca fascicularis*, *10% Callithrix jacchus*   |
| **Sex**             | *50% Male* / *%50 Female*                                                   |
| **Age (years)**     | Rhesus 3–25, Cynomolgus 2–20, Marmoset 1–8                      |
| **Weight (kg)**     | Allometric mapping with stochastic variability                  |
| **Baseline Vitals** | HR, Temp, PainScore, WBC, Cortisol, CRP (±5% daily variability) |
| **Organ Function**  | Liver (ALT/AST), Kidney (Creatinine), species-adjusted ranges   |

**2️⃣ Surgical & Procedural Parameters**

| Feature                 | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| **Procedure Type**      | Orthopedic, Abdominal, Cardiothoracic, Neurosurgery    |
| **Surgery Complexity**  | 3–6 ordinal scale (technical demand & invasiveness)    |
| **Duration (min)**      | 120–170 (procedure-dependent normal distribution)      |
| **Anesthesia Protocol** | Multi-drug regimens modulating physiological responses |

**3️⃣ Veterinarian Expertise**

| Feature                | Description                                                     |
| ---------------------- | --------------------------------------------------------------- |
| **VetType**            | Resident (2–4 yrs), Board-Eligible (3–6 yrs), DACLAM (7–20 yrs) |
| **OperativeCareSkill** | 1–5 scale; influences recovery dynamics and biomarkers          |

**4️⃣ Longitudinal Physiological & Functional Measures**
- Daily tracking (15 days): HR, Temp, PainScore, Mobility, Appetite  
- Biomarkers: WBC, Cortisol (circadian-modulated), CRP, liver/kidney enzymes  
- Derived Metric: `SPORI_daily` (0–100) integrating all risk components  
- Dynamic Modulation: Driven by interventions, rare events, and cross-feature dependencies  
- Noise Structure: Multivariate correlated Gaussian noise with autoregressive jitter  

**5️⃣ Rare Adverse Events**
- Modeled based on **species susceptibility** and **procedure complexity**  
- Includes: Severe Infection, Organ Failure, Hemorrhage, Cardiac Arrest  
- Occur between **days 2–5**, possibly extending beyond day 15  
- Stored as per-subject and per-day event logs  
- Trigger **correlated physiological perturbations** adjusted by VetType and CareSkill  

#### 🔹 Feature Simulation Principles
1. Autoregressive, time-dependent trajectories (decay + drift)  
2. Nonlinear cross-feature dependencies  
3. Circadian cortisol variation modulated by pain/stress  
4. Veterinary skill-based recovery modulation  
5. Multivariate stochastic noise for inter-subject variability  
6. Species-specific stress sensitivity  
7. Anesthesia and medication influence  
8. Probabilistic rare-event integration  
9. Efficient **vectorized generation (~5,000 subjects × 15 days)**  

#### 🔹 Derived Indices — SPORI_daily
Composite index reflecting **physiological**, **biochemical**, **functional**, and **procedural** parameters.

| Component       | Weight                    |
| --------------- | ------------------------- |
| Heart Rate      | 20%                       |
| Pain Score      | 20%                       |
| WBC             | 15%                       |
| Cortisol        | 15%                       |
| CRP             | 10%                       |
| Liver Function  | 5%                        |
| Kidney Function | 5%                        |
| Mobility        | 10%                       |
| Appetite        | 10% *(inverse weighting)* |

- **Normalization:** Min–max per subject  
- **Risk Levels:** Low ≤33, Moderate 34–66, High >66  

#### 🔹 Trajectory Clustering
- KMeans clustering identifies pseudo-recovery phenotypes (early vs. late recovery)  
- Reveals **heterogeneous risk patterns** and **latent recovery subtypes**  
- Enables **hypothesis generation** for translational modeling  

#### 🔹 Outputs
| Output                              | Description                 |
| ----------------------------------- | --------------------------- |
| `PrimatePostOpGuard_Summary.csv`    | Subject-level summary data  |
| `PrimatePostOpGuard_TimeSeries.csv` | Day-level longitudinal data |

---

### Phase 3.2 — Exploratory Data Analysis (EDA) & Visualization
Provides a **visual and analytical overview** of simulated recovery, highlighting **physiological dynamics, surgical factors, expertise effects**, and **rare event detection**.

#### 🔹 0️⃣ Data Preparation & Overview
- Unified all datasets into a **high-fidelity analysis framework**  
- Categorized subjects by **SPORI risk levels** (Low / Moderate / High)  
- Standardized features for comparability  
- Selected key longitudinal markers: HR, PainScore, WBC, CRP, Temp, Cortisol, Mobility, Appetite  
- Maintains strict **3Rs compliance** and **AAALAC-aligned ethical design**  

#### 🔹 1️⃣ Recovery Dynamics Over Time
- Daily monitoring of physiological and biomarker recovery  
- Visualized **mean ± SD** trends to identify normal vs. deviating trajectories  
- Supports **early clinical alerts** for outlier recoveries  

#### 🔹 2️⃣ SPORI Trajectories with Risk Zones
- Longitudinal SPORI visualization (color-coded by risk category)  
- Enables **rapid identification** of critical periods and high-risk individuals  
- Promotes adaptive intervention scheduling  

#### 🔹 3️⃣ Baseline Characteristics by Species & Sex
- Comparative analysis of **age, weight, HR, Temp, WBC, CRP, Cortisol**  
- Reveals **species- and sex-specific** trends affecting post-op recovery  
- Informs **tailored perioperative strategies**  

#### 🔹 4️⃣ Surgical Procedure Impact
- SPORI trajectory comparisons across **procedure type, duration, complexity**  
- Highlights the **relationship between invasiveness and recovery quality**  
- Guides **evidence-based surgical refinement**  

#### 🔹 5️⃣ Experience & Patient Outcomes
- Correlates **VetType** (Resident → DACLAM) with recovery outcomes  
- Quantifies **impact of skill level** on SPORI trajectories  
- Supports training optimization and **ACLAM-compliant performance benchmarks**  

#### 🔹 6️⃣ Recovery Clustering & Rare Events
- Cluster-based **recovery phenotype discovery**  
- Integration of **rare adverse events** (infection, organ failure)  
- Enables **high-risk prioritization** and resource allocation per AAALAC/NIH standards  

#### 🔹 7️⃣ Lab Markers Across Risk Levels
- Stratified visualization of **CRP, WBC, liver enzymes** by SPORI class  
- Detects **early biomarker warning signals**  
- Strengthens **preventive intervention frameworks**  

**Phase 3 Summary:**  
*Delivers a comprehensive, interpretable analysis of simulated post-operative recovery in NHPs — integrating physiology, surgical parameters, veterinary expertise, and rare-event tracking to support ethical, evidence-based care refinement.*

---

## Phase 4 — Feature Engineering & Dataset Preprocessing

### Phase 4.1 — Feature Engineering & Rare Event Integration
Derives **quantitative, clinically interpretable features** from longitudinal post-operative NHP data, enabling **robust predictive modeling** and handling **rare events effectively**.

#### 🔹 1️⃣ Multi-Factor Health Indices
Composite indices integrating **physiological, surgical, and environmental factors**:

- **StressScore (0–10):** Quantifies peri- and post-operative stress from environment, handling, and procedures  
- **ImmuneIndex:** Weighted combination of WBC, CRP, and cortisol; reflects immune competence & inflammatory response  
- **InfectionRisk:** Combines immune status, surgery duration, stress, and rare adverse events → individualized infection likelihood  
- **HealingDelay:** Models recovery delays based on PainScore, StressScore, surgery duration, and rare complications  
- **SPORI_adapted:** Adaptive recovery index integrating HealingDelay, InfectionRisk, and rare events → normalized, actionable metric  

> Clinically interpretable indices highlight **drivers behind predicted complications** for veterinarians.

#### 🔹 2️⃣ Temporal Feature Engineering
Transforms longitudinal measurements to capture **dynamic recovery patterns**:

- **Delta features (`_delta`):** Day-to-day changes for detecting rapid deviations  
- **Slope features (`_slope`):** Differences from baseline indicating cumulative trends  
- **Rolling averages (`_roll3`):** 3-day moving averages reduce noise while retaining meaningful patterns  

> Enables **early detection of deterioration** and supports **timely veterinary interventions**.

#### 🔹 3️⃣ Feature Reduction & Collinearity Management
- Remove **highly collinear numeric features** (Pearson r > 0.9)  
- Preserves **model stability** and **interpretability**

#### 🔹 4️⃣ Robust Scaling
- Apply **RobustScaler** to numeric features → resilient to outliers, including rare events  
- Preserves meaningful clinical variation without distortion  

> Ensures **robust modeling** while maintaining **signal integrity** for veterinary use.

#### 🔹 5️⃣ Key Engineered Features & Clinical Significance

| Feature       | Clinical Relevance                                                                      |
| ------------- | --------------------------------------------------------------------------------------- |
| HealingDelay  | Combines pain, stress, procedural factors → estimates recovery delays                   |
| InfectionRisk | Integrates immunologic, surgical, stress determinants → individualized infection risk   |
| SPORI_adapted | Composite recovery index → accounts for rare events, supports monitoring & intervention |

**Phase 4.1 Summary:**  
*Produces rare-event aware, clinically interpretable, longitudinal features ready for robust predictive modeling.*

---

### Phase 4.2 — Categorical Feature Encoding
Transforms categorical variables into ML-ready numerical representations.

#### 🔹 1️⃣ Identification of Categorical Features
- **Species:** NHP species (*Macaca mulatta*, *Macaca fascicularis*, *Callithrix jacchus*)  
- **Sex:** Male / Female  
- **SurgicalProcedure:** Type of surgery performed  
- **VetType:** Veterinarian experience level  
- **RareAdverseEventType:** Rare complications observed  

> Captures **biologically and clinically meaningful differences** while preserving interpretability.

#### 🔹 2️⃣ Rare Event Handling
- Missing `RareAdverseEventType` → `"NoEvent"`  
- Extremely rare categories (<0.5%) → `"OtherEvent"`  

> Maintains integrity of **rare adverse event information** without introducing noise.

#### 🔹 3️⃣ One-Hot Encoding (OHE)
- Converts categorical features to **binary matrix**  
- Drops first category → avoids multicollinearity  
- Encoded features remain **interpretable per category**  

> Enables ML models to **leverage categorical distinctions** clearly.

#### 🔹 4️⃣ Integration with Main Dataset
- Drop original categorical columns  
- Concatenate encoded columns → preserves **index alignment**  
- Dataset is now **fully numerical and model-ready**  

**Phase 4.2 Summary:**  
*All categorical variables are encoded, rare-event aware, and fully integrated for interpretable machine learning.*

---

### Phase 4.3 — Scaling & Normalization
Standardizes numerical features for robust, rare-event-aware modeling.

#### 🔹 1️⃣ Identification of Numerical Features
- Detect all numeric columns: physiological measurements, lab values, engineered indices, synthetic features  

> Preserves **clinical and biological interpretability**.

#### 🔹 2️⃣ Near-Constant Feature Handling
- Exclude features with near-zero variance (std < 1e-6)  
- Focus scaling on **informative, variable features**  

> Prevents noise from destabilizing the model.

#### 🔹 3️⃣ Robust Scaling
- Apply **RobustScaler** to numeric features  
- Resistant to **outliers and rare events**  
- Maintains **relative differences** for clinically significant events  

> Ensures comparability, improves **model convergence**, and maintains **risk-aware predictions**.

#### 🔹 4️⃣ Summary & Dataset Readiness
- All numeric features standardized, rare events preserved  
- Dataset ready for **supervised and unsupervised predictive tasks**  

**Phase 4.3 Summary:**  
*Numerical features are robustly scaled, preserving rare-event signals for stable, interpretable ML.*

---

### Phase 4.4 — Final NaN & Inf Imputation + Unsupervised Dataset Save
Ensures a fully clean, rare-event-aware dataset for unsupervised modeling.

#### 🔹 1️⃣ Dataset Initialization
- Start from fully processed DataFrame  
- Create dedicated copy for unsupervised modeling → prevents contamination of supervised dataset  

#### 🔹 2️⃣ Robust NaN & Inf Handling
- Replace `np.inf` / `-np.inf` → `NaN`  
- Numeric columns: fill `NaN` → `0`  
- Object/categorical columns: fill `NaN` → `"Unknown"`  
- Final check for missing/infinite values  

> Guarantees **dataset integrity** and **rare-event preservation**.

#### 🔹 3️⃣ Dataset Integrity Verification
- Confirm final dataset shape and subject count (`SubjectID`)  
- Review rare-event distribution to ensure representation  
- Ensures reliability for **unsupervised analysis** without skewing rare-event signals  

#### 🔹 4️⃣ Saving Unsupervised-Ready Dataset
- Save as `data/long_df_unsupervised_ready.csv`  
- Fully preprocessed, rare-event aware, ready for **clustering, PCA, or other unsupervised tasks**  

**Phase 4.4 Summary:**  
*Produces a fully cleaned, rare-event-aware dataset prepared for robust unsupervised learning.*

---

## Phase 5 — Predictive Modeling & Ensemble Learning

### Phase 5.1 — Unsupervised Learning: Biologically Plausible Pseudo-Labels
Generates **subject-level pseudo-labels** for supervised and unsupervised modeling of NHP post-operative outcomes. Prioritizes **biological plausibility, clinical relevance, and ethical principles** under 3Rs guidelines.

#### 🔹 1️⃣ Subject-Level Aggregation & Vet-Relevance
- Aggregate numeric and temporal features per subject: *HR, PainScore, WBC, CRP, SPORI, HealingDelay, InfectionRisk*  
- Summary stats: *Mean, SD, Min, Max*; trends: *Slope over post-op days*  
- Retain static features: *Species, Sex, Age, Weight, Surgery Type, VetType, RareAdverseEventEver*  
- Compute **RareEventFrac** for exposure to uncommon complications  

> Respects **biological variability** and highlights **clinically meaningful patterns**.

#### 🔹 2️⃣ Stratified Sampling for Rare Events
- Max 3,000 subjects sampled  
- Rare-event oversampling (30%) ensures robust representation  
- Non-rare subjects fill remaining sample  

> Captures **high-risk outcomes** without inflating prevalence.

#### 🔹 3️⃣ Dimensionality Reduction & Clustering
- StandardScaler → uniform feature ranges  
- PCA → top 12 components for latent recovery visualization  
- MiniBatchKMeans → 2–3 biologically relevant recovery subtypes  
- Output: *pseudo_cluster* labels  

> Preserves **clinically interpretable recovery patterns**.

#### 🔹 4️⃣ Pseudo-Label Generation
- *RecoveryTime_days* → HealingDelay + SPORI + RareEventFrac  
- *ComplicationSeverity* → SPORI + PainScore slope + rare events  
- *InfectionProb* → adjusted for rare-event exposure  
- *ComplicationType* → categorical likelihood & severity  

> Produces **synthetic yet realistic outcomes** without live animal use.

#### 🔹 5️⃣ Mapping to Longitudinal Dataset
- Map clusters to simplified recovery patterns: Fast Recovery (low-risk) vs Slow/High-Risk  
- Supports **intuitive visualization and clinical interpretation**

#### 🔹 6️⃣ Recovery Trajectory Visualization
- Scatter plot: *SPORI vs HealingDelay*, colored by recovery patterns  
- Highlights fast vs slow/high-risk recoveries  

#### 🔹 7️⃣ Saving Models & Data
- PCA, MiniBatchKMeans, Scaler saved  
- Subject-level pseudo-labels → `pseudo_subject_labels.csv`  
- Longitudinal dataset → `long_df_clusters_subject_level.csv`  

**Phase 5.1 Summary:**  
*Generates biologically plausible, ethically grounded pseudo-labels and recovery clusters.*

---

### Phase 5.2 — Supervised Learning
Predicts post-operative complications using **subject-level features**, preserving **rare-event signals** for ethical, actionable predictions.

#### 🔹 1️⃣ Dataset Loading & Preparation
- Load `long_df_clusters_subject_level.csv`  
- Target: *ComplicationType*  
- Exclude pseudo-label source features → prevent **data leakage**  
- Replace NaN / infinite values  

#### 🔹 2️⃣ Categorical Feature Encoding
- One-hot encode object-type columns (`drop_first=True`)  
- Remove near-constant features → reduces noise and overfitting  

#### 🔹 3️⃣ Train/Test Split
- 80/20 split at **subject level** to preserve independence  

#### 🔹 4️⃣ Model Training: LightGBM
- Classifier: LightGBM  
- Parameters: `max_depth=4`, `learning_rate=0.03`, `n_estimators=300`, `class_weight='balanced'`, `min_child_samples=30`  
- Row & column subsampling → improves robustness  

> Captures **non-linear interactions** while preserving rare-event signal.

#### 🔹 5️⃣ Predictions & Risk Scores
- Predicted probabilities normalized  
- Risk Scores scaled 0–100 for vet interpretability  
- Predicted class → highest probability  

#### 🔹 6️⃣ Model Evaluation
- Classification report: Accuracy, Precision, Recall, F1  
- Macro ROC-AUC  
- Bootstrap 95% CI for rare classes  

#### 🔹 7️⃣ Sample Risk Scores
- Top 5 predicted scores for **high-risk prioritization**  

#### 🔹 8️⃣ Saving the Model
- Saved as `models/lgb_complication_type_stable_vet_safe.pkl`  

**Phase 5.2 Summary:**  
*Interpretable supervised learning framework for predicting post-op complications.*

---

### Phase 5.3 — Supervised Training: Classification & Regression
Multi-target supervised learning for NHP post-op outcomes.

#### 🔹 1️⃣ Dataset Loading & Preprocessing
- Targets:  
  - Classification: *ComplicationType*, *Infection*  
  - Regression: *RecoveryTime_days*, *ComplicationSeverity*  
- Drop identifiers and target columns  
- Encode categorical/object columns using `LabelEncoder`  

#### 🔹 2️⃣ Classifier Training
- LightGBM with stratified K-Fold CV (n_splits=3)  
- Anti-overfit hyperparameters: `max_depth=4`, `num_leaves=15`, `subsample=0.8`, `colsample_bytree=0.8`, `class_weight='balanced'`  
- Early stopping, fold-wise metrics saved  

#### 🔹 3️⃣ Regressor Training
- LightGBM regressors with stratified folds via binned targets  
- Monitors fold-wise MSE, early stopping applied  

#### 🔹 4️⃣ Model Execution
- Train all targets independently with silent logging  
- Ensures reproducibility  

#### 🔹 5️⃣ Categorical Encoder Preservation
- LabelEncoders saved → `models/categorical_encoders_turbo_fast_silent.joblib`  

**Phase 5.3 Summary:**  
*Produces stable, reproducible classifiers & regressors for NHP post-op outcomes.*

---

### Phase 5.4 — Ensemble Training & Evaluation
Builds **robust, rare-event aware ensembles** for clinically interpretable predictions.

#### 🔹 1️⃣ Dataset Loading & Feature Preparation
- Load clustered, pseudo-labeled dataset  
- Targets: Classification (*ComplicationType*), Regression (*RecoveryTime_days*, *ComplicationSeverity*)  
- One-hot encode categorical features, fill missing → `"Unknown"`  

#### 🔹 2️⃣ Classification Ensemble
- Stratified train/test split  
- SMOTE oversampling → mitigate rare-event imbalance  
- Ensemble: two LightGBM classifiers (soft voting)  
- Metrics: Accuracy, F1-score, ROC-AUC  
- Top 5% high-risk flagged for vet review  

#### 🔹 3️⃣ Regression Ensemble
- Ensemble of two LightGBM regressors (VotingRegressor)  
- Metrics: MAE, RMSE, R²  
- Robust to outliers and rare events  

> Combine ensemble predictions for **targeted monitoring**, **early intervention**, and **risk prioritization**.

**Phase 5.4 Summary:**  
*Robust, rare-event aware ensemble models ensure stable, clinically interpretable post-operative predictions.*

---

## Phase 6 — Model Evaluation, Explainability & Decision Framework

### Phase 6.1 — Model Evaluation
Rigorous evaluation of ensemble models across **classification**, **regression**, and **pseudo-clustering**, designed for **clinical interpretability and decision support**.

#### 🔹 1️⃣ Dataset Loading & Feature Preparation
- Dataset: `long_df_clusters_subject_level.csv`  
- Prediction targets:  
  - **Classification** → `ComplicationType`  
  - **Regression** → `RecoveryTime_days`, `ComplicationSeverity`  
- Excluded columns: `SubjectID`, `RecoveryPatternSimplified`  
- Categorical encoding protocol:  
  - Fill missing values → `"Unknown"`  
  - Apply consistent one-hot encoding  

> Ensures **feature consistency** and valid downstream evaluation.

#### 🔹 2️⃣ Classification Metrics — `ComplicationType`
- Outputs: predicted **classes + probabilities**  
- Metrics used:  
  - Accuracy  
  - Precision (macro)  
  - Recall (macro)  
  - F1-score (macro)  
  - ROC-AUC (multi-class, ovo)  

> Evaluates **discrimination performance** and **resistance to overfitting**.

#### 🔹 3️⃣ Regression Metrics
- Continuous outcomes:  
  - `RecoveryTime_days` → recovery duration prediction  
  - `ComplicationSeverity` → severity estimation  
- Metrics used: **MAE, RMSE, R²**  

> Supports **quantitative post-op risk assessment** and **early intervention planning**.

#### 🔹 4️⃣ Clustering Evaluation — Subject-Level Recovery Phenotypes
- Metric: **Silhouette Score** on numeric features  
- Purpose: validate biologically plausible pseudo-label separation  

> Confirms **interpretability of recovery patterns** across subjects.

> Final Pipeline → PCA + MiniBatchKMeans chosen over UMAP + HDBSCAN for smoother & biologically plausible patterns suitable for pseudo-label generation.  
> Combine outputs to guide **targeted post-op care, risk triage, and resource allocation**.

---

### Phase 6.2 — Model Explainability
Interpreting ensemble LightGBM predictions for NHP post-operative complications.

#### 🔹 1️⃣ Dataset & Feature Preparation
- Load clustered dataset  
- Drop targets: *ComplicationType*, *RecoveryTime_days*, *ComplicationSeverity*  
- Drop identifiers: *SubjectID*, *RecoveryPatternSimplified*  
- Align categorical features using saved **LabelEncoders**  
- One-hot encode remaining categorical features  

> Maintains integrity and valid explainability analysis.

#### 🔹 2️⃣ Load Ensemble Classifier & Extract Booster
- Load ensemble classifier: `ComplicationType_ensemble_classifier.pkl`  
- Extract **first LightGBM estimator** for SHAP  
- Align features to booster internal names  

#### 🔹 3️⃣ SHAP Computation
- `shap.TreeExplainer` on aligned booster  
- Sample ≤1000 observations for speed  
- Normalize multiclass outputs  
- Obtain **global feature importance** and **per-subject contributions**  

#### 🔹 4️⃣ Global SHAP Plots
- Summary bar plot: mean absolute SHAP values  
- Summary scatter plot: feature value vs contribution  

#### 🔹 5️⃣ Per-Subject Explainability
- Top 5 high-risk subjects  
- Individual SHAP waterfall plots (*SPORI, HealingDelay, PainScore*)  

#### 🔹 6️⃣ Partial Dependence Plots (PDPs)
- Top 5 features by SHAP importance  
- Visualize **average response curves**  

**Phase 6.2 Summary:**  
*Transparent, vet-focused explainability using SHAP and PDPs for population-level and individual NHP insights.*

---

### Phase 6.3 — Biologically Plausible Decision Framework
Translates ensemble predictions into actionable post-operative risk scores.

#### 🔹 1️⃣ Dataset & Ensemble Loading
- Load `long_df_clusters_subject_level.csv`  
- Load **ComplicationType ensemble classifier** + categorical encoders  
- Encode categorical features consistently  

#### 🔹 2️⃣ Feature Alignment & Prediction
- Align features to LightGBM estimator  
- Predict **per-subject complication probabilities**  
- Assign **PredictedRisk** (max class probability)  

#### 🔹 3️⃣ Actionable Risk Categories
| PredictedRisk | Category |
|---------------|----------|
| ≤0.3          | Low |
| 0.3–0.6       | Medium |
| 0.6–0.8       | High |
| >0.8          | Critical |

> Prioritize high-risk NHPs for veterinary attention.

#### 🔹 4️⃣ Top Contributing Features
- Extract top 3 global features influencing predictions  
- Merge with per-subject data → vet-focused interpretation  

#### 🔹 5️⃣ Colony-Level Summary
- Aggregate by `pseudo_cluster`  
- Compute:
  - **AvgRisk:** mean predicted risk  
  - **HighRiskPct:** fraction of NHPs in High/Critical  

> Enables population-level monitoring and resource allocation.

#### 🔹 6️⃣ Outputs & Pipeline
- Per-NHP risk & top features → `data/vet_risk_scores_top_features.csv`  
- Colony-level summary → `data/vet_colony_summary.csv`  
- Unified pipeline → `models/primate_postop_pipeline.pkl`  

**Phase 6.3 Summary:**  
*Delivers actionable, biologically plausible post-op risk framework with NHP-level scores, top feature insights, and colony-level monitoring.*

---

# Phase 7 — Core Prediction Function: Static Multi-Species Post-Operative Dashboard & Risk Visualization — PrimatePostOpGuard™

Phase 7 delivers a **static, clinically-informed framework** to predict **post-operative complications in non-human primates (NHPs)**, visualize **risk evolution over hours and days**, and generate **clinically actionable recommendations**. It supports **multiple NHP species** and **cohort-level insights**, with a **clinically-informed interpretation** of risk.

## 🔹 1️⃣ Models & Feature Setup

- Load ensemble classifier for post-op complication prediction + associated categorical encoders  
- Extract primary LightGBM model for SHAP interpretability and risk computation  
- Define human-readable feature labels & species-specific notes for dashboard visualization  
- Align input features with model’s expected format  

## 🔹 2️⃣ Core Prediction Workflow

**Function:** `predictNHPComplicationDashboardTime_MultiSpecies_Educational_Static`  

**Inputs:**  
- List of NHP records (dicts) including species, weight, age, anesthesia details, clinical parameters  
- Multi-subject simulation (`n_subjects_per_species`) for cohort-level predictions  

**Workflow Steps:**  
1. Filter records per species → prepare species-specific datasets  
2. Encode categorical features using pre-trained encoders  
3. Align features to LightGBM format  
4. Predict **risk probabilities** & most likely complication per subject  
5. Compute **SHAP values** for explainability  
6. Extract **top 8 contributing features** per subject  
7. Compute **95% risk confidence intervals**  
8. Assign **risk levels** (Moderate/Low → High → Critical) with recommended actions  
9. Generate **cohort-level predictions**  

> Supports clinically-informed, interpretable predictions at **subject & colony levels**.

## 🔹 3️⃣ Dashboard Components (Static)

- 📊 **SHAP Bar Chart** — Top feature contributions per NHP (green ↓ / red ↑)  
- 📊 **Per-Species SHAP Heatmaps** — Multi-subject risk contributions per complication type  
- 📊 **Multi-Subject Risk Summary** — Cohort-level predicted risk bar chart + recommended interventions  
- 📊 **Risk Over Time Plot** — Hourly post-op risk per subject  
- 📊 **Risk Over Days Plot** — Daily post-op risk projection (0–14 days) highlighting trends & interventions  

> Provides **static, visually interpretable, actionable insights**.

## 🔹 4️⃣ Outputs & Interpretation

| Key | Description |
|-----|------------|
| RiskScore | Predicted risk % for primary complication per NHP |
| RiskLevel | Risk category: Moderate/Low, High, Critical |
| ActionRecommendation | Clinically-guided intervention based on predicted risk |
| TopFeatures | Top 8 SHAP feature contributions per subject |
| ConfidenceInterval | 95% CI for predicted risk |
| SHAPBar | Static bar chart visualizing feature impacts |
| SHAPHeatmaps | Per-species, multi-subject heatmaps per complication type |
| MultiNHPRiskSummary | Cohort-level risk bar chart per species |
| RiskOverTime | Hourly post-op risk visualization per NHP |
| RiskOverDays | Daily post-op risk projection per NHP |

## 🔹 5️⃣ Usage Example

**Input Example:**  
- Species: Rhesus / Cynomolgus / Marmoset  
- Weight: 6.5 kg / 5.8 kg / 0.45 kg  
- Age: 120 / 90 / 200 days  
- Anesthesia Duration: 45 / 50 / 30 min  
- Early Recovery Temp: 37.8 / 38.1 / 38.5 °C  
- Surgery Type: Minor  

**Example Outputs:**  
- RiskScore: 24.5%  
- RiskLevel: Moderate  
- ActionRecommendation: Monitor vitals closely; consider additional analgesia  
- TopFeatures: EarlyRecoveryTemp, WBC, AnesthesiaDuration, HeartRate, PainScore, CRP, Cortisol, Mobility  

**Static Visualizations:**  
- SHAP Bar Chart  
- Per-Species SHAP Heatmaps  
- Multi-Subject Risk Summary  
- Risk Over Time Plot  
- Risk Over Days Projection  

> Illustrates **individual & colony-level actionable insights** across species.

## 🔹 Phase 7 Summary

*Phase 7 delivers a static, interpretable, clinically actionable multi-species dashboard for predicting NHP post-op complications, supporting both subject-level & colony-level decision-making with SHAP interpretability, risk visualization, and actionable recommendations.*

---

# PrimatePostOpGuard™ — Project Conclusion

## 🔹 Key Achievements
- Developed **ethically grounded AI framework** for NHP post-op complication prediction  
- Created **subject-level pseudo-labels** for robust modeling without extra animal use  
- Built **stable, rare-event aware supervised models** (classification & regression)  
- Implemented **ensemble methods** for predictive stability & clinical interpretability  
- Developed **SHAP & PDP explainability** for population- & individual-level insights  
- Delivered **interactive dashboard** for real-time risk visualization & recommendations  

## 🔹 Ethical & Scientific Impact
- Prioritized **3Rs principles** (Replacement, Reduction, Refinement)  
- Preserved **biological variability** and **clinically relevant recovery patterns**  
- Enabled **clinically contextualized triage** and **colony-level risk management**

## 🔹 Clinical & Translational Value
- Early detection of **high-risk post-op complications**  
- Quantitative **risk scores** + **feature-level explanations**  
- Supports **evidence-based refinement of recovery protocols**

## 🔹 Future Directions
- Expand to **multi-center NHP datasets**  
- Integrate **continuous learning** for adaptive predictions  
- Explore **translational applications** for **human post-surgical recovery modeling**  
- Enhance interactive dashboard with predictive simulations & intervention modeling  

## ✅ Summary & Impact
*PrimatePostOpGuard™ delivers a comprehensive, reproducible, and ethically responsible AI framework for NHP post-operative care, combining predictive power, interpretability, and actionable insights at both subject- and colony-levels while adhering to 3Rs principles.*

---

## ⚠️ Data & Model Disclaimer

All data in this repository is **fully synthetic** and generated for **research and demonstration purposes**.  
Metrics, predictions, and risk scores are **illustrative only** and **should not be used for real veterinary decision-making**.  

> The models reflect patterns in pseudo-labels and synthetic outcomes designed to explore interpretable AI in nonhuman primate post-operative care.  

> ⚠️ **Note:** This repository reflects my current methodology and development approach. While some aspects of the code or modeling workflow could be refined, the project is released under the **MIT License**, allowing anyone to explore, adapt, or improve it for research and educational purposes.

### Important Notes

- **Mixed-species datasets:**  
  - 50% Macaca mulatta  
  - 40% Macaca fascicularis  
  - 10% Callithrix jacchus  

  Each species has distinct physiological baselines (HR, Temp, WBC, etc.), increasing inter-species variability. Low or negative silhouette scores in unsupervised analyses are expected in this mixed-species scenario.

- **Unsupervised approaches explored:**  
  1. **UMAP + HDBSCAN** — useful for visualization and exploratory analysis; produced low/negative silhouette scores due to high inter-species variability.  
  2. **PCA + MiniBatchKMeans** — generated biologically plausible pseudo-labels for **supervised regression and classification models**. Despite low silhouette scores, pseudo-labels captured interpretable recovery patterns.

- **Real-world implications:**  
  - Supervised models would be preferred if true outcomes were available.  
  - Current workflow demonstrates interpretable AI techniques without direct access to primates.  
  - Future work would train models directly on ground-truth data once available.

---

## Acknowledgments and References

**PrimatePostOpGuard™** is built upon foundational work in **nonhuman primate medicine**, **laboratory animal welfare**, and **interpretable artificial intelligence**. The project upholds the *Refinement* principle of the 3Rs, aiming to ethically enhance post-operative monitoring and predictive modeling for nonhuman primates using exclusively **synthetic, regulatory-compliant data**.

The author gratefully acknowledges:

* **The Laboratory Nonhuman Primate, Second Edition (2018)** for providing comprehensive insights into primate biology, physiology, and welfare considerations essential to biomedical research.
* **AAALAC International**, **IACUC**, and **USDA** for defining the ethical and oversight frameworks that inspired every phase of this project’s design.
* The **ACLAM (American College of Laboratory Animal Medicine)** community for advancing humane, evidence-based approaches in laboratory animal medicine and refinement.
* The broader **3Rs (Replacement, Reduction, Refinement)** community for its enduring commitment to ethical innovation and scientific integrity.
* Developers of key open-source machine learning libraries—**scikit-learn**, **XGBoost**, **LightGBM**, **SHAP**, and **Plotly**—whose tools make transparent and reproducible AI research possible.

---

### Image Credits

* **Rhesus macaque** — Courtesy of *California National Primate Research Center (CNPRC)*.
* **Cynomolgus macaque** — Courtesy of the *NC3Rs Macaque Website*.
* **Common marmoset** — Courtesy of *Dr. Amber Hoggatt*.

All images are used with proper attribution for educational, illustrative, and non-commercial purposes in alignment with ethical communication standards.

---

### References

1. Fox, J.G., Anderson, L.C., Otto, G., Pritchett‑Corning, K., & Whary, M.T. (Eds.). (2015). *Laboratory Animal Medicine* (3rd ed.). Academic Press.  
2. Bennett, B.T., Schapiro, S.J., & Reinhardt, V. (Eds.). (2018). *The Laboratory Nonhuman Primate* (2nd ed.). CRC Press.  
3. National Research Council. (2011). *Guide for the Care and Use of Laboratory Animals* (8th ed.). National Academies Press.  

*Note:* The referenced materials serve as **ethical and methodological context** rather than direct data or code sources. **PrimatePostOpGuard™** is an independent, **synthetic-data project** developed exclusively for **research, educational, and refinement purposes**.

---

## GitHub Repositories for Previous Work

- [PostOpPainGuard™](https://github.com/Ibrahim-El-Khouli/PostOpPainGuard.git)
- [LECI - Lab Environmental Comfort Index](https://github.com/Ibrahim-El-Khouli/LECI-Lab-Environmental-Comfort-Index.git)  
- [Lab Animal Health Risk Prediction](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Health-Risk-Prediction.git)  
- [Lab Animal Growth Prediction](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Growth-Prediction.git)  

---

## License

**PrimatePostOpGuard™** is released under the **MIT License** — free for academic, research, and non-commercial use.