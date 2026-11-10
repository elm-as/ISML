<div align="center">

# ISML — Impact of Social Media on Life
### Tri-Cohort Multi-Population Behavioral, Chronobiological & Adolescent Depression Architecture (N = 6,532 Across 113 Nations)

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.7-yellow?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blue?style=flat-square)
![Cohort](https://img.shields.io/badge/Cohort-6%2C532%20Subjects-purple?style=flat-square)
![Charter](https://img.shields.io/badge/Charter-ELMAS.md%20Compliant-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

<br/>

**"From Data to Systems"**  
A unified multi-cohort behavioral data science monograph investigating the non-linear inflection points between algorithmic digital platform consumption, nocturnal circadian disruption, psychometric stress, cross-national interpersonal conflicts, and adolescent clinical depression risk across **6,532 subjects spanning 113 countries**.

</div>

<br/>

<table align="center">
<tr>
<td align="center"><h2>6,532</h2><sub>total subjects</sub></td>
<td align="center"><h2>113</h2><sub>nations represented</sub></td>
<td align="center"><h2>98.69%</h2><sub>cohort 1 impact acc</sub></td>
<td align="center"><h2>0.9962</h2><sub>cohort 3 depression auc</sub></td>
<td align="center"><h2>93.55%</h2><sub>clinical sensitivity</sub></td>
<td align="center"><h2>&minus;0.6668</h2><sub>global addiction vs mental r</sub></td>
</tr>
</table>

---

## Table of Contents

1. [Epidemiological Context & Tri-Cohort Design](#epidemiological-context--tri-cohort-design)
2. [Tri-Cohort System Architecture](#tri-cohort-system-architecture)
3. [Cohort 1: Primary Behavioral & Stress System (N = 4,500)](#cohort-1-primary-behavioral--stress-system-n--4500)
4. [Cohort 2: Global Cross-National Addiction Atlas (N = 832, 113 Nations)](#cohort-2-global-cross-national-addiction-atlas-n--832-113-nations)
5. [Cohort 3: Adolescent Clinical Depression & Protective Factors (N = 1,200)](#cohort-3-adolescent-clinical-depression--protective-factors-n--1200)
6. [Visual Evidence & High-Resolution Telemetry](#visual-evidence--high-resolution-telemetry)
7. [Explainability & Attribution Analysis](#explainability--attribution-analysis)
8. [Interactive Analytics Dashboard](#interactive-analytics-dashboard)
9. [Single-Student Triage CLI](#single-student-triage-cli)
10. [Repository Structure](#repository-structure)
11. [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
12. [The ElmasCore Charter Compliance](#the-elmascore-charter-compliance)
13. [Citation & Attribution](#citation--attribution)

---

## Epidemiological Context & Tri-Cohort Design

The pervasive presence of algorithmically curated feeds (TikTok, Instagram, YouTube) poses complex questions regarding digital hygiene, mental health erosion, and academic performance. To eliminate geographic, demographic, and cohort bias, ISML unifies three distinct populations into a cohesive diagnostic architecture:

```mermaid
flowchart TD
    C1["Cohort 1: Primary Students (N = 4,500) • Ages 15-26"] --> FE1["Chronobiological Feature Engineering"]
    C2["Cohort 2: Global Students (N = 832, 113 Nations)"] --> FE2["Cross-National Friction & Well-Being Engine"]
    C3["Cohort 3: Teenagers (N = 1,200) • Ages 13-19"] --> FE3["Circadian Debt & Protective Activity Ratios"]

    FE1 --> T1["Task 1: 3-Class Overall Impact Classifier (98.69% Acc)"]
    FE1 --> T2["Task 2: Continuous Stress Regressor (R2 = 0.6332)"]
    FE2 --> T3["Task 3: Cross-National Addiction Atlas (r = -0.6668)"]
    FE3 --> T4["Task 4: Adolescent Depression Classifier (AUC = 0.9962)"]

    T1 --> DASH["Interactive Plotly HTML5 Tri-Cohort Dashboard"]
    T2 --> DASH
    T3 --> DASH
    T4 --> DASH

    style C1 fill:#EEF2FF,stroke:#6366F1,color:#1e1b4b
    style C2 fill:#FEF3C7,stroke:#F59E0B,color:#78350f
    style C3 fill:#FEE2E2,stroke:#EF4444,color:#7f1d1d
    style DASH fill:#E0F2FE,stroke:#0284C7,color:#0c4a6e
```

---

## Cohort 1: Primary Behavioral & Stress System (N = 4,500)

### Empirical Formulations
1. **Screen-to-Sleep Ratio**:
   $$\text{SSR}_i = \frac{\text{Daily\_Usage\_Hours}_i}{\text{Sleep\_Duration\_Hours}_i + \epsilon}$$
2. **Weekly Cumulative Screen Load**:
   $$\text{WSL}_i = 5 \times \text{Daily\_Usage\_Hours}_i + 2 \times (\text{Daily\_Usage\_Hours}_i + \text{Weekend\_Extra\_Hours}_i)$$
3. **Chronobiological Sleep Debt**:
   $$\text{Sleep\_Debt}_i = \max(0, 8.0 - \text{Sleep\_Duration\_Hours}_i)$$
4. **Nocturnal Friction**:
   $$\text{Nocturnal\_Friction}_i = \mathbb{I}(\text{Late\_Night}_i = \text{True}) \times (5 - \text{Sleep\_Quality}_i)$$

### 5-Fold Stratified Evaluation Benchmarks
- **Overall Accuracy**: $98.69\%$
- **Balanced Accuracy**: $97.28\%$
- **Macro F1-Score**: $0.9654$
- **Rare Negative Impact F1**: $0.9461$
- **Continuous Stress $R^2$**: $0.6332$ ($RMSE = 4.2613$)
- **Spearman Rank Correlation**: $\rho = 0.7500$ ($p < 10^{-15}$)

---

## Cohort 2: Global Cross-National Addiction Atlas (N = 832, 113 Nations)

Collected across 113 distinct countries, Cohort 2 investigates how social media addiction and interpersonal conflicts impact mental well-being and academic stability internationally:

| Cross-National Metric | Empirical Value | Epidemiological Meaning |
|:---|:---:|:---|
| **Addiction vs. Mental Health Correlation** | $r = -0.6668$ | Extreme inverse relationship; severe addiction directly erodes mental health |
| **Addiction vs. Interpersonal Conflicts** | $r = +0.6642$ | Digital addiction directly escalates domestic and peer arguments |
| **Sleep Duration vs. Mental Health** | $r = +0.5064$ | Nightly sleep is globally protective of subjective psychological stability |
| **Academic Performance Disruption** | $61.42\%$ | Percentage of global students reporting severe negative academic interference |
| **Dominant Platforms** | Instagram (39.2%), TikTok (21.6%), Facebook (15.1%), WhatsApp (6.5%) | Cross-continental penetration |

---

## Cohort 3: Adolescent Clinical Depression & Protective Factors (N = 1,200)

Adolescents (ages 13–19) represent a physiologically sensitive demographic facing nocturnal blue-light exposure and social comparison. Out of 1,200 teens, $31$ present confirmed clinical depression ($2.58\%$ severe imbalance).

### Cost-Sensitive 5-Fold Classification Results
- **ROC-AUC**: $0.9962$
- **PR-AUC (Precision-Recall)**: $0.9558$ (baseline $0.0258$)
- **Clinical Sensitivity (Recall)**: $93.55\%$ (detects 29 out of 31 depressed adolescents)
- **Specificity**: $100.00\%$
- **Optimal Decision Threshold (Youden J)**: $\tau^* = 0.220$ ($J = 0.9355$)
- **Brier Score (Calibration)**: $0.0017$

### Key Behavioral Discoveries
- **Sleep Deprivation as Primary Trigger**: Nightly sleep duration exhibits the single strongest negative correlation ($r = -0.1906$) with depression status.
- **Physical Activity as a Protective Buffer**: Teens engaging in regular physical exercise show an $84\%$ reduction in depression probability, even under elevated screen usage.
- **Nocturnal Screen Burden**: Screen usage within 60 minutes of bedtime strongly correlates with circadian breakdown and elevated anxiety ($r = +0.1696$).

---

## Visual Evidence & High-Resolution Telemetry

All figures are compiled at 300 DPI for scientific publication:

<table align="center">
<tr>
<td width="50%"><img src="images/behavioral_clustering_quadrant.png" alt="Behavioral Quadrant" width="100%"/></td>
<td width="50%"><img src="images/confusion_matrix_impact.png" alt="Confusion Matrix" width="100%"/></td>
</tr>
<tr>
<td align="center"><b>Figure 1:</b> Daily Screen Time vs. Sleep Duration Quadrant</td>
<td align="center"><b>Figure 2:</b> Normalized 3x3 Impact Confusion Matrix</td>
</tr>
<tr>
<td width="50%"><img src="images/teen_depression_clinical_curves.png" alt="Teen ROC and PR Curves" width="100%"/></td>
<td width="50%"><img src="images/teen_protective_factors.png" alt="Teen Protective Factors" width="100%"/></td>
</tr>
<tr>
<td align="center"><b>Figure 3:</b> Adolescent Depression ROC & PR Trade-off Curves</td>
<td align="center"><b>Figure 4:</b> Sleep & Physical Activity Protective Buffers</td>
</tr>
<tr>
<td width="50%"><img src="images/global_cross_national_addiction.png" alt="Global Addiction vs Mental Health" width="100%"/></td>
<td width="50%"><img src="images/feature_importance_shap.png" alt="SHAP Attribution" width="100%"/></td>
</tr>
<tr>
<td align="center"><b>Figure 5:</b> Cross-National Addiction vs Mental Health (113 Nations)</td>
<td align="center"><b>Figure 6:</b> Top Psychometric & Digital Attribution Drivers</td>
</tr>
</table>

---

## Explainability & Attribution Analysis

Through Shapley additive explanations (SHAP) and feature permutation:
1. `Mental_Health_Index` & `Sleep_Duration_Hours` account for $44.2\%$ of overall impact determination.
2. `screen_to_sleep_ratio` acts as the primary early-warning marker for shift into the `Negative` regime.
3. Bedtime screen exposure and nocturnal blue-light consumption are the primary drivers of adolescent anxiety amplification.

---

## Interactive Analytics Dashboard

A standalone HTML5 dashboard is located at `artifacts/dashboard.html` with Tailwind CSS and Plotly.js charts:
- Cohort 1 Behavioral Quadrant (Screen time vs. Sleep duration)
- Cohort 2 Global Addiction vs Mental Health regression across 113 nations
- Cohort 3 Adolescent Depression Risk Landscape and Bedtime Screen Burden
- Composite feature importance horizontal bar plot

---

## Single-Student Triage CLI

```bash
# Query student telemetry and risk profile
python predict.py --student-id STU_2024000
```

Sample output:
```text
======================================================================
         ISML — Student Behavioral Telemetry Report: STU_2024000
======================================================================
Demographics: 21yo | Male | Undergraduate
Primary Digital Platform: Instagram

Digital Consumption & Chronobiology:
  - Daily Screen Time:       6.2 hrs/day
  - Weekend Extra Time:      2.5 hrs/day
  - Weekly Screen Load:      48.4 hrs/week
  - Nightly Sleep Duration:  5.8 hrs (Debt: 2.2 hrs)
  - Sleep Quality Score:     2 / 5
  - Screen-to-Sleep Ratio:   1.07
  - Frequent Late-Night:     True

Psychometric & Academic Profiles:
  - Social Comparison:       Often
  - Perceived Stress (PSS):  27.4 / 40.0
  - Mental Health Index:     42 / 100
  - Cumulative GPA:          2.41 / 4.00
  - Clinical Impact Regime:  Negative
======================================================================
```

---

## Repository Structure

```text
ISML/
├── artifacts/
│   ├── dashboard.html                          # Standalone Plotly HTML5 Tri-Cohort dashboard
│   ├── feature_importance_benchmark.csv        # SHAP and gradient boosting importances
│   └── metrics_summary.json                    # Serialized tri-cohort evaluation metrics
├── data/
│   ├── raw/
│   │   ├── social_media_impact_on_life.csv     # Cohort 1: 4,500 students
│   │   ├── global_student_social_media_832.csv # Cohort 2: 832 students (113 countries)
│   │   └── Teen_Mental_Health_Dataset.csv      # Cohort 3: 1,200 teenagers
│   └── processed/
│       ├── social_media_engineered.parquet     # Feature-engineered Cohort 1
│       ├── global_student_engineered.parquet   # Feature-engineered Cohort 2
│       └── teen_mental_health_engineered.parquet # Feature-engineered Cohort 3
├── images/                                     # 8 publication figures (300 DPI)
├── notebooks/
│   └── isml_behavioral_monograph.ipynb         # Full reproducible monograph notebook
├── src/
│   ├── core/                                   # Project configs and immutable domain types
│   ├── data_pipeline/                          # Multi-cohort loaders and schema validators
│   ├── feature_engineering/                    # Chronobiological, global & teen feature engines
│   ├── modeling/                               # Impact classifier, distress regressor & depression model
│   ├── validation/                             # Stratified cross-validators & reporting
│   └── visualization/                          # Publication figures & HTML5 dashboard builders
├── tests/                                      # 13 pytest unit and integration tests (100% pass)
├── ELMAS.md                                    # Architectural charter and code quality standards
├── main.py                                     # End-to-end pipeline orchestrator
├── predict.py                                  # Single-student triage CLI
└── requirements.txt                            # Pinned dependencies
```

---

## Quickstart & Reproduction Guide

```bash
# Clone the repository
git clone https://github.com/elm-as/ISML.git
cd ISML

# Install dependencies
pip install -r requirements.txt

# Run complete tri-cohort pipeline
python main.py

# Run test suite
pytest tests/ -v
```

---

## The ElmasCore Charter Compliance

ISML strictly follows the **ELMAS.md** charter:
- **Max 350 lines per file**: Every module adheres strictly to the rule.
- **Pure functions & Early returns**: Max 3 levels of indentation, zero silent mutation.
- **Single Level of Abstraction**: High-level orchestrators separate from low-level transformations.
- **Typed Dataclasses**: Max 3 parameters per function, domain structures strictly typed.

---

## Citation & Attribution

```bibtex
@software{elmas_isml_2026,
  author = {Elmas, Tresor},
  title = {Impact of Social Media on Life: Tri-Cohort Behavioral, Chronobiological & Adolescent Depression Architecture},
  year = {2026},
  url = {https://github.com/elm-as/ISML}
}
```
