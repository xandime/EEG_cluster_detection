# External Validation Methods for Clustering

**Purpose:** Assess whether cluster assignments are associated with external demographic variables (age, sex).

---

## Age Validation (Continuous Variable)

### 1. One-Way ANOVA
**Question:** Do clusters differ significantly in mean age?

**Test:** F-statistic comparing between-cluster vs. within-cluster variance

**Null Hypothesis:** All clusters have equal mean age

**Output:**
- **F-statistic** and **p-value** (significance)
- **Degrees of freedom:** (k-1, n-k) where k = # clusters, n = # participants

---

### 2. Effect Size: Omega-Squared (ω²)
**Question:** How much age variance do clusters explain?

**Why not p-value alone:** Large samples make small effects significant

**Interpretation:**
- **ω² < 0.05:** Negligible effect (clusters don't explain age)
- **ω² = 0.05-0.10:** Small effect
- **ω² = 0.10-0.20:** Medium effect (age moderately related to clusters)
- **ω² > 0.20:** Large effect (age strongly determines clusters)

**Scientific meaning:**
- Low ω² (e.g., 0.08): Clusters capture EEG variance **beyond** just age → good!
- High ω² (e.g., 0.60): Clusters are basically "age bins" → less interesting

---

### 3. Post-Hoc Test: Games-Howell
**Question:** Which specific cluster pairs differ in age?

**Why Games-Howell:** Robust to unequal variances across clusters (common in real data)

**Output:**
- Pairwise comparisons (Cluster 1 vs. 2, 1 vs. 3, 2 vs. 3)
- Mean age difference with 95% confidence interval
- Adjusted p-value (controls for multiple comparisons)

**Example:**
```
Cluster 1 vs. 2: Mean diff = -5.3 years, 95% CI [-8.1, -2.5], p = 0.001
→ Cluster 2 is significantly older
```

---

### 4. Assumption Checks

#### Levene's Test (Homoscedasticity)
**Tests:** Do clusters have equal age variance?

**Decision rule:**
- p ≥ 0.05 → Equal variances → Use Tukey HSD (slightly more powerful)
- p < 0.05 → Unequal variances → Use Games-Howell

#### Shapiro-Wilk Test (Normality)
**Tests:** Is age normally distributed within each cluster?

**Decision rule:**
- All clusters p ≥ 0.05 → Use ANOVA
- Any cluster p < 0.05 → Use Kruskal-Wallis (non-parametric alternative)

---

### 5. Robustness: Kruskal-Wallis Test
**When to use:**
- Age distribution is skewed
- Small sample sizes
- Outliers present

**How it works:** Ranks all ages, tests if mean ranks differ between clusters

**Post-hoc:** Dunn's test with Bonferroni correction

---

## Sex Validation (Categorical Variable)

### 1. Chi-Square Test of Independence
**Question:** Is cluster membership associated with sex?

**Null Hypothesis:** Cluster and sex are independent

**Output:**
- **χ² statistic** and **p-value**
- **Degrees of freedom:** (# clusters - 1) × (# sex categories - 1)

**Assumption:** All cells in contingency table have expected count ≥ 5
- If violated → Use Fisher's exact test

---

### 2. Effect Size: Cramér's V
**Question:** How strong is the cluster-sex association?

**Range:** 0 (no association) to 1 (perfect association)

**Interpretation (for 3 clusters × 2 sexes):**
- **V < 0.10:** Negligible
- **V = 0.10-0.20:** Small
- **V = 0.20-0.30:** Medium
- **V > 0.30:** Large

**Scientific meaning:**
- V = 0.15: Weak association, clusters not driven by sex → good!
- V = 0.45: Strong association, may indicate sex-specific EEG patterns

---

### 3. Standardized Residuals
**Question:** Which cluster-sex combinations are over/underrepresented?

**Calculation:** (Observed - Expected) / √Expected

**Interpretation:**
- **|SR| > 2:** Cell contributes significantly to χ²
- **|SR| > 3:** Strong deviation from independence

**Example:**
```
          Cluster 0   Cluster 1   Cluster 2
Male        +2.5        -0.8        -1.2    ← Cluster 0 has excess males
Female      -2.5        +0.8        +1.2
```

**Use:** Identifies specific patterns (e.g., "Cluster 0 is 70% male vs. 50% expected")

---

## Advanced Methods

### 1. Multinomial Logistic Regression
**Question:** How well do age and sex **predict** cluster membership?

**Model:** P(Cluster = k) ~ β₀ + β₁×Age + β₂×Sex

**Output:**
- **Odds ratios:** Effect of 1-year age increase or being female on cluster odds
- **Pseudo-R²:** Variance explained by demographics

**Example:**
```
Cluster 2 (vs. Cluster 1):
- OR(Age) = 1.05 → Each year increases odds by 5%
- OR(Female) = 2.3 → Females 2.3× more likely in Cluster 2
```

**Advantage:** Tests demographics as **predictors**, not just associations

---

### 2. Two-Way ANCOVA
**Question:** Does the age-cluster relationship depend on sex?

**Model:** EEG_Feature ~ Cluster + Age + Sex + Age×Sex

**Tests:**
- **Cluster effect:** Do clusters differ in this feature?
- **Age effect:** Does age predict the feature?
- **Sex effect:** Do males/females differ?
- **Interaction (Age×Sex):** Does age slope differ by sex?

**Limitation:** Requires choosing a specific EEG feature as outcome

---

## Recommended Workflow

### Step 1: Assumption Checks 
1. Shapiro-Wilk test for age normality per cluster
2. Levene's test for age variance equality
3. Check contingency table cell counts ≥ 5

### Step 2: Primary Tests 
1. **Age:** ANOVA + omega-squared + Games-Howell
2. **Sex:** Chi-square + Cramér's V + standardized residuals

### Step 3: Reporting
For each test, report:
- Test statistic with degrees of freedom
- P-value
- Effect size (ω² or Cramér's V)
- 95% confidence intervals (for post-hoc tests)

---

## Python Packages

| Package | Purpose |
|---------|---------|
| **`pingouin`** | ANOVA, Games-Howell, omega-squared |
| **`scipy.stats`** | Chi-square, Levene's, Shapiro-Wilk |
| **`scikit-posthocs`** | Dunn's test (if using Kruskal-Wallis) |
| **`statsmodels`** | Logistic regression, ANCOVA |

---

## Decision Tree

```
AGE:
├─ Normal + equal variances? → ANOVA + Tukey HSD
├─ Normal + unequal variances? → ANOVA + Games-Howell ✓ (safest)
└─ Non-normal? → Kruskal-Wallis + Dunn's test

SEX:
├─ All cells ≥ 5? → Chi-square + Cramér's V ✓
└─ Any cell < 5? → Fisher's exact test

EFFECT SIZES:
├─ Age → Omega-squared (ω²)
└─ Sex → Cramér's V
```

---

## Interpretation Guidelines

### Good Clustering Result
- **Age:** p < 0.05, ω² = 0.10-0.15 (significant but moderate)
  - → Clusters capture age-related EEG changes but aren't just age bins
- **Sex:** p < 0.05, V = 0.15-0.25 (small-medium effect)
  - → Some sex differences present but not dominant

### Concerning Result
- **Age:** ω² > 0.50 → Clusters are redundant with age (just use age instead)
- **Sex:** V > 0.50 → Clusters are redundant with sex

### Null Result (Also Valid!)
- **Age:** p > 0.05, ω² < 0.02 → Clusters unrelated to age
- **Sex:** p > 0.05, V < 0.05 → Clusters unrelated to sex
- → Interpretation: Clusters represent EEG phenotypes independent of demographics

---

## Example Write-Up

> "Age differed significantly across clusters (F(2, 207) = 12.4, p < 0.001, ω² = 0.10). 
> Post-hoc Games-Howell tests revealed that Cluster 1 (M = 52.3, SD = 8.1) was significantly 
> older than Cluster 2 (M = 47.0, SD = 9.3, p = 0.002) and Cluster 3 (M = 46.5, SD = 7.9, 
> p = 0.001), with no difference between Clusters 2 and 3 (p = 0.78).
>
> Sex distribution differed across clusters (χ²(2) = 8.7, p = 0.013, V = 0.20). Standardized 
> residuals indicated that Cluster 3 contained significantly more males (70% vs. 50% expected, 
> SR = +2.3) than predicted under independence."

---

## References

- **Games-Howell test:** Games & Howell (1976). Pairwise multiple comparison procedures with unequal N's and/or variances.
- **Omega-squared:** Olejnik & Algina (2003). Generalized eta and omega squared statistics.
- **Cramér's V:** Cramér (1946). Mathematical Methods of Statistics.
- **Effect size interpretation:** Cohen (1988). Statistical Power Analysis for the Behavioral Sciences.

