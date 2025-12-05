import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import pingouin as pg
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# ========================================
# COMPREHENSIVE ANALYSIS FUNCTION
# ========================================

def analyze_continuous_variable(df: pd.DataFrame,
                                 group_col: str,
                                 value_col: str,
                                 alpha: float = 0.05,
                                 show_plot: bool = True,
                                 save_path: Optional[str] = None) -> Dict:
    """
    Comprehensive analysis of a continuous variable (e.g., Age) across groups (e.g., Clusters).

    Automatically selects the appropriate statistical test based on assumption checks.

    Logic Flow:
        1. Check normality (Shapiro-Wilk per group)
        2. Check variance homogeneity (Levene's test)
        3. Select and run main test:
           - Normal + Equal Variance → One-Way ANOVA
           - Normal + Unequal Variance → Welch's ANOVA
           - Non-Normal → Kruskal-Wallis
        4. If significant, run post-hoc:
           - After ANOVA → Games-Howell
           - After Kruskal-Wallis → Dunn's test
        5. Calculate effect size (ω² or ε²)
        6. Generate visualization

    Args:
        df: DataFrame containing the data
        group_col: Column name for group labels (e.g., 'cluster')
        value_col: Column name for continuous variable (e.g., 'age')
        alpha: Significance level (default: 0.05)
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Dictionary containing:
            - 'test_name': Name of the test used
            - 'statistic': Test statistic
            - 'p_value': P-value of the main test
            - 'significant': Boolean indicating significance
            - 'significant_pairs': List of significant pairwise comparisons
            - 'effect_size': Effect size value
            - 'effect_size_name': Name of effect size measure
            - 'effect_interpretation': Interpretation of effect size
            - 'assumptions': Dict with normality and variance check results
            - 'posthoc_table': DataFrame with post-hoc results (if applicable)
            - 'figure': matplotlib Figure object

    Example:
        >>> df = pd.DataFrame({'cluster': [0,0,0,1,1,1], 'age': [25,30,28,55,60,58]})
        >>> result = analyze_continuous_variable(df, 'cluster', 'age')
    """

    # =========================================================================
    # STEP 1: ASSUMPTION CHECK - NORMALITY (Shapiro-Wilk per group)
    # =========================================================================
    print("=" * 70)
    print(f"ANALYZING: {value_col} by {group_col}")
    print("=" * 70)

    groups = df[group_col].unique()
    n_groups = len(groups)

    print(f"\n[1/5] CHECKING NORMALITY (Shapiro-Wilk test per group)...")

    normality_results = {}
    all_normal = True

    for group in sorted(groups):
        group_data = df[df[group_col] == group][value_col].dropna()

        if len(group_data) >= 3:  # Shapiro-Wilk requires n >= 3
            stat, p = stats.shapiro(group_data)
            is_normal = p > alpha
            normality_results[group] = {'statistic': stat, 'p_value': p, 'normal': is_normal}

            if not is_normal:
                all_normal = False

            status = "✓ Normal" if is_normal else "✗ Non-normal"
            print(f"       Group {group}: W={stat:.4f}, p={p:.4f} → {status}")
        else:
            normality_results[group] = {'statistic': None, 'p_value': None, 'normal': None}
            print(f"       Group {group}: Insufficient samples (n={len(group_data)})")

    normality_assumption = all_normal
    print(f"       → Overall: {'NORMAL' if normality_assumption else 'NON-NORMAL'}")

    # =========================================================================
    # STEP 2: ASSUMPTION CHECK - VARIANCE HOMOGENEITY (Levene's test)
    # =========================================================================
    print(f"\n[2/5] CHECKING VARIANCE HOMOGENEITY (Levene's test)...")

    group_data_list = [df[df[group_col] == g][value_col].dropna().values for g in sorted(groups)]
    levene_stat, levene_p = stats.levene(*group_data_list)
    equal_variance = levene_p > alpha

    status = "✓ Equal" if equal_variance else "✗ Unequal"
    print(f"       Levene's test: W={levene_stat:.4f}, p={levene_p:.4f} → {status} variances")

    # =========================================================================
    # STEP 3: SELECT AND RUN MAIN TEST
    # =========================================================================
    print(f"\n[3/5] SELECTING AND RUNNING MAIN TEST...")

    if normality_assumption and equal_variance:
        # Normal + Equal Variance → One-Way ANOVA
        test_name = "One-Way ANOVA"
        print(f"       Assumptions: Normal=True, Equal Variance=True")
        print(f"       → Running {test_name}...")

        f_stat, p_value = stats.f_oneway(*group_data_list)
        statistic = f_stat
        test_type = "parametric"

    elif normality_assumption and not equal_variance:
        # Normal + Unequal Variance → Welch's ANOVA
        test_name = "Welch's ANOVA"
        print(f"       Assumptions: Normal=True, Equal Variance=False")
        print(f"       → Running {test_name}...")

        # Use pingouin for Welch's ANOVA
        welch_result = pg.welch_anova(data=df, dv=value_col, between=group_col)
        f_stat = welch_result['F'].values[0]
        p_value = welch_result['p-unc'].values[0]
        statistic = f_stat
        test_type = "parametric"

    else:
        # Non-Normal → Kruskal-Wallis
        test_name = "Kruskal-Wallis"
        print(f"       Assumptions: Normal=False")
        print(f"       → Running {test_name}...")

        h_stat, p_value = stats.kruskal(*group_data_list)
        statistic = h_stat
        test_type = "non-parametric"

    significant = p_value < alpha
    sig_symbol = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

    print(f"       Result: statistic={statistic:.4f}, p={p_value:.4f} {sig_symbol}")
    print(f"       → {'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'} (α={alpha})")

    # =========================================================================
    # STEP 4: POST-HOC ANALYSIS (if significant)
    # =========================================================================
    print(f"\n[4/5] POST-HOC ANALYSIS...")

    significant_pairs = []
    posthoc_table = None

    if significant and n_groups > 2:
        if test_type == "parametric":
            # Games-Howell (robust to unequal variances)
            print(f"       → Running Games-Howell post-hoc test...")
            posthoc_result = pg.pairwise_gameshowell(data=df, dv=value_col, between=group_col)
            posthoc_table = posthoc_result

            # Extract significant pairs
            for _, row in posthoc_result.iterrows():
                if row['pval'] < alpha:
                    pair = (int(row['A']), int(row['B']))
                    significant_pairs.append(pair)
                    print(f"       {row['A']} vs {row['B']}: p={row['pval']:.4f} *")

        else:
            # Dunn's test with Bonferroni correction
            print(f"       → Running Dunn's test (Bonferroni correction)...")
            posthoc_matrix = sp.posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust='bonferroni')

            # Convert matrix to pairwise table
            pairs_list = []
            sorted_groups = sorted(groups)
            for i, g1 in enumerate(sorted_groups):
                for g2 in sorted_groups[i+1:]:
                    p_adj = posthoc_matrix.loc[g1, g2]
                    pairs_list.append({'group_1': g1, 'group_2': g2, 'p_adj': p_adj})
                    if p_adj < alpha:
                        significant_pairs.append((int(g1), int(g2)))
                        print(f"       {g1} vs {g2}: p_adj={p_adj:.4f} *")

            posthoc_table = pd.DataFrame(pairs_list)

        if not significant_pairs:
            print(f"       → No significant pairwise differences found")
    elif significant and n_groups == 2:
        print(f"       → Only 2 groups, no post-hoc needed (main test is sufficient)")
        significant_pairs = [(sorted(groups)[0], sorted(groups)[1])]
    else:
        print(f"       → Skipped (main test not significant)")

    # =========================================================================
    # STEP 5: EFFECT SIZE
    # =========================================================================
    print(f"\n[5/5] CALCULATING EFFECT SIZE...")

    # Get overall data
    all_values = df[value_col].dropna().values
    all_groups = df.loc[df[value_col].notna(), group_col].values

    n_total = len(all_values)
    grand_mean = np.mean(all_values)

    if test_type == "parametric":
        # Omega-Squared (ω²) for ANOVA
        effect_size_name = "Omega-squared (ω²)"

        # Calculate SS_between
        ss_between = sum([
            len(df[df[group_col] == g][value_col].dropna()) *
            (df[df[group_col] == g][value_col].mean() - grand_mean) ** 2
            for g in groups
        ])

        # Calculate SS_total
        ss_total = np.sum((all_values - grand_mean) ** 2)

        # Calculate SS_within and MS_within
        ss_within = ss_total - ss_between
        df_between = n_groups - 1
        df_within = n_total - n_groups
        ms_within = ss_within / df_within

        # Omega-squared
        effect_size = (ss_between - df_between * ms_within) / (ss_total + ms_within)
        effect_size = max(0, effect_size)  # Can't be negative

    else:
        # Epsilon-Squared (ε²) for Kruskal-Wallis
        effect_size_name = "Epsilon-squared (ε²)"

        # ε² = H / (n² - 1) / (n - 1)  simplified: H / (n - 1)
        effect_size = (statistic - n_groups + 1) / (n_total - n_groups)
        effect_size = max(0, min(1, effect_size))  # Bound between 0 and 1

    # Interpret effect size
    if effect_size < 0.01:
        effect_interpretation = "Negligible"
    elif effect_size < 0.06:
        effect_interpretation = "Small"
    elif effect_size < 0.14:
        effect_interpretation = "Medium"
    else:
        effect_interpretation = "Large"

    print(f"       {effect_size_name} = {effect_size:.4f} ({effect_interpretation})")

    # =========================================================================
    # STEP 6: VISUALIZATION
    # =========================================================================
    print(f"\n[PLOT] Generating visualization...")

    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create boxplot with stripplot overlay
    sns.boxplot(data=df, x=group_col, y=value_col, hue=group_col, ax=ax,
                palette='Set2', width=0.5, boxprops=dict(alpha=0.7), legend=False)
    sns.stripplot(data=df, x=group_col, y=value_col, ax=ax,
                  color='black', alpha=0.5, size=5, jitter=True)

    # Add title with test result
    title = f"{value_col} by {group_col}\n"
    title += f"{test_name}: "
    if test_type == "parametric":
        title += f"F={statistic:.2f}, "
    else:
        title += f"H={statistic:.2f}, "
    title += f"p={p_value:.4f} {sig_symbol}"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(group_col, fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)

    # Add effect size annotation
    ax.text(0.98, 0.98, f"{effect_size_name}\n{effect_size:.3f} ({effect_interpretation})",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add group statistics
    stats_text = "Group means:\n"
    for g in sorted(groups):
        g_mean = df[df[group_col] == g][value_col].mean()
        g_std = df[df[group_col] == g][value_col].std()
        g_n = len(df[df[group_col] == g][value_col].dropna())
        stats_text += f"  {g}: {g_mean:.1f}±{g_std:.1f} (n={g_n})\n"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"       Figure saved to: {save_path}")

    if show_plot:
        plt.ion()
        plt.show()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test used: {test_name}")
    print(f"Result: {'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'} (p={p_value:.4f})")
    if significant_pairs:
        print(f"Significant pairs: {significant_pairs}")
    print(f"Effect size: {effect_size_name} = {effect_size:.4f} ({effect_interpretation})")
    print("=" * 70)

    # Return results dictionary
    return {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': significant,
        'significant_pairs': significant_pairs,
        'effect_size': effect_size,
        'effect_size_name': effect_size_name,
        'effect_interpretation': effect_interpretation,
        'assumptions': {
            'normality': normality_assumption,
            'normality_results': normality_results,
            'equal_variance': equal_variance,
            'levene_p': levene_p
        },
        'posthoc_table': posthoc_table,
        'figure': fig
    }


def analyze_categorical_variable(df: pd.DataFrame,
                                  group_col: str,
                                  cat_col: str,
                                  alpha: float = 0.05,
                                  show_plot: bool = True,
                                  save_path: Optional[str] = None) -> Dict:
    """
    Comprehensive analysis of categorical variable relationships (e.g., Cluster vs Sex).

    Automatically selects the appropriate statistical test based on expected frequencies.

    Logic Flow:
        1. Create contingency table (cross-tabulation)
        2. Check expected frequencies:
           - If any cell < 5 and 2x2 table → Fisher's Exact Test
           - If any cell < 5 and larger table → Chi-Square with warning
           - Otherwise → Chi-Square Test of Independence
        3. Calculate Adjusted Standardized Residuals (post-hoc)
        4. Identify significant cells (|residual| > 1.96)
        5. Calculate Cramér's V effect size
        6. Generate stacked bar chart visualization

    Args:
        df: DataFrame containing the data
        group_col: Column name for group labels (e.g., 'cluster')
        cat_col: Column name for categorical variable (e.g., 'sex', 'age_group')
        alpha: Significance level (default: 0.05)
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Dictionary containing:
            - 'test_name': Name of the test used
            - 'statistic': Test statistic (chi2 or odds ratio)
            - 'p_value': P-value of the test
            - 'significant': Boolean indicating significance
            - 'contingency_table': Observed frequencies
            - 'expected_frequencies': Expected frequencies under independence
            - 'residuals_table': Adjusted standardized residuals
            - 'significant_cells': List of (group, category, residual) tuples
            - 'cramers_v': Cramér's V effect size
            - 'cramers_v_interpretation': Interpretation of effect size
            - 'assumption_warning': Warning message if assumptions violated
            - 'figure': matplotlib Figure object

    Example:
        >>> df = pd.DataFrame({'cluster': [0,0,1,1,2,2], 'sex': ['M','F','M','M','F','F']})
        >>> result = analyze_categorical_variable(df, 'cluster', 'sex')
    """

    # =========================================================================
    # STEP 1: CREATE CONTINGENCY TABLE
    # =========================================================================
    print("=" * 70)
    print(f"ANALYZING: {cat_col} by {group_col}")
    print("=" * 70)

    print(f"\n[1/5] CREATING CONTINGENCY TABLE...")

    # Create contingency table
    contingency_table = pd.crosstab(df[group_col], df[cat_col])

    n_groups = contingency_table.shape[0]
    n_categories = contingency_table.shape[1]
    n_total = contingency_table.sum().sum()

    print(f"       Table shape: {n_groups} groups × {n_categories} categories")
    print(f"       Total observations: {n_total}")
    print(f"\n       Observed frequencies:")
    print(contingency_table.to_string().replace('\n', '\n       '))

    # =========================================================================
    # STEP 2: ASSUMPTION CHECK - EXPECTED FREQUENCIES
    # =========================================================================
    print(f"\n[2/5] CHECKING EXPECTED FREQUENCIES...")

    # Calculate expected frequencies
    chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    expected_df = pd.DataFrame(expected,
                               index=contingency_table.index,
                               columns=contingency_table.columns)

    # Check if any expected frequency < 5
    min_expected = expected.min()
    cells_below_5 = (expected < 5).sum()
    total_cells = expected.size
    pct_below_5 = (cells_below_5 / total_cells) * 100

    print(f"       Minimum expected frequency: {min_expected:.2f}")
    print(f"       Cells with expected < 5: {cells_below_5}/{total_cells} ({pct_below_5:.1f}%)")

    assumption_warning = None

    # =========================================================================
    # STEP 3: SELECT AND RUN MAIN TEST
    # =========================================================================
    print(f"\n[3/5] SELECTING AND RUNNING MAIN TEST...")

    is_2x2 = (n_groups == 2 and n_categories == 2)

    if min_expected < 5:
        if is_2x2:
            # Fisher's Exact Test for 2x2 tables
            test_name = "Fisher's Exact Test"
            print(f"       Expected frequency < 5 detected in 2×2 table")
            print(f"       → Running {test_name}...")

            odds_ratio, p_value = stats.fisher_exact(contingency_table.values)
            statistic = odds_ratio

        else:
            # Chi-Square with warning for larger tables
            test_name = "Chi-Square Test (with warning)"
            assumption_warning = (f"WARNING: {cells_below_5} cells ({pct_below_5:.1f}%) have expected "
                                  f"frequency < 5. Results may be unreliable.")
            print(f"       ⚠ {assumption_warning}")
            print(f"       → Running Chi-Square Test anyway (no alternative for non-2×2)...")

            statistic = chi2_stat
            p_value = p_value_chi2
    else:
        # Standard Chi-Square Test
        test_name = "Chi-Square Test"
        print(f"       All expected frequencies ≥ 5 ✓")
        print(f"       → Running {test_name}...")

        statistic = chi2_stat
        p_value = p_value_chi2

    significant = p_value < alpha
    sig_symbol = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

    if test_name == "Fisher's Exact Test":
        print(f"       Result: Odds Ratio={statistic:.4f}, p={p_value:.4f} {sig_symbol}")
    else:
        print(f"       Result: χ²={statistic:.4f}, df={dof}, p={p_value:.4f} {sig_symbol}")
    print(f"       → {'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'} (α={alpha})")

    # =========================================================================
    # STEP 4: POST-HOC - ADJUSTED STANDARDIZED RESIDUALS
    # =========================================================================
    print(f"\n[4/5] CALCULATING ADJUSTED STANDARDIZED RESIDUALS...")

    # Calculate adjusted standardized residuals
    # Formula: (observed - expected) / sqrt(expected * (1 - row_margin/n) * (1 - col_margin/n))
    observed = contingency_table.values

    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)

    # Adjusted standardized residuals (ASR)
    residuals = (observed - expected) / np.sqrt(
        expected * (1 - row_totals / n_total) * (1 - col_totals / n_total)
    )

    residuals_df = pd.DataFrame(residuals,
                                index=contingency_table.index,
                                columns=contingency_table.columns)

    # Identify significant cells (|residual| > 1.96 for α=0.05)
    z_critical = stats.norm.ppf(1 - alpha / 2)  # 1.96 for α=0.05
    significant_cells = []

    print(f"       Critical value: |z| > {z_critical:.2f}")
    print(f"\n       Adjusted Standardized Residuals:")

    for group in contingency_table.index:
        for cat in contingency_table.columns:
            resid = residuals_df.loc[group, cat]
            obs = contingency_table.loc[group, cat]
            exp = expected_df.loc[group, cat]

            if abs(resid) > z_critical:
                direction = "OVER" if resid > 0 else "UNDER"
                significant_cells.append({
                    'group': group,
                    'category': cat,
                    'observed': obs,
                    'expected': exp,
                    'residual': resid,
                    'direction': direction
                })

    # Print residuals table
    print(residuals_df.round(2).to_string().replace('\n', '\n       '))

    if significant_cells:
        print(f"\n       Significant cells (|z| > {z_critical:.2f}):")
        for cell in significant_cells:
            symbol = "↑" if cell['direction'] == "OVER" else "↓"
            print(f"       {symbol} {cell['group']} × {cell['category']}: "
                  f"z={cell['residual']:.2f} ({cell['direction']}-represented)")
    else:
        print(f"\n       No significant deviations found")

    # =========================================================================
    # STEP 5: EFFECT SIZE - CRAMÉR'S V
    # =========================================================================
    print(f"\n[5/5] CALCULATING EFFECT SIZE (Cramér's V)...")

    # Cramér's V = sqrt(chi2 / (n * min(r-1, c-1)))
    min_dim = min(n_groups - 1, n_categories - 1)
    if min_dim > 0:
        cramers_v = np.sqrt(chi2_stat / (n_total * min_dim))
    else:
        cramers_v = 0.0

    # Interpret effect size (Cohen's conventions)
    if cramers_v < 0.10:
        cramers_v_interpretation = "Negligible"
    elif cramers_v < 0.20:
        cramers_v_interpretation = "Small"
    elif cramers_v < 0.30:
        cramers_v_interpretation = "Medium"
    else:
        cramers_v_interpretation = "Large"

    print(f"       Cramér's V = {cramers_v:.4f} ({cramers_v_interpretation})")

    # =========================================================================
    # STEP 6: VISUALIZATION - STACKED BAR CHART
    # =========================================================================
    print(f"\n[PLOT] Generating stacked bar chart...")

    plt.ioff()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Percentage stacked bar chart ---
    # Calculate percentages within each group
    pct_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

    # Plot stacked bar chart
    pct_table.plot(kind='bar', stacked=True, ax=axes[0],
                   colormap='Set2', edgecolor='black', linewidth=0.5)

    axes[0].set_title(f'{cat_col} Distribution by {group_col}\n(Percentages)',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel(group_col, fontsize=11)
    axes[0].set_ylabel('Percentage (%)', fontsize=11)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(title=cat_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    axes[0].set_ylim(0, 100)

    # Add percentage labels on bars
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.1f%%', label_type='center', fontsize=8)

    # --- Plot 2: Residuals heatmap ---
    sns.heatmap(residuals_df, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-3, vmax=3, ax=axes[1],
                cbar_kws={'label': 'Adjusted Standardized Residual'},
                linewidths=0.5, linecolor='gray')

    # Mark significant cells
    for i, group in enumerate(residuals_df.index):
        for j, cat in enumerate(residuals_df.columns):
            if abs(residuals_df.loc[group, cat]) > z_critical:
                axes[1].add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                                 edgecolor='black', lw=2))

    axes[1].set_title(f'Adjusted Standardized Residuals\n(|z| > {z_critical:.2f} marked)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel(cat_col, fontsize=11)
    axes[1].set_ylabel(group_col, fontsize=11)

    # Add test result as suptitle
    if test_name == "Fisher's Exact Test":
        suptitle = f"{test_name}: OR={statistic:.2f}, p={p_value:.4f} {sig_symbol}"
    else:
        suptitle = f"{test_name}: χ²={statistic:.2f}, p={p_value:.4f} {sig_symbol}"
    suptitle += f" | Cramér's V={cramers_v:.3f} ({cramers_v_interpretation})"

    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"       Figure saved to: {save_path}")

    if show_plot:
        plt.ion()
        plt.show()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test used: {test_name}")
    if assumption_warning:
        print(f"⚠ {assumption_warning}")
    print(f"Result: {'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'} (p={p_value:.4f})")
    if significant_cells:
        print(f"Significant associations:")
        for cell in significant_cells:
            symbol = "↑" if cell['direction'] == "OVER" else "↓"
            print(f"  {symbol} {cell['group']} has {cell['direction'].lower()} {cell['category']} "
                  f"(z={cell['residual']:.2f})")
    print(f"Effect size: Cramér's V = {cramers_v:.4f} ({cramers_v_interpretation})")
    print("=" * 70)

    # Return results dictionary
    return {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'dof': dof if test_name != "Fisher's Exact Test" else None,
        'significant': significant,
        'contingency_table': contingency_table,
        'expected_frequencies': expected_df,
        'residuals_table': residuals_df,
        'significant_cells': significant_cells,
        'cramers_v': cramers_v,
        'cramers_v_interpretation': cramers_v_interpretation,
        'assumption_warning': assumption_warning,
        'figure': fig
    }


# ========================================
# AGE VALIDATION (Continuous Variable)
# ========================================

def one_way_anova(age: np.ndarray, cluster_labels: np.ndarray) -> Dict:
    """
    Perform one-way ANOVA to test if clusters differ significantly in mean age.

    Question: Do clusters have different average ages?
    Null Hypothesis: All clusters have equal mean age.

    Args:
        age: numpy array of shape (N,). Age for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment for each participant.

    Returns:
        dict containing:
            - 'f_statistic': F-statistic value
            - 'p_value': significance level (reject null if p < 0.05)
            - 'df_between': degrees of freedom between groups (k-1)
            - 'df_within': degrees of freedom within groups (n-k)
            - 'mean_ages': mean age per cluster
            - 'std_ages': standard deviation of age per cluster
            - 'n_per_cluster': number of participants per cluster

    Interpretation:
        - p < 0.05: Clusters have significantly different mean ages
        - p >= 0.05: No significant age difference between clusters

    Example:
        >>> ages = np.array([25, 30, 35, 60, 65, 70])
        >>> clusters = np.array([0, 0, 0, 1, 1, 1])
        >>> result = one_way_anova(ages, clusters)
        >>> print(f"F={result['f_statistic']:.2f}, p={result['p_value']:.4f}")
    """
    # Group ages by cluster
    unique_clusters = np.unique(cluster_labels)
    age_groups = [age[cluster_labels == c] for c in unique_clusters]

    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*age_groups)

    # Calculate descriptive statistics per cluster
    mean_ages = {c: np.mean(age[cluster_labels == c]) for c in unique_clusters}
    std_ages = {c: np.std(age[cluster_labels == c], ddof=1) for c in unique_clusters}
    n_per_cluster = {c: np.sum(cluster_labels == c) for c in unique_clusters}

    # Degrees of freedom
    k = len(unique_clusters)  # number of clusters
    n = len(age)  # total participants
    df_between = k - 1
    df_within = n - k

    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'df_between': df_between,
        'df_within': df_within,
        'mean_ages': mean_ages,
        'std_ages': std_ages,
        'n_per_cluster': n_per_cluster
    }

def two_way_anova(age: np.ndarray, cluster_labels: np.ndarray, sex: np.ndarray) -> Dict:
    """
    Perform two-way ANOVA to test effects of cluster and sex on age.

    Question: Do clusters and sex independently/jointly affect age distribution?
    Tests:
        1. Main effect of cluster on age
        2. Main effect of sex on age
        3. Interaction effect (cluster × sex)

    Args:
        age: numpy array of shape (N,). Age for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.
        sex: numpy array of shape (N,). Sex category (e.g., 'M', 'F').

    Returns:
        dict containing:
            - 'cluster_effect': dict with F-statistic and p-value for cluster main effect
            - 'sex_effect': dict with F-statistic and p-value for sex main effect
            - 'interaction_effect': dict with F-statistic and p-value for cluster×sex interaction
            - 'summary_table': pandas DataFrame with full ANOVA table

    Interpretation:
        - cluster_effect p < 0.05: Clusters differ in age (independent of sex)
        - sex_effect p < 0.05: Males and females differ in age (independent of clusters)
        - interaction_effect p < 0.05: Age difference between sexes varies by cluster

    Example:
        >>> ages = np.array([25, 30, 35, 60, 65, 70])
        >>> clusters = np.array([0, 0, 0, 1, 1, 1])
        >>> sex = np.array(['M', 'F', 'M', 'F', 'M', 'F'])
        >>> result = two_way_anova(ages, clusters, sex)
        >>> print(result['summary_table'])
    """
    # Create DataFrame for statsmodels
    df = pd.DataFrame({
        'age': age,
        'cluster': cluster_labels.astype(str),
        'sex': sex.astype(str)
    })

    # Fit OLS model with interaction term using statsmodels
    model = ols('age ~ C(cluster) + C(sex) + C(cluster):C(sex)', data=df).fit()
    anova_table = anova_lm(model, typ=2)  # Type II SS for unbalanced designs

    # Extract results
    cluster_effect = {
        'f_statistic': anova_table.loc['C(cluster)', 'F'],
        'p_value': anova_table.loc['C(cluster)', 'PR(>F)'],
        'df': (int(anova_table.loc['C(cluster)', 'df']), int(anova_table.loc['Residual', 'df']))
    }
    sex_effect = {
        'f_statistic': anova_table.loc['C(sex)', 'F'],
        'p_value': anova_table.loc['C(sex)', 'PR(>F)'],
        'df': (int(anova_table.loc['C(sex)', 'df']), int(anova_table.loc['Residual', 'df']))
    }
    interaction_effect = {
        'f_statistic': anova_table.loc['C(cluster):C(sex)', 'F'],
        'p_value': anova_table.loc['C(cluster):C(sex)', 'PR(>F)'],
        'df': (int(anova_table.loc['C(cluster):C(sex)', 'df']), int(anova_table.loc['Residual', 'df']))
    }

    # Create clean summary table
    summary_table = pd.DataFrame({
        'Source': ['Cluster', 'Sex', 'Cluster × Sex', 'Residual'],
        'SS': [anova_table.loc['C(cluster)', 'sum_sq'],
               anova_table.loc['C(sex)', 'sum_sq'],
               anova_table.loc['C(cluster):C(sex)', 'sum_sq'],
               anova_table.loc['Residual', 'sum_sq']],
        'df': [int(anova_table.loc['C(cluster)', 'df']),
               int(anova_table.loc['C(sex)', 'df']),
               int(anova_table.loc['C(cluster):C(sex)', 'df']),
               int(anova_table.loc['Residual', 'df'])],
        'F': [cluster_effect['f_statistic'], sex_effect['f_statistic'],
              interaction_effect['f_statistic'], np.nan],
        'p-value': [cluster_effect['p_value'], sex_effect['p_value'],
                    interaction_effect['p_value'], np.nan]
    })

    return {
        'cluster_effect': cluster_effect,
        'sex_effect': sex_effect,
        'interaction_effect': interaction_effect,
        'summary_table': summary_table
    }

def omega_squared(age: np.ndarray, cluster_labels: np.ndarray) -> Dict:
    """
    Calculate omega-squared effect size for one-way ANOVA.

    Question: How much age variance do clusters explain?

    Args:
        age: numpy array of shape (N,). Age for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.

    Returns:
        dict containing:
            - 'omega_squared': ω² value (can be negative if effect is very small)
            - 'interpretation': Qualitative interpretation of effect size

    Interpretation:
        - ω² < 0.01: Negligible effect
        - ω² = 0.01-0.06: Small effect
        - ω² = 0.06-0.14: Medium effect
        - ω² > 0.14: Large effect

    Note: Omega-squared adjusts for sample size, unlike eta-squared.

    Example:
        >>> result = omega_squared(ages, clusters)
        >>> print(f"ω² = {result['omega_squared']:.3f} ({result['interpretation']})")
    """
    # Perform ANOVA to get F-statistic
    unique_clusters = np.unique(cluster_labels)
    age_groups = [age[cluster_labels == c] for c in unique_clusters]
    f_stat, _ = stats.f_oneway(*age_groups)

    # Calculate degrees of freedom
    k = len(unique_clusters)  # number of groups
    n = len(age)  # total sample size
    df_between = k - 1
    df_within = n - k

    # Calculate mean square within (error)
    grand_mean = np.mean(age)
    ss_total = np.sum((age - grand_mean) ** 2)
    ss_between = sum([np.sum(cluster_labels == c) * (np.mean(age[cluster_labels == c]) - grand_mean) ** 2
                      for c in unique_clusters])
    ss_within = ss_total - ss_between
    ms_within = ss_within / df_within

    # Calculate omega-squared
    omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within)

    # Interpret effect size
    if omega_sq < 0.01:
        interpretation = "Negligible"
    elif omega_sq < 0.06:
        interpretation = "Small"
    elif omega_sq < 0.14:
        interpretation = "Medium"
    else:
        interpretation = "Large"

    return {
        'omega_squared': omega_sq,
        'interpretation': interpretation
    }

def levene_test(age: np.ndarray, cluster_labels: np.ndarray) -> Dict:
    """
    Perform Levene's test for homogeneity of variance across clusters.

    Question: Do clusters have equal age variance?

    Args:
        age: numpy array of shape (N,). Age for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.

    Returns:
        dict containing:
            - 'statistic': Levene's test statistic
            - 'p_value': significance level
            - 'homoscedastic': Boolean, True if variances are equal (p >= 0.05)
            - 'recommendation': Which post-hoc test to use

    Interpretation:
        - p >= 0.05: Equal variances → Use Tukey HSD
        - p < 0.05: Unequal variances → Use Games-Howell

    Example:
        >>> result = levene_test(ages, clusters)
        >>> print(f"Levene's test: {result['recommendation']}")
    """
    unique_clusters = np.unique(cluster_labels)
    age_groups = [age[cluster_labels == c] for c in unique_clusters]

    statistic, p_value = stats.levene(*age_groups)

    homoscedastic = p_value >= 0.05
    recommendation = "Use Tukey HSD (equal variances)" if homoscedastic else "Use Games-Howell (unequal variances)"

    return {
        'statistic': statistic,
        'p_value': p_value,
        'homoscedastic': homoscedastic,
        'recommendation': recommendation
    }

def shapiro_wilk_test(age: np.ndarray, cluster_labels: np.ndarray) -> Dict:
    """
    Perform Shapiro-Wilk test for normality within each cluster.

    Question: Is age normally distributed within each cluster?

    Args:
        age: numpy array of shape (N,). Age for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.

    Returns:
        dict containing:
            - 'results': dict with statistic and p-value for each cluster
            - 'all_normal': Boolean, True if all clusters pass normality (p >= 0.05)
            - 'recommendation': Which test to use

    Interpretation:
        - All p >= 0.05: Normal distribution → Use ANOVA
        - Any p < 0.05: Non-normal distribution → Use Kruskal-Wallis

    Example:
        >>> result = shapiro_wilk_test(ages, clusters)
        >>> print(f"Normality: {result['recommendation']}")
    """
    unique_clusters = np.unique(cluster_labels)
    results = {}

    for cluster in unique_clusters:
        cluster_ages = age[cluster_labels == cluster]
        if len(cluster_ages) >= 3:  # Shapiro-Wilk requires at least 3 samples
            statistic, p_value = stats.shapiro(cluster_ages)
            results[cluster] = {
                'statistic': statistic,
                'p_value': p_value,
                'normal': p_value >= 0.05
            }
        else:
            results[cluster] = {
                'statistic': None,
                'p_value': None,
                'normal': None
            }

    all_normal = all([r['normal'] for r in results.values() if r['normal'] is not None])
    recommendation = "Use ANOVA (all normal)" if all_normal else "Use Kruskal-Wallis (non-normal detected)"

    return {
        'results': results,
        'all_normal': all_normal,
        'recommendation': recommendation
    }

def kruskal_wallis_test(age: np.ndarray, cluster_labels: np.ndarray) -> Dict:
    """
    Perform Kruskal-Wallis test (non-parametric alternative to ANOVA).

    Question: Do clusters differ in age distribution?
    Use when: Age is not normally distributed or has outliers.

    Args:
        age: numpy array of shape (N,). Age for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.

    Returns:
        dict containing:
            - 'h_statistic': Kruskal-Wallis H-statistic
            - 'p_value': significance level
            - 'median_ages': median age per cluster
            - 'mean_ranks': mean rank per cluster

    Interpretation:
        - p < 0.05: Clusters differ significantly in age distribution
        - p >= 0.05: No significant difference in age distribution

    Note: Use Dunn's test for post-hoc pairwise comparisons.

    Example:
        >>> result = kruskal_wallis_test(ages, clusters)
        >>> print(f"H={result['h_statistic']:.2f}, p={result['p_value']:.4f}")
    """
    unique_clusters = np.unique(cluster_labels)
    age_groups = [age[cluster_labels == c] for c in unique_clusters]

    h_statistic, p_value = stats.kruskal(*age_groups)

    # Calculate median ages
    median_ages = {c: np.median(age[cluster_labels == c]) for c in unique_clusters}

    # Calculate mean ranks
    from scipy.stats import rankdata
    ranks = rankdata(age)
    mean_ranks = {c: np.mean(ranks[cluster_labels == c]) for c in unique_clusters}

    return {
        'h_statistic': h_statistic,
        'p_value': p_value,
        'median_ages': median_ages,
        'mean_ranks': mean_ranks
    }


# ========================================
# SEX VALIDATION (Categorical Variable)
# ========================================

def chi_square_test(sex: np.ndarray, cluster_labels: np.ndarray) -> Dict:
    """
    Perform chi-square test of independence for sex distribution across clusters.

    Question: Is cluster membership associated with sex?
    Null Hypothesis: Cluster and sex are independent.

    Args:
        sex: numpy array of shape (N,). Sex category for each participant (e.g., 'M', 'F').
        cluster_labels: numpy array of shape (N,). Cluster assignment for each participant.

    Returns:
        dict containing:
            - 'chi2_statistic': Chi-square statistic value
            - 'p_value': significance level (reject null if p < 0.05)
            - 'dof': degrees of freedom
            - 'expected_freq': expected frequencies under independence
            - 'observed_freq': observed contingency table
            - 'standardized_residuals': standardized residuals for each cell

    Interpretation:
        - p < 0.05: Sex distribution differs significantly across clusters
        - p >= 0.05: Sex distribution is independent of clusters
        - |standardized_residuals| > 2: Cell contributes significantly to chi-square

    Example:
        >>> sex = np.array(['M', 'F', 'M', 'F', 'M', 'F'])
        >>> clusters = np.array([0, 0, 0, 1, 1, 1])
        >>> result = chi_square_test(sex, clusters)
        >>> print(f"χ²={result['chi2_statistic']:.2f}, p={result['p_value']:.4f}")
    """
    # Create contingency table
    contingency_table = pd.crosstab(cluster_labels, sex)

    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Calculate standardized residuals
    standardized_residuals = (contingency_table.values - expected) / np.sqrt(expected)

    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'expected_freq': expected,
        'observed_freq': contingency_table,
        'standardized_residuals': pd.DataFrame(
            standardized_residuals,
            index=contingency_table.index,
            columns=contingency_table.columns
        )
    }

def cramers_v(sex: np.ndarray, cluster_labels: np.ndarray) -> Dict:
    """
    Calculate Cramér's V effect size for chi-square test.

    Question: How strong is the cluster-sex association?

    Args:
        sex: numpy array of shape (N,). Sex category for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.

    Returns:
        dict containing:
            - 'cramers_v': Cramér's V value (0 to 1)
            - 'interpretation': Qualitative interpretation of effect size

    Interpretation:
        - V < 0.10: Negligible association
        - V = 0.10-0.20: Small association
        - V = 0.20-0.30: Medium association
        - V > 0.30: Large association

    Example:
        >>> result = cramers_v(sex, clusters)
        >>> print(f"Cramér's V = {result['cramers_v']:.3f} ({result['interpretation']})")
    """
    # Create contingency table
    contingency_table = pd.crosstab(cluster_labels, sex)

    # Perform chi-square test
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)

    # Calculate Cramér's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
    cramers_v_value = np.sqrt(chi2 / (n * min_dim))

    # Interpret effect size
    if cramers_v_value < 0.10:
        interpretation = "Negligible"
    elif cramers_v_value < 0.20:
        interpretation = "Small"
    elif cramers_v_value < 0.30:
        interpretation = "Medium"
    else:
        interpretation = "Large"

    return {
        'cramers_v': cramers_v_value,
        'interpretation': interpretation
    }


# =============================================================================
# POST-HOC TESTS
# =============================================================================

def games_howell_test(age: np.ndarray, cluster_labels: np.ndarray) -> Dict:
    """
    Perform Games-Howell post-hoc test for pairwise cluster comparisons.

    Use when: Clusters have unequal variances (Levene's test p < 0.05).
    Advantage: Robust to heteroscedasticity and unequal sample sizes.

    Args:
        age: numpy array of shape (N,). Age for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.

    Returns:
        dict containing:
            - 'pairwise_comparisons': pandas DataFrame with pairwise results
    """
    # Use pingouin for proper Games-Howell implementation
    df = pd.DataFrame({'age': age, 'cluster': cluster_labels})

    # Run Games-Howell test
    result = pg.pairwise_gameshowell(data=df, dv='age', between='cluster')

    # Rename columns for consistency
    result = result.rename(columns={
        'A': 'cluster_1', 'B': 'cluster_2',
        'mean(A)': 'mean_1', 'mean(B)': 'mean_2',
        'diff': 'mean_diff', 'pval': 'p_value'
    })
    result['reject_null'] = result['p_value'] < 0.05

    return {'pairwise_comparisons': result}



def dunn_test(age: np.ndarray, cluster_labels: np.ndarray, p_adjust: str = 'bonferroni') -> Dict:
    """
    Perform Dunn's test for post-hoc pairwise comparisons after Kruskal-Wallis.

    Use when: Kruskal-Wallis test is significant (p < 0.05).
    Purpose: Identify which cluster pairs differ in age distribution.

    Args:
        age: numpy array of shape (N,). Age for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.
        p_adjust: Method for p-value adjustment ('bonferroni', 'holm', 'fdr_bh', etc.)

    Returns:
        dict containing:
            - 'pairwise_comparisons': pandas DataFrame with pairwise results
            - 'p_value_matrix': symmetric matrix of adjusted p-values
    """
    # Use scikit-posthocs for proper Dunn's test implementation
    df = pd.DataFrame({'age': age, 'cluster': cluster_labels})

    # Get p-value matrix with specified correction
    p_matrix = sp.posthoc_dunn(df, val_col='age', group_col='cluster', p_adjust=p_adjust)

    # Convert to pairwise comparison DataFrame
    unique_clusters = np.sort(np.unique(cluster_labels))
    comparisons = []

    for i, c1 in enumerate(unique_clusters):
        for c2 in unique_clusters[i+1:]:
            age1 = age[cluster_labels == c1]
            age2 = age[cluster_labels == c2]

            p_adj = p_matrix.loc[c1, c2]

            comparisons.append({
                'cluster_1': c1, 'cluster_2': c2,
                'median_1': np.median(age1), 'median_2': np.median(age2),
                'median_diff': np.median(age1) - np.median(age2),
                'p_adj': p_adj,
                'reject_null': p_adj < 0.05
            })

    return {
        'pairwise_comparisons': pd.DataFrame(comparisons),
        'p_value_matrix': p_matrix
    }



def fisher_exact_test(sex: np.ndarray, cluster_labels: np.ndarray) -> Dict:
    """
    Perform Fisher's exact test for sex-cluster association (small samples).

    Use when: Any cell in contingency table has expected count < 5.
    Alternative to: Chi-square test (which requires all cells >= 5).

    Args:
        sex: numpy array of shape (N,). Sex category (must have exactly 2 categories).
        cluster_labels: numpy array of shape (N,). Cluster assignment.

    Returns:
        dict containing test results and contingency table
    """
    from scipy.stats import fisher_exact

    contingency_table = pd.crosstab(cluster_labels, sex)

    if contingency_table.shape != (2, 2):
        return {
            'statistic': None, 'p_value': None,
            'contingency_table': contingency_table, 'valid': False,
            'message': "Fisher's exact test requires exactly 2 clusters and 2 sex categories"
        }

    statistic, p_value = fisher_exact(contingency_table.values)

    return {
        'statistic': statistic, 'p_value': p_value,
        'contingency_table': contingency_table, 'valid': True
    }


def print_games_howell_results(games_howell_result: Dict):
    """Pretty print Games-Howell post-hoc test results."""
    print("=" * 70)
    print("GAMES-HOWELL POST-HOC TEST: Which cluster pairs differ in age?")
    print("=" * 70)

    df = games_howell_result['pairwise_comparisons']

    for _, row in df.iterrows():
        c1, c2 = row['cluster_1'], row['cluster_2']
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"

        print(f"\nCluster {c1} vs. Cluster {c2}:")
        print(f"  Mean age: {row['mean_1']:.2f} vs {row['mean_2']:.2f}")
        print(f"  Mean difference: {row['mean_diff']:+.2f} years")
        print(f"  p-value: {row['p_value']:.4f} {sig}")
    print("=" * 70)


def print_dunn_results(dunn_result: Dict):
    """Pretty print Dunn's test results."""
    print("=" * 70)
    print("DUNN'S TEST: Post-hoc for Kruskal-Wallis")
    print("=" * 70)

    df = dunn_result['pairwise_comparisons']

    for _, row in df.iterrows():
        c1, c2 = row['cluster_1'], row['cluster_2']
        sig = "***" if row['p_adj'] < 0.001 else "**" if row['p_adj'] < 0.01 else "*" if row['p_adj'] < 0.05 else "ns"

        print(f"\nCluster {c1} vs. Cluster {c2}:")
        print(f"  Median age: {row['median_1']:.2f} vs {row['median_2']:.2f}")
        print(f"  Median difference: {row['median_diff']:+.2f} years")
        print(f"  p-adj: {row['p_adj']:.4f} {sig}")
    print("=" * 70)


def print_fisher_results(fisher_result: Dict):
    """Pretty print Fisher's exact test results."""
    print("=" * 70)
    print("FISHER'S EXACT TEST: Sex-cluster association (small samples)")
    print("=" * 70)

    if not fisher_result['valid']:
        print(f"WARNING: {fisher_result['message']}")
        print("Use chi_square_test() instead for larger tables.")
    else:
        print(f"Odds Ratio: {fisher_result['statistic']:.4f}")
        print(f"p-value: {fisher_result['p_value']:.4f}")

        if fisher_result['p_value'] < 0.001:
            print("Result: *** HIGHLY SIGNIFICANT (p < 0.001)")
        elif fisher_result['p_value'] < 0.01:
            print("Result: ** SIGNIFICANT (p < 0.01)")
        elif fisher_result['p_value'] < 0.05:
            print("Result: * SIGNIFICANT (p < 0.05)")
        else:
            print("Result: NOT SIGNIFICANT (p >= 0.05)")

        print("\nContingency Table:")
        print(fisher_result['contingency_table'])
    print("=" * 70)




# ========================================
# PLOTTING FUNCTIONS
# ========================================

def plot_age_distribution(age: np.ndarray,
                          cluster_labels: np.ndarray,
                          plot_type: str = 'violin',
                          save_path: Optional[str] = None,
                          show_plot: bool = False) -> plt.Figure:
    """
    Plot age distribution across clusters.

    Args:
        age: numpy array of shape (N,). Age for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.
        plot_type: Type of plot - 'violin', 'box', or 'both'
        save_path: Optional path to save the figure as PNG. If None, not saved.
        show_plot: Whether to display the plot (default: False)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_age_distribution(ages, clusters, plot_type='violin',
        ...                             save_path='age_dist.png')
    """
    # Create DataFrame
    df = pd.DataFrame({'Age': age, 'Cluster': cluster_labels})
    plt.ioff()

    # Prepare data for plotting
    unique_clusters = np.sort(np.unique(cluster_labels))
    age_by_cluster = [age[cluster_labels == c] for c in unique_clusters]

    if plot_type == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.set_style("whitegrid")
        # Violin plot
        sns.violinplot(data=df, x='Cluster', y='Age', hue='Cluster', ax=axes[0], palette='Set2', legend=False, )
        axes[0].set_title('Age Distribution by Cluster (Violin Plot)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Cluster', fontsize=12)
        axes[0].set_ylabel('Age (years)', fontsize=12)

        # Box plot
        sns.boxplot(data=df, x='Cluster', y='Age', hue='Cluster', ax=axes[1], palette='Set2', legend=False)
        axes[1].set_title('Age Distribution by Cluster (Box Plot)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Cluster', fontsize=12)
        axes[1].set_ylabel('Age (years)', fontsize=12)


    else:
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.set_style("whitegrid")
        if plot_type == 'violin':
            sns.violinplot(data=df, x='Cluster', y='Age', hue='Cluster', ax=ax, palette='Set2', legend=False)
            ax.set_title('Age Distribution by Cluster (Violin Plot)', fontsize=14, fontweight='bold')
        elif plot_type == 'box':
            sns.boxplot(data=df, x='Cluster', y='Age', hue='Cluster', ax=ax, palette='Set2', legend=False)
            ax.set_title('Age Distribution by Cluster (Box Plot)', fontsize=14, fontweight='bold')

        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Age (years)', fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.ion()
    return fig



def plot_sex_distribution(sex: np.ndarray,
                          cluster_labels: np.ndarray,
                          plot_type: str = 'count',
                          save_path: Optional[str] = None,
                          show_plot: bool = False) -> plt.Figure:
    """
    Plot sex distribution across clusters.

    Args:
        sex: numpy array of shape (N,). Sex category for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.
        plot_type: Type of plot - 'count' (counts), 'proportion' (percentages), or 'both'
        save_path: Optional path to save the figure as PNG. If None, not saved.
        show_plot: Whether to display the plot (default: True)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_sex_distribution(sex, clusters, plot_type='both',
        ...                             save_path='sex_dist.png')
    """
    # Create DataFrame
    df = pd.DataFrame({'Sex': sex, 'Cluster': cluster_labels})
    plt.ioff()

    if plot_type == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.set_style("whitegrid")
        # Count plot
        sns.countplot(data=df, x='Cluster', hue='Sex', ax=axes[0], palette='Set1')
        axes[0].set_title('Sex Distribution by Cluster (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Cluster', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].legend(title='Sex')

        # Proportion plot
        ct = pd.crosstab(df['Cluster'], df['Sex'], normalize='index') * 100
        ct.plot(kind='bar', stacked=False, ax=axes[1], color=['#e74c3c', '#3498db'])
        axes[1].set_title('Sex Distribution by Cluster (Percentages)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Cluster', fontsize=12)
        axes[1].set_ylabel('Percentage (%)', fontsize=12)
        axes[1].legend(title='Sex')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        axes[1].grid(True, alpha=0.3, axis='y')

    else:
        fig, ax = plt.subplots(figsize=(8, 6))

        if plot_type == 'count':
            sns.set_style("whitegrid")
            sns.countplot(data=df, x='Cluster', hue='Sex', ax=ax, palette='Set1')
            ax.set_title('Sex Distribution by Cluster (Counts)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12)

        elif plot_type == 'proportion':
            ct = pd.crosstab(df['Cluster'], df['Sex'], normalize='index') * 100
            ct.plot(kind='bar', stacked=False, ax=ax, color=['#e74c3c', '#3498db'])
            ax.set_title('Sex Distribution by Cluster (Percentages)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.grid(True, alpha=0.3, axis='y')

        ax.set_xlabel('Cluster', fontsize=12)
        ax.legend(title='Sex')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.ion()
    return fig


def plot_interaction(age: np.ndarray,
                     cluster_labels: np.ndarray,
                     sex: np.ndarray,
                     save_path: Optional[str] = None,
                     show_plot: bool = False) -> plt.Figure:
    """
    Plot interaction between cluster and sex on age (for two-way ANOVA).

    Args:
        age: numpy array of shape (N,). Age for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.
        sex: numpy array of shape (N,). Sex category.
        save_path: Optional path to save the figure as PNG. If None, not saved.
        show_plot: Whether to display the plot (default: True)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_interaction(ages, clusters, sex, save_path='interaction.png')
    """
    # Create DataFrame
    df = pd.DataFrame({'Age': age, 'Cluster': cluster_labels, 'Sex': sex})
    plt.ioff()
    # Calculate means for each cluster-sex combination
    means = df.groupby(['Cluster', 'Sex'])['Age'].mean().reset_index()

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines for each sex
    for sex_val in means['Sex'].unique():
        sex_data = means[means['Sex'] == sex_val]
        ax.plot(sex_data['Cluster'], sex_data['Age'], marker='o', linewidth=2,
                markersize=10, label=sex_val)

    ax.set_title('Cluster × Sex Interaction Effect on Age', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Mean Age (years)', fontsize=12)
    ax.legend(title='Sex', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()
    plt.ion()
    return fig


def plot_age_histograms(age: np.ndarray,
                        cluster_labels: np.ndarray,
                        save_path: Optional[str] = None,
                        show_plot: bool = False) -> plt.Figure:
    """
    Plot age histograms for each cluster separately.

    Args:
        age: numpy array of shape (N,). Age for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.
        save_path: Optional path to save the figure as PNG. If None, not saved.
        show_plot: Whether to display the plot (default: True)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_age_histograms(ages, clusters, save_path='age_hists.png')
    """
    unique_clusters = np.sort(np.unique(cluster_labels))
    n_clusters = len(unique_clusters)
    plt.ioff()
    # Determine subplot layout
    ncols = min(3, n_clusters)
    nrows = (n_clusters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))

    if n_clusters == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_clusters > 1 else [axes]

    for idx, cluster in enumerate(unique_clusters):
        cluster_ages = age[cluster_labels == cluster]

        axes[idx].hist(cluster_ages, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].axvline(np.mean(cluster_ages), color='red', linestyle='--', linewidth=2,
                          label=f'Mean = {np.mean(cluster_ages):.1f}')
        axes[idx].axvline(np.median(cluster_ages), color='green', linestyle='--', linewidth=2,
                          label=f'Median = {np.median(cluster_ages):.1f}')
        axes[idx].set_title(f'Cluster {cluster} (n={len(cluster_ages)})', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Age (years)', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Age Distribution per Cluster', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()
    plt.ioff()
    return fig


def plot_contingency_heatmap(sex: np.ndarray,
                             cluster_labels: np.ndarray,
                             show_values: str = 'both',
                             save_path: Optional[str] = None,
                             show_plot: bool = False) -> plt.Figure:
    """
    Plot heatmap of contingency table with standardized residuals.

    Args:
        sex: numpy array of shape (N,). Sex category for each participant.
        cluster_labels: numpy array of shape (N,). Cluster assignment.
        show_values: What to display - 'counts', 'residuals', or 'both'
        save_path: Optional path to save the figure as PNG. If None, not saved.
        show_plot: Whether to display the plot (default: True)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_contingency_heatmap(sex, clusters, show_values='both',
        ...                                save_path='contingency.png')
    """
    # Create contingency table
    contingency_table = pd.crosstab(cluster_labels, sex)
    plt.ioff()
    # Calculate chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    standardized_residuals = (contingency_table.values - expected) / np.sqrt(expected)

    if show_values == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Counts heatmap
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', ax=axes[0],
                    cbar_kws={'label': 'Count'})
        axes[0].set_title('Observed Counts', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Sex', fontsize=12)
        axes[0].set_ylabel('Cluster', fontsize=12)
        # Standardized residuals heatmap
        sns.heatmap(standardized_residuals, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, ax=axes[1], cbar_kws={'label': 'Standardized Residual'},
                    vmin=-3, vmax=3)
        axes[1].set_title(f'Standardized Residuals (χ²={chi2:.2f}, p={p_value:.4f})',
                          fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Sex', fontsize=12)
        axes[1].set_ylabel('Cluster', fontsize=12)


    else:
        fig, ax = plt.subplots(figsize=(8, 6))

        if show_values == 'counts':
            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', ax=ax,
                        cbar_kws={'label': 'Count'})
            ax.set_title(f'Contingency Table (χ²={chi2:.2f}, p={p_value:.4f})',
                         fontsize=14, fontweight='bold')
        elif show_values == 'residuals':
            sns.heatmap(standardized_residuals, annot=True, fmt='.2f', cmap='RdBu_r',
                        center=0, ax=ax, cbar_kws={'label': 'Standardized Residual'},
                        vmin=-3, vmax=3)
            ax.set_title(f'Standardized Residuals (χ²={chi2:.2f}, p={p_value:.4f})',
                         fontsize=14, fontweight='bold')

        ax.set_xlabel('Sex', fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()
    plt.ion()
    return fig


# ========================================
# HELPER FUNCTIONS
# ========================================

def print_chi_square_results(chi_square_result: Dict):
    """
    Pretty print chi-square test results.

    Args:
        chi_square_result: Result dictionary from chi_square_test()
    """
    print("=" * 60)
    print("CHI-SQUARE TEST: Is sex distribution independent of clusters?")
    print("=" * 60)
    print(f"χ² statistic: {chi_square_result['chi2_statistic']:.4f}")
    print(f"p-value: {chi_square_result['p_value']:.4f}")
    print(f"Degrees of freedom: {chi_square_result['dof']}")

    # Significance
    if chi_square_result['p_value'] < 0.001:
        print("Result: *** HIGHLY SIGNIFICANT (p < 0.001)")
    elif chi_square_result['p_value'] < 0.01:
        print("Result: ** SIGNIFICANT (p < 0.01)")
    elif chi_square_result['p_value'] < 0.05:
        print("Result: * SIGNIFICANT (p < 0.05)")
    else:
        print("Result: NOT SIGNIFICANT (p >= 0.05)")

    print("\nObserved Frequencies:")
    print(chi_square_result['observed_freq'])

    print("\nStandardized Residuals (|SR| > 2 indicates significant deviation):")
    print(chi_square_result['standardized_residuals'])
    print("=" * 60)


def print_effect_size(effect_size_result: Dict, test_type: str = "cramers_v"):
    """
    Pretty print effect size results.

    Args:
        effect_size_result: Result dictionary from cramers_v() or omega_squared()
        test_type: Either "cramers_v" or "omega_squared"
    """
    print("=" * 60)
    if test_type == "cramers_v":
        print("EFFECT SIZE: Cramér's V")
        print("=" * 60)
        print(f"Cramér's V: {effect_size_result['cramers_v']:.3f}")
        print(f"Interpretation: {effect_size_result['interpretation']}")
    elif test_type == "omega_squared":
        print("EFFECT SIZE: Omega-Squared (ω²)")
        print("=" * 60)
        print(f"ω²: {effect_size_result['omega_squared']:.3f}")
        print(f"Interpretation: {effect_size_result['interpretation']}")
    print("=" * 60)


def print_assumption_tests(levene_result: Dict = None,
                           shapiro_result: Dict = None):
    """
    Pretty print ANOVA assumption test results.

    Args:
        levene_result: Result dictionary from levene_test()
        shapiro_result: Result dictionary from shapiro_wilk_test()
    """
    print("=" * 60)
    print("ANOVA ASSUMPTION TESTS")
    print("=" * 60)

    if levene_result:
        print("\nLevene's Test (Homogeneity of Variance):")
        print(f"  Statistic: {levene_result['statistic']:.4f}")
        print(f"  p-value: {levene_result['p_value']:.4f}")
        print(f"  Homoscedastic: {'Yes' if levene_result['homoscedastic'] else 'No'}")
        print(f"  → {levene_result['recommendation']}")

    if shapiro_result:
        print("\nShapiro-Wilk Test (Normality per Cluster):")
        for cluster, result in shapiro_result['results'].items():
            if result['statistic'] is not None:
                print(f"  Cluster {cluster}: W={result['statistic']:.4f}, "
                      f"p={result['p_value']:.4f} "
                      f"({'Normal' if result['normal'] else 'Non-normal'})")
            else:
                print(f"  Cluster {cluster}: Insufficient samples (n < 3)")
        print(f"  → {shapiro_result['recommendation']}")

    print("=" * 60)


def print_kruskal_wallis_results(kw_result: Dict):
    """
    Pretty print Kruskal-Wallis test results.

    Args:
        kw_result: Result dictionary from kruskal_wallis_test()
    """
    print("=" * 60)
    print("KRUSKAL-WALLIS TEST: Non-parametric test for age differences")
    print("=" * 60)
    print(f"H-statistic: {kw_result['h_statistic']:.4f}")
    print(f"p-value: {kw_result['p_value']:.4f}")

    # Significance
    if kw_result['p_value'] < 0.001:
        print("Result: *** HIGHLY SIGNIFICANT (p < 0.001)")
    elif kw_result['p_value'] < 0.01:
        print("Result: ** SIGNIFICANT (p < 0.01)")
    elif kw_result['p_value'] < 0.05:
        print("Result: * SIGNIFICANT (p < 0.05)")
    else:
        print("Result: NOT SIGNIFICANT (p >= 0.05)")

    print("\nMedian age per cluster:")
    for cluster in sorted(kw_result['median_ages'].keys()):
        median = kw_result['median_ages'][cluster]
        mean_rank = kw_result['mean_ranks'][cluster]
        print(f"  Cluster {cluster}: Median = {median:.2f} years, Mean Rank = {mean_rank:.2f}")
    print("=" * 60)


def print_anova_results(anova_result: Dict, test_type: str = "one-way"):
    """
    Pretty print ANOVA results in a readable format.

    Args:
        anova_result: Result dictionary from one_way_anova() or two_way_anova()
        test_type: Either "one-way" or "two-way"
    """
    if test_type == "one-way":
        print("=" * 60)
        print("ONE-WAY ANOVA: Do clusters differ in mean age?")
        print("=" * 60)
        print(f"F-statistic: {anova_result['f_statistic']:.4f}")
        print(f"p-value: {anova_result['p_value']}")
        print(f"Degrees of freedom: ({anova_result['df_between']}, {anova_result['df_within']})")

        # Significance
        if anova_result['p_value'] < 0.001:
            print("Result: *** HIGHLY SIGNIFICANT (p < 0.001)")
        elif anova_result['p_value'] < 0.01:
            print("Result: ** SIGNIFICANT (p < 0.01)")
        elif anova_result['p_value'] < 0.05:
            print("Result: * SIGNIFICANT (p < 0.05)")
        else:
            print("Result: NOT SIGNIFICANT (p >= 0.05)")

        print("\nMean age per cluster:")
        for cluster in sorted(anova_result['mean_ages'].keys()):
            mean = anova_result['mean_ages'][cluster]
            std = anova_result['std_ages'][cluster]
            n = anova_result['n_per_cluster'][cluster]
            print(f"  Cluster {cluster}: {mean:.2f} ± {std:.2f} years (n={n})")
        print("=" * 60)

    elif test_type == "two-way":
        print("=" * 60)
        print("TWO-WAY ANOVA: Effects of cluster and sex on age")
        print("=" * 60)
        print("\nMain Effect of Cluster:")
        print(f"  F = {anova_result['cluster_effect']['f_statistic']:.4f}, "
              f"p = {anova_result['cluster_effect']['p_value']:.4f}")
        print(f"  {'SIGNIFICANT' if anova_result['cluster_effect']['p_value'] < 0.05 else 'NOT SIGNIFICANT'}")

        print("\nMain Effect of Sex:")
        print(f"  F = {anova_result['sex_effect']['f_statistic']:.4f}, "
              f"p = {anova_result['sex_effect']['p_value']:.4f}")
        print(f"  {'SIGNIFICANT' if anova_result['sex_effect']['p_value'] < 0.05 else 'NOT SIGNIFICANT'}")

        print("\nInteraction Effect (Cluster × Sex):")
        print(f"  F = {anova_result['interaction_effect']['f_statistic']:.4f}, "
              f"p = {anova_result['interaction_effect']['p_value']:.4f}")
        print(f"  {'SIGNIFICANT' if anova_result['interaction_effect']['p_value'] < 0.05 else 'NOT SIGNIFICANT'}")

        print("\n" + "=" * 60)
        print("Full ANOVA Table:")
        print("=" * 60)
        print(anova_result['summary_table'].to_string(index=False))
        print("=" * 60)


def load_external_variables(external_csv_path: str,
                            subject_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load age and sex data from external CSV file.

    Args:
        external_csv_path: Path to CSV file with columns: subject_id (index), age, sex
        subject_ids: Optional array of subject IDs to filter and order the data.
                    If provided, ensures age/sex align with your clustering data.

    Returns:
        Tuple of (age_array, sex_array) in the same order as subject_ids

    Example:
        >>> age, sex = load_external_variables('dataset/Dortmund_age&sex.csv', subject_ids)
    """
    df = pd.read_csv(external_csv_path, index_col=0)

    if subject_ids is not None:
        # Filter and reorder to match subject_ids
        df = df.loc[subject_ids]

    sex = df['sex'].values

    if 'age' in df.columns:
        age = df['age'].values
        return age, sex

    age_group = df['age_group'].values
    return age_group, sex




