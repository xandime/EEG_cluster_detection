import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


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
        'cluster': cluster_labels.astype(str),  # treat as categorical
        'sex': sex
    })

    # Perform two-way ANOVA using scipy (OLS approach)
    # Main effect of cluster (controlling for sex)
    unique_clusters = np.unique(cluster_labels)
    unique_sex = np.unique(sex)

    # Grand mean
    grand_mean = np.mean(age)
    n = len(age)

    # Sum of squares calculation
    # Total SS
    ss_total = np.sum((age - grand_mean) ** 2)

    # Cluster main effect SS
    ss_cluster = 0
    for c in unique_clusters:
        mask = cluster_labels == c
        n_c = np.sum(mask)
        mean_c = np.mean(age[mask])
        ss_cluster += n_c * (mean_c - grand_mean) ** 2

    # Sex main effect SS
    ss_sex = 0
    for s in unique_sex:
        mask = sex == s
        n_s = np.sum(mask)
        mean_s = np.mean(age[mask])
        ss_sex += n_s * (mean_s - grand_mean) ** 2

    # Interaction SS (cell means - row means - col means + grand mean)
    ss_interaction = 0
    for c in unique_clusters:
        for s in unique_sex:
            mask = (cluster_labels == c) & (sex == s)
            n_cs = np.sum(mask)
            if n_cs > 0:
                mean_cs = np.mean(age[mask])
                mean_c = np.mean(age[cluster_labels == c])
                mean_s = np.mean(age[sex == s])
                ss_interaction += n_cs * (mean_cs - mean_c - mean_s + grand_mean) ** 2

    # Residual SS
    ss_residual = ss_total - ss_cluster - ss_sex - ss_interaction

    # Degrees of freedom
    df_cluster = len(unique_clusters) - 1
    df_sex = len(unique_sex) - 1
    df_interaction = df_cluster * df_sex
    df_residual = n - (len(unique_clusters) * len(unique_sex))

    # Mean squares
    ms_cluster = ss_cluster / df_cluster
    ms_sex = ss_sex / df_sex
    ms_interaction = ss_interaction / df_interaction
    ms_residual = ss_residual / df_residual

    # F-statistics
    f_cluster = ms_cluster / ms_residual
    f_sex = ms_sex / ms_residual
    f_interaction = ms_interaction / ms_residual

    # P-values
    p_cluster = 1 - stats.f.cdf(f_cluster, df_cluster, df_residual)
    p_sex = 1 - stats.f.cdf(f_sex, df_sex, df_residual)
    p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_residual)

    # Create summary table
    summary_table = pd.DataFrame({
        'Source': ['Cluster', 'Sex', 'Cluster × Sex', 'Residual', 'Total'],
        'SS': [ss_cluster, ss_sex, ss_interaction, ss_residual, ss_total],
        'df': [df_cluster, df_sex, df_interaction, df_residual, n - 1],
        'MS': [ms_cluster, ms_sex, ms_interaction, ms_residual, np.nan],
        'F': [f_cluster, f_sex, f_interaction, np.nan, np.nan],
        'p-value': [p_cluster, p_sex, p_interaction, np.nan, np.nan]
    })

    return {
        'cluster_effect': {
            'f_statistic': f_cluster,
            'p_value': p_cluster,
            'df': (df_cluster, df_residual)
        },
        'sex_effect': {
            'f_statistic': f_sex,
            'p_value': p_sex,
            'df': (df_sex, df_residual)
        },
        'interaction_effect': {
            'f_statistic': f_interaction,
            'p_value': p_interaction,
            'df': (df_interaction, df_residual)
        },
        'summary_table': summary_table
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

    age = df['age'].values
    sex = df['sex'].values

    return age, sex
