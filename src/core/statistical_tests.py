"""
Statistical testing utilities for CEI benchmark analysis.

Provides statistical tests for comparing model performance:
- Paired bootstrap tests for performance differences
- Chi-squared tests for classification disparities
- Multiple comparison corrections (Bonferroni, FDR)
- Effect size calculations

Example:
    >>> from core.statistical_tests import paired_bootstrap_test
    >>> diff, p_value, ci = paired_bootstrap_test(scores_a, scores_b)
    >>> print(f"Difference: {diff:.3f}, p={p_value:.4f}, 95% CI: {ci}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import stats


@dataclass
class BootstrapResult:
    """Result of a bootstrap test."""

    observed_diff: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_resamples: int

    @property
    def ci(self) -> Tuple[float, float]:
        """Return confidence interval as tuple."""
        return (self.ci_lower, self.ci_upper)

    @property
    def is_significant(self) -> bool:
        """Check if result is significant at p < 0.05."""
        return self.p_value < 0.05

    def __str__(self) -> str:
        """Human-readable representation."""
        sig = "*" if self.is_significant else ""
        return (
            f"diff={self.observed_diff:.4f}, p={self.p_value:.4f}{sig}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> BootstrapResult:
    """
    Perform paired bootstrap test for difference in means.

    Tests whether scores_a and scores_b have significantly different means
    using a paired bootstrap procedure.

    Args:
        scores_a: Scores for condition A (n_samples,)
        scores_b: Scores for condition B (n_samples,)
        n_resamples: Number of bootstrap resamples
        confidence_level: Confidence level for CI (default 0.95)
        random_state: Random seed for reproducibility

    Returns:
        BootstrapResult with difference, p-value, and CI
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired test requires equal-length arrays")

    rng = np.random.default_rng(random_state)
    n_samples = len(scores_a)

    # Observed difference
    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # Bootstrap resampling
    bootstrap_diffs = np.zeros(n_resamples)
    for i in range(n_resamples):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_diffs[i] = np.mean(scores_a[indices]) - np.mean(scores_b[indices])

    # Two-sided p-value
    # Count how often bootstrap distribution is more extreme than observed
    centered_diffs = bootstrap_diffs - np.mean(bootstrap_diffs)
    p_value = np.mean(np.abs(centered_diffs) >= np.abs(observed_diff))

    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return BootstrapResult(
        observed_diff=float(observed_diff),
        p_value=float(p_value),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_resamples=n_resamples,
    )


def bootstrap_confidence_interval(
    scores: np.ndarray,
    statistic: str = "mean",
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        scores: Sample scores
        statistic: Statistic to compute ("mean", "median", "std")
        n_resamples: Number of bootstrap resamples
        confidence_level: Confidence level for CI
        random_state: Random seed

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(scores)

    stat_funcs = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
    }

    if statistic not in stat_funcs:
        raise ValueError(f"Unknown statistic: {statistic}")

    stat_func = stat_funcs[statistic]
    point_estimate = stat_func(scores)

    # Bootstrap
    bootstrap_stats = np.zeros(n_resamples)
    for i in range(n_resamples):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_stats[i] = stat_func(scores[indices])

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(point_estimate), float(ci_lower), float(ci_upper)


@dataclass
class ChiSquaredResult:
    """Result of a chi-squared test."""

    chi2_statistic: float
    p_value: float
    dof: int
    expected_frequencies: np.ndarray
    cramers_v: float

    @property
    def is_significant(self) -> bool:
        """Check if result is significant at p < 0.05."""
        return self.p_value < 0.05

    def __str__(self) -> str:
        """Human-readable representation."""
        sig = "*" if self.is_significant else ""
        return f"chi2={self.chi2_statistic:.2f}, dof={self.dof}, p={self.p_value:.4f}{sig}, V={self.cramers_v:.3f}"


def chi_squared_test(
    observed: np.ndarray,
) -> ChiSquaredResult:
    """
    Perform chi-squared test for independence.

    Args:
        observed: Contingency table of observed frequencies

    Returns:
        ChiSquaredResult with test statistics
    """
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    # Calculate Cramer's V for effect size
    n = observed.sum()
    min_dim = min(observed.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

    return ChiSquaredResult(
        chi2_statistic=float(chi2),
        p_value=float(p_value),
        dof=int(dof),
        expected_frequencies=expected,
        cramers_v=float(cramers_v),
    )


def bonferroni_correction(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: Array of p-values
        alpha: Family-wise error rate

    Returns:
        Tuple of (adjusted_p_values, corrected_alpha)
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    corrected_alpha = alpha / n_tests
    adjusted_p_values = np.minimum(p_values * n_tests, 1.0)

    return adjusted_p_values, corrected_alpha


def benjamini_hochberg_correction(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values
        alpha: False discovery rate

    Returns:
        Tuple of (adjusted_p_values, significant_mask)
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    # Sort p-values and get ranks
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    ranks = np.arange(1, n_tests + 1)

    # Calculate adjusted p-values
    adjusted = sorted_p * n_tests / ranks

    # Ensure monotonicity
    for i in range(n_tests - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    adjusted = np.minimum(adjusted, 1.0)

    # Unsort
    unsorted_adjusted = np.zeros_like(adjusted)
    unsorted_adjusted[sorted_indices] = adjusted

    # Significance mask
    significant = unsorted_adjusted < alpha

    return unsorted_adjusted, significant


def cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
    pooled: bool = True,
) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        group_a: Scores for group A
        group_b: Scores for group B
        pooled: Whether to use pooled standard deviation

    Returns:
        Cohen's d effect size
    """
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)

    if pooled:
        n_a, n_b = len(group_a), len(group_b)
        var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        return (mean_a - mean_b) / pooled_std
    else:
        std_a = np.std(group_a, ddof=1)
        return (mean_a - mean_b) / std_a


def cliffs_delta(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """
    Calculate Cliff's delta effect size (non-parametric).

    Args:
        group_a: Scores for group A
        group_b: Scores for group B

    Returns:
        Cliff's delta (-1 to 1)
    """
    n_a, n_b = len(group_a), len(group_b)

    # Count dominance pairs
    greater = 0
    less = 0

    for a in group_a:
        for b in group_b:
            if a > b:
                greater += 1
            elif a < b:
                less += 1

    return (greater - less) / (n_a * n_b)


def interpret_effect_size(d: float, measure: str = "cohens_d") -> str:
    """
    Interpret effect size magnitude.

    Args:
        d: Effect size value
        measure: Type of effect size ("cohens_d" or "cliffs_delta")

    Returns:
        Interpretation string
    """
    abs_d = abs(d)

    if measure == "cohens_d":
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    elif measure == "cliffs_delta":
        if abs_d < 0.147:
            return "negligible"
        elif abs_d < 0.33:
            return "small"
        elif abs_d < 0.474:
            return "medium"
        else:
            return "large"
    else:
        raise ValueError(f"Unknown measure: {measure}")


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = 10000,
    statistic: str = "mean_diff",
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Perform permutation test for group differences.

    Args:
        group_a: Scores for group A
        group_b: Scores for group B
        n_permutations: Number of permutations
        statistic: Statistic to compute ("mean_diff", "median_diff")
        random_state: Random seed

    Returns:
        Tuple of (observed_statistic, p_value)
    """
    rng = np.random.default_rng(random_state)

    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    if statistic == "mean_diff":
        stat_func = lambda a, b: np.mean(a) - np.mean(b)
    elif statistic == "median_diff":
        stat_func = lambda a, b: np.median(a) - np.median(b)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    observed = stat_func(group_a, group_b)

    # Permutation distribution
    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_stat = stat_func(perm_a, perm_b)
        if abs(perm_stat) >= abs(observed):
            count_extreme += 1

    p_value = count_extreme / n_permutations

    return float(observed), float(p_value)


@dataclass
class ChiSquareGoodnessOfFitResult:
    """Result of chi-square goodness of fit test."""

    chi2_statistic: float
    p_value: float
    dof: int
    observed: np.ndarray
    expected: np.ndarray
    cramers_v: float

    @property
    def is_significant(self) -> bool:
        """Check if result is significant at p < 0.05."""
        return self.p_value < 0.05

    def __str__(self) -> str:
        """Human-readable representation."""
        sig = "*" if self.is_significant else ""
        return f"chi2={self.chi2_statistic:.2f}, dof={self.dof}, p={self.p_value:.4f}{sig}, V={self.cramers_v:.3f}"


def chi_square_goodness_of_fit(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
) -> ChiSquareGoodnessOfFitResult:
    """
    Perform chi-squared goodness-of-fit test.

    Tests whether observed frequencies differ from expected frequencies.
    If expected is not provided, assumes uniform distribution.

    Args:
        observed: Observed frequency counts
        expected: Expected frequency counts (optional, defaults to uniform)

    Returns:
        ChiSquareGoodnessOfFitResult with test statistics
    """
    observed = np.asarray(observed)

    if expected is None:
        # Uniform distribution
        n = observed.sum()
        expected = np.full_like(observed, n / len(observed), dtype=float)
    else:
        expected = np.asarray(expected)

    # Chi-square statistic
    chi2 = np.sum((observed - expected) ** 2 / expected)
    dof = len(observed) - 1
    p_value = 1 - stats.chi2.cdf(chi2, dof)

    # Cramer's V effect size
    n = observed.sum()
    cramers_v = np.sqrt(chi2 / n) if n > 0 else 0.0

    return ChiSquareGoodnessOfFitResult(
        chi2_statistic=float(chi2),
        p_value=float(p_value),
        dof=int(dof),
        observed=observed,
        expected=expected,
        cramers_v=float(cramers_v),
    )


@dataclass
class FleissKappaResult:
    """Result of Fleiss' kappa calculation."""

    kappa: float
    observed_agreement: float
    expected_agreement: float
    n_subjects: int
    n_raters: int
    n_categories: int
    interpretation: str

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"kappa={self.kappa:.3f} ({self.interpretation})"


def fleiss_kappa(
    ratings: np.ndarray,
) -> FleissKappaResult:
    """
    Calculate Fleiss' kappa for inter-rater agreement.

    Args:
        ratings: Matrix of shape (n_subjects, n_categories) where each cell
                contains the number of raters who assigned that category
                to that subject.

    Returns:
        FleissKappaResult with kappa and agreement statistics.

    Example:
        >>> # 10 subjects, 3 categories, 5 raters each
        >>> ratings = np.array([
        ...     [0, 0, 5],  # All 5 raters chose category 3
        ...     [3, 2, 0],  # 3 chose cat 1, 2 chose cat 2
        ...     [1, 1, 3],  # Split decision
        ...     ...
        ... ])
        >>> result = fleiss_kappa(ratings)
    """
    ratings = np.asarray(ratings)
    n_subjects, n_categories = ratings.shape
    n_raters = ratings[0].sum()  # Number of raters per subject

    # Proportion of assignments to each category
    p_j = ratings.sum(axis=0) / (n_subjects * n_raters)

    # Expected agreement (by chance)
    P_e = np.sum(p_j ** 2)

    # Observed agreement per subject
    P_i = (np.sum(ratings ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    # Mean observed agreement
    P_o = np.mean(P_i)

    # Fleiss' kappa
    if P_e == 1:
        kappa = 1.0
    else:
        kappa = (P_o - P_e) / (1 - P_e)

    # Interpretation
    if kappa < 0:
        interpretation = "poor (below chance)"
    elif kappa < 0.2:
        interpretation = "slight"
    elif kappa < 0.4:
        interpretation = "fair"
    elif kappa < 0.6:
        interpretation = "moderate"
    elif kappa < 0.8:
        interpretation = "substantial"
    else:
        interpretation = "almost perfect"

    return FleissKappaResult(
        kappa=float(kappa),
        observed_agreement=float(P_o),
        expected_agreement=float(P_e),
        n_subjects=n_subjects,
        n_raters=int(n_raters),
        n_categories=n_categories,
        interpretation=interpretation,
    )
