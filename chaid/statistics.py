"""
Statistical functions for CHAID algorithm.

Chi-square test statistic:
    χ² = Σᵢ Σⱼ (nᵢⱼ - eᵢⱼ)² / eᵢⱼ
    
    where eᵢⱼ = (nᵢ· × n·ⱼ) / n  (expected frequency under H₀)

Degrees of freedom:
    ddl = (c-1)(d-1)

P-value (Section 3.2.4):
    p = P(χ²_ddl ≥ χ²_obs)

Bonferroni multipliers:
    - Nominal: m = S(c,g) where S is Stirling number of 2nd kind
    - Ordinal: m = C(c-1, g-1)
    - Floating: m = C(c-2, g-2) + g × C(c-2, g-1)
"""

import numpy as np
from scipy import stats
from scipy.special import comb
from typing import Tuple, Optional
from dataclasses import dataclass

from .types import PredictorType


@dataclass
class ChiSquareResult:
    """
    Result of a chi-square independence test.
    
    Attributes:
        statistic: The chi-square test statistic
        degrees_of_freedom: Degrees of freedom (c-1)(d-1)
        p_value: Raw p-value P(χ² ≥ χ²_obs)
        p_value_adjusted: Bonferroni-adjusted p-value (if applicable)
        expected_frequencies: Matrix of expected frequencies under H₀
        observed_frequencies: Matrix of observed frequencies
        bonferroni_multiplier: The multiplier used for adjustment (if applicable)
    """
    statistic: float
    degrees_of_freedom: int
    p_value: float
    p_value_adjusted: Optional[float]
    expected_frequencies: np.ndarray
    observed_frequencies: np.ndarray
    bonferroni_multiplier: Optional[float] = None


def compute_expected_frequencies(contingency_table: np.ndarray) -> np.ndarray:
    """
    Compute expected frequencies under H₀ (independence).
    
    Formula from main.tex Section 3.2.1:
        eᵢⱼ = (nᵢ· × n·ⱼ) / n
    
    Args:
        contingency_table: c × d matrix of observed frequencies
        
    Returns:
        c × d matrix of expected frequencies
    """
    row_sums = contingency_table.sum(axis=1, keepdims=True)  # nᵢ·
    col_sums = contingency_table.sum(axis=0, keepdims=True)  # n·ⱼ
    total = contingency_table.sum()  # n
    
    if total == 0:
        return np.zeros_like(contingency_table, dtype=float)
    
    expected = (row_sums * col_sums) / total
    return expected


def compute_chi_square_statistic(
    observed: np.ndarray, 
    expected: np.ndarray
) -> Tuple[float, int]:
    """
    Compute chi-square test statistic and degrees of freedom.
    
    Formula:
        χ² = Σᵢ Σⱼ (nᵢⱼ - eᵢⱼ)² / eᵢⱼ
    
    Degrees of freedom:
        ddl = (c-1)(d-1)
    
    Args:
        observed: c × d matrix of observed frequencies
        expected: c × d matrix of expected frequencies
        
    Returns:
        Tuple of (chi_square_statistic, degrees_of_freedom)
    """
    c, d = observed.shape
    
    # Handle zero expected frequencies (avoid division by zero)
    # Only include cells where expected > 0
    mask = expected > 0
    
    if not np.any(mask):
        return 0.0, max(0, (c - 1) * (d - 1))
    
    chi_sq = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
    
    # Degrees of freedom: (c-1)(d-1)
    # However, we need to adjust for rows/cols with zero totals
    non_zero_rows = np.sum(observed.sum(axis=1) > 0)
    non_zero_cols = np.sum(observed.sum(axis=0) > 0)
    
    ddl = max(0, (non_zero_rows - 1) * (non_zero_cols - 1))
    
    return float(chi_sq), ddl


def chi_square_test(
    contingency_table: np.ndarray,
    predictor_type: Optional[PredictorType] = None,
    c_original: Optional[int] = None,
    g_final: Optional[int] = None,
    apply_bonferroni: bool = False
) -> ChiSquareResult:
    """
    Perform chi-square test of independence on a contingency table.
    
    Args:
        contingency_table: c × d matrix of observed frequencies
        predictor_type: Type of predictor (for Bonferroni correction)
        c_original: Original number of categories before merging
        g_final: Final number of groups after merging
        apply_bonferroni: Whether to apply Bonferroni correction
        
    Returns:
        ChiSquareResult containing test statistic, p-value, etc.
    """
    contingency_table = np.atleast_2d(contingency_table).astype(float)
    
    # Compute expected frequencies
    expected = compute_expected_frequencies(contingency_table)
    
    # Compute chi-square statistic and degrees of freedom
    chi_sq, ddl = compute_chi_square_statistic(contingency_table, expected)
    
    # Compute p-value from chi-square distribution
    # p = P(χ²_ddl ≥ χ²_obs)
    if ddl > 0:
        p_value = 1.0 - stats.chi2.cdf(chi_sq, ddl)
    else:
        p_value = 1.0  # No degrees of freedom means no test possible
    
    # Apply Bonferroni correction if requested
    p_value_adjusted = None
    mult = None
    
    if apply_bonferroni and predictor_type is not None and c_original is not None and g_final is not None:
        mult = bonferroni_multiplier(c_original, g_final, predictor_type)
        p_value_adjusted = min(1.0, p_value * mult)
    
    return ChiSquareResult(
        statistic=chi_sq,
        degrees_of_freedom=ddl,
        p_value=p_value,
        p_value_adjusted=p_value_adjusted,
        expected_frequencies=expected,
        observed_frequencies=contingency_table,
        bonferroni_multiplier=mult
    )


def stirling_second_kind(n: int, k: int) -> int:
    """
    Compute Stirling number of the second kind S(n, k).
    
    S(n, k) = number of ways to partition n elements into k non-empty subsets.
    
    Formula from main.tex Section 4.1.1:
        S(c,g) = (1/g!) × Σᵢ₌₀^(g-1) (-1)ⁱ × C(g,i) × (g-i)^c
    
    Note: The formula in main.tex for the Bonferroni multiplier shows:
        m_nominal(c,g) = Σᵢ₌₀^(g-1) (-1)ⁱ × (g-i)^c / (i! × (g-i)!)
    
    This is equivalent to S(c,g) × g! / g! = S(c,g) when properly computed.
    
    Args:
        n: Number of elements
        k: Number of subsets
        
    Returns:
        Stirling number S(n, k)
    """
    if k <= 0 or n < k:
        return 0
    if k == 1 or k == n:
        return 1
    
    # Use recurrence relation: S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    # Build bottom-up for efficiency
    prev = [0] * (k + 1)
    prev[1] = 1
    
    for i in range(2, n + 1):
        curr = [0] * (k + 1)
        for j in range(1, min(i, k) + 1):
            curr[j] = j * prev[j] + prev[j - 1]
        prev = curr
    
    return prev[k]


def bonferroni_multiplier(
    c: int, 
    g: int, 
    predictor_type: PredictorType
) -> float:
    """
    Compute Bonferroni multiplier for p-value adjustment.
    
    The multiplier equals the number of ways to partition c categories
    into g groups, respecting the predictor type constraints.
    
    Formulas from main.tex Section 4.4.3:
    
    Nominal (free predictors):
        m = S(c,g) × g! = Σᵢ₌₀^(g-1) (-1)ⁱ × (g-i)^c / (i! × (g-i)!)
        
        Note: The formula shows S(c,g) but we need the number of ways
        to partition into g LABELED groups, which is S(c,g) since
        we are counting partitions not arrangements.
        
    Ordinal (monotonic predictors):
        m = C(c-1, g-1)
        
    Floating predictors:
        m = C(c-2, g-2) + g × C(c-2, g-1)
    
    Args:
        c: Original number of categories
        g: Final number of groups (after merging)
        predictor_type: Type of predictor
        
    Returns:
        Bonferroni multiplier
    """
    if g < 1 or c < g:
        return 1.0
    
    if g == 1:
        # Only one group = only one way
        return 1.0
    
    if predictor_type == PredictorType.NOMINAL:
        # Number of partitions of c elements into g non-empty subsets
        # This is S(c,g) - Stirling number of 2nd kind
        # But the formula in main.tex shows: Σ (-1)^i (g-i)^c / (i!(g-i)!)
        # which equals S(c,g)
        return float(stirling_second_kind(c, g))
    
    elif predictor_type == PredictorType.ORDINAL:
        # Number of ways to partition c ordered elements into g contiguous groups
        # = C(c-1, g-1)
        return float(comb(c - 1, g - 1, exact=True))
    
    elif predictor_type == PredictorType.FLOATING:
        # c-1 ordinal categories + 1 floating category
        # m = C(c-2, g-2) + g × C(c-2, g-1)
        if c < 2 or g < 2:
            return 1.0
        term1 = comb(c - 2, g - 2, exact=True) if g >= 2 else 0
        term2 = g * comb(c - 2, g - 1, exact=True) if g >= 1 else 0
        return float(term1 + term2)
    
    return 1.0


def build_contingency_table(
    X_groups: np.ndarray,
    Y: np.ndarray,
    y_categories: np.ndarray
) -> np.ndarray:
    """
    Build contingency table from grouped predictor values and dependent variable.
    
    Args:
        X_groups: Array of group labels for each observation
        Y: Array of dependent variable values
        y_categories: Unique categories of Y
        
    Returns:
        g × d contingency table
    """
    unique_groups = np.unique(X_groups[~np.isnan(X_groups.astype(float))])
    
    g = len(unique_groups)
    d = len(y_categories)
    
    table = np.zeros((g, d), dtype=int)
    
    group_to_idx = {grp: i for i, grp in enumerate(unique_groups)}
    cat_to_idx = {cat: j for j, cat in enumerate(y_categories)}
    
    for x_val, y_val in zip(X_groups, Y):
        if x_val in group_to_idx and y_val in cat_to_idx:
            table[group_to_idx[x_val], cat_to_idx[y_val]] += 1
    
    return table
