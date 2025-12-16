"""
Merging logic for CHAID algorithm.

The merging procedure:
1. Build contingency table for predictor × dependent variable
2. For each pair of categories (respecting predictor type constraints):
   - Compute chi-square for 2×d subtable
   - Find pair with minimum chi-square (maximum similarity)
3. If p-value > α_merge, merge the pair
4. Repeat until all pairs have significant differences

The degrees of freedom for pairwise comparison is d-1.
"""

import numpy as np
from typing import List, Tuple, FrozenSet, Optional, Set
from itertools import combinations

from .types import PredictorType
from .statistics import chi_square_test, ChiSquareResult
from .history import MergeRecord, SplitCheckRecord


def get_mergeable_pairs(
    groups: List[FrozenSet],
    predictor_type: PredictorType,
    ordered_categories: Optional[List] = None,
    floating_category: Optional[any] = None
) -> List[Tuple[int, int]]:
    """
    Get all pairs of groups that can be merged based on predictor type.
    
    Args:
        groups: Current list of category groups
        predictor_type: Type of predictor
        ordered_categories: For ordinal/floating, the ordered list of categories
        floating_category: For floating predictors, the floating category value
        
    Returns:
        List of (index1, index2) pairs that can be merged
    """
    n = len(groups)
    
    if predictor_type == PredictorType.NOMINAL:
        # Any pair can be merged
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    elif predictor_type == PredictorType.ORDINAL:
        # Only contiguous groups can be merged
        # Groups are assumed to be in order
        return [(i, i + 1) for i in range(n - 1)]
    
    elif predictor_type == PredictorType.FLOATING:
        # Ordinal categories can only merge with neighbors
        # Floating category can merge with anyone
        pairs = []
        
        # Find which group contains the floating category
        floating_group_idx = None
        for i, group in enumerate(groups):
            if floating_category in group:
                floating_group_idx = i
                break
        
        for i in range(n):
            for j in range(i + 1, n):
                # If either is the floating group, can merge
                if floating_group_idx is not None and (i == floating_group_idx or j == floating_group_idx):
                    pairs.append((i, j))
                # Otherwise, must be contiguous (ordinal rule)
                elif j == i + 1:
                    # Check if they are truly contiguous in the ordinal sense
                    # This requires that neither is the floating group
                    if floating_group_idx is None or (i != floating_group_idx and j != floating_group_idx):
                        pairs.append((i, j))
        
        return pairs
    
    return []


def compute_pairwise_chi_square(
    group1_data: np.ndarray,
    group2_data: np.ndarray,
    y_categories: np.ndarray
) -> ChiSquareResult:
    """
    Compute chi-square test for two category groups.
    
    This creates a 2×d contingency table and computes chi-square.
    Degrees of freedom = d-1 (per Kendall & Stuart 1961).
    
    Args:
        group1_data: Y values for observations in group 1
        group2_data: Y values for observations in group 2
        y_categories: All unique Y categories
        
    Returns:
        ChiSquareResult for the pairwise test
    """
    d = len(y_categories)
    cat_to_idx = {cat: j for j, cat in enumerate(y_categories)}
    
    # Build 2×d contingency table
    table = np.zeros((2, d), dtype=int)
    
    for y_val in group1_data:
        if y_val in cat_to_idx:
            table[0, cat_to_idx[y_val]] += 1
    
    for y_val in group2_data:
        if y_val in cat_to_idx:
            table[1, cat_to_idx[y_val]] += 1
    
    return chi_square_test(table)


def get_group_y_values(
    group: FrozenSet,
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    Get Y values for observations where X is in the given group.
    
    Args:
        group: Set of X category values
        X: Array of predictor values
        Y: Array of dependent variable values
        
    Returns:
        Array of Y values for matching observations
    """
    mask = np.isin(X, list(group))
    return Y[mask]


def merge_categories(
    X: np.ndarray,
    Y: np.ndarray,
    y_categories: np.ndarray,
    predictor_type: PredictorType,
    alpha_merge: float,
    ordered_categories: Optional[List] = None,
    floating_category: Optional[any] = None
) -> Tuple[List[FrozenSet], List[MergeRecord]]:
    """
    Perform category merging for a predictor.
    
    Algorithm from main.tex Section 4.2.2:
    1. Start with each category as its own group
    2. Find pair with minimum chi-square
    3. If p-value > alpha_merge, merge and repeat
    4. Stop when all pairs are significantly different
    
    Args:
        X: Predictor values
        Y: Dependent variable values
        y_categories: Unique Y categories
        predictor_type: Type of predictor
        alpha_merge: Significance threshold for merging
        ordered_categories: For ordinal/floating, the category order
        floating_category: For floating predictors, the floating category
        
    Returns:
        Tuple of (final_groups, merge_history)
    """
    # Initialize: each unique X value is its own group
    unique_x = sorted(set(X))
    
    # For ordinal predictors, maintain order
    if predictor_type in (PredictorType.ORDINAL, PredictorType.FLOATING):
        if ordered_categories is not None:
            # Filter to only those present in data
            unique_x = [c for c in ordered_categories if c in set(X)]
        # else keep sorted order
    
    groups: List[FrozenSet] = [frozenset([cat]) for cat in unique_x]
    merge_history: List[MergeRecord] = []
    
    # Iteratively merge until no more merges possible
    while len(groups) > 1:
        # Get all mergeable pairs
        pairs = get_mergeable_pairs(
            groups, predictor_type, 
            ordered_categories, floating_category
        )
        
        if not pairs:
            break
        
        # Find pair with minimum chi-square (maximum similarity)
        best_pair = None
        best_chi_sq = float('inf')
        best_result = None
        
        for i, j in pairs:
            group1_y = get_group_y_values(groups[i], X, Y)
            group2_y = get_group_y_values(groups[j], X, Y)
            
            # Skip if either group is empty
            if len(group1_y) == 0 or len(group2_y) == 0:
                continue
            
            result = compute_pairwise_chi_square(group1_y, group2_y, y_categories)
            
            if result.statistic < best_chi_sq:
                best_chi_sq = result.statistic
                best_pair = (i, j)
                best_result = result
        
        if best_pair is None or best_result is None:
            break
        
        i, j = best_pair
        categories_before = tuple(groups)
        merged_pair = (groups[i], groups[j])
        
        # Check if p-value exceeds merge threshold
        if best_result.p_value > alpha_merge:
            # Merge the groups
            new_group = groups[i] | groups[j]
            new_groups = [g for k, g in enumerate(groups) if k not in (i, j)]
            
            # Insert merged group at position of first group (maintain order for ordinal)
            insert_pos = min(i, j)
            new_groups.insert(insert_pos, new_group)
            
            groups = new_groups
            
            merge_history.append(MergeRecord(
                categories_before=categories_before,
                categories_after=tuple(groups),
                merged_pair=merged_pair,
                chi_square=best_result.statistic,
                degrees_of_freedom=best_result.degrees_of_freedom,
                p_value=best_result.p_value,
                was_merged=True,
                reason=f"p-value ({best_result.p_value:.4f}) > α_merge ({alpha_merge})"
            ))
        else:
            # Cannot merge - all remaining pairs are significant
            merge_history.append(MergeRecord(
                categories_before=categories_before,
                categories_after=categories_before,
                merged_pair=merged_pair,
                chi_square=best_result.statistic,
                degrees_of_freedom=best_result.degrees_of_freedom,
                p_value=best_result.p_value,
                was_merged=False,
                reason=f"p-value ({best_result.p_value:.4f}) ≤ α_merge ({alpha_merge}), stopping"
            ))
            break
    
    return groups, merge_history


def get_all_binary_splits(
    compound_group: FrozenSet,
    predictor_type: PredictorType,
    ordered_categories: Optional[List] = None,
    floating_category: Optional[any] = None
) -> List[Tuple[FrozenSet, FrozenSet]]:
    """
    Get all valid binary splits of a compound group.
    
    Args:
        compound_group: The group to split
        predictor_type: Type of predictor
        ordered_categories: For ordinal/floating, the category order
        floating_category: For floating predictors, the floating category
        
    Returns:
        List of (group1, group2) tuples representing valid splits
    """
    categories = list(compound_group)
    n = len(categories)
    
    if n < 2:
        return []
    
    if predictor_type == PredictorType.NOMINAL:
        # Any binary partition is valid
        splits = []
        for r in range(1, n):
            for subset in combinations(categories, r):
                group1 = frozenset(subset)
                group2 = compound_group - group1
                if group1 and group2:
                    # Avoid duplicates (A|B same as B|A)
                    if (group2, group1) not in splits:
                        splits.append((group1, group2))
        return splits
    
    elif predictor_type == PredictorType.ORDINAL:
        # Only contiguous splits are valid
        # Categories must be sorted by their order
        if ordered_categories:
            sorted_cats = [c for c in ordered_categories if c in compound_group]
        else:
            sorted_cats = sorted(categories)
        
        splits = []
        for i in range(1, len(sorted_cats)):
            group1 = frozenset(sorted_cats[:i])
            group2 = frozenset(sorted_cats[i:])
            splits.append((group1, group2))
        return splits
    
    elif predictor_type == PredictorType.FLOATING:
        # Similar to ordinal, but floating category can go anywhere
        ordinal_cats = [c for c in compound_group if c != floating_category]
        has_floating = floating_category in compound_group
        
        if ordered_categories:
            sorted_ordinal = [c for c in ordered_categories if c in ordinal_cats]
        else:
            sorted_ordinal = sorted(ordinal_cats)
        
        splits = []
        
        # Contiguous splits of ordinal categories
        for i in range(1, len(sorted_ordinal)):
            group1 = frozenset(sorted_ordinal[:i])
            group2 = frozenset(sorted_ordinal[i:])
            
            if has_floating:
                # Floating can go to either side
                splits.append((group1 | {floating_category}, group2))
                splits.append((group1, group2 | {floating_category}))
            else:
                splits.append((group1, group2))
        
        # Floating alone vs rest
        if has_floating and len(ordinal_cats) > 0:
            splits.append((frozenset([floating_category]), frozenset(ordinal_cats)))
        
        return splits
    
    return []


def check_splits(
    groups: List[FrozenSet],
    X: np.ndarray,
    Y: np.ndarray,
    y_categories: np.ndarray,
    predictor_type: PredictorType,
    alpha_split: float,
    ordered_categories: Optional[List] = None,
    floating_category: Optional[any] = None
) -> Tuple[List[FrozenSet], List[SplitCheckRecord], bool]:
    """
    Check if any compound group should be split (Step 2 of CHAID).
    
    From main.tex Section 4.2.3:
    For each compound group with ≥3 original categories:
    1. Examine all valid dichotomies
    2. Find the one with maximum chi-square
    3. If p-value < alpha_split, implement the split
    
    Args:
        groups: Current groups
        X: Predictor values
        Y: Dependent variable values
        y_categories: Unique Y categories
        predictor_type: Type of predictor
        alpha_split: Significance threshold for splitting
        ordered_categories: For ordinal/floating, the category order
        floating_category: For floating predictors, the floating category
        
    Returns:
        Tuple of (updated_groups, split_history, any_splits_made)
    """
    split_history: List[SplitCheckRecord] = []
    any_splits = False
    
    # Check each compound group (≥3 original categories)
    i = 0
    while i < len(groups):
        group = groups[i]
        
        if len(group) < 3:
            i += 1
            continue
        
        # Get all valid binary splits
        all_splits = get_all_binary_splits(
            group, predictor_type,
            ordered_categories, floating_category
        )
        
        if not all_splits:
            i += 1
            continue
        
        # Find split with maximum chi-square
        best_split = None
        best_chi_sq = -1.0
        best_result = None
        
        for split_group1, split_group2 in all_splits:
            group1_y = get_group_y_values(split_group1, X, Y)
            group2_y = get_group_y_values(split_group2, X, Y)
            
            if len(group1_y) == 0 or len(group2_y) == 0:
                continue
            
            result = compute_pairwise_chi_square(group1_y, group2_y, y_categories)
            
            if result.statistic > best_chi_sq:
                best_chi_sq = result.statistic
                best_split = (split_group1, split_group2)
                best_result = result
        
        if best_split is not None and best_result is not None:
            if best_result.p_value < alpha_split:
                # Implement the split
                new_groups = groups[:i] + list(best_split) + groups[i+1:]
                groups = new_groups
                any_splits = True
                
                split_history.append(SplitCheckRecord(
                    compound_category=group,
                    best_split=best_split,
                    chi_square=best_result.statistic,
                    degrees_of_freedom=best_result.degrees_of_freedom,
                    p_value=best_result.p_value,
                    was_split=True,
                    reason=f"p-value ({best_result.p_value:.4f}) < α_split ({alpha_split})"
                ))
                
                # Don't increment i - need to check the new groups
            else:
                split_history.append(SplitCheckRecord(
                    compound_category=group,
                    best_split=best_split,
                    chi_square=best_result.statistic,
                    degrees_of_freedom=best_result.degrees_of_freedom,
                    p_value=best_result.p_value,
                    was_split=False,
                    reason=f"p-value ({best_result.p_value:.4f}) ≥ α_split ({alpha_split})"
                ))
                i += 1
        else:
            i += 1
    
    return groups, split_history, any_splits


def optimize_predictor(
    X: np.ndarray,
    Y: np.ndarray,
    y_categories: np.ndarray,
    predictor_type: PredictorType,
    alpha_merge: float,
    alpha_split: float,
    ordered_categories: Optional[List] = None,
    floating_category: Optional[any] = None
) -> Tuple[List[FrozenSet], List[MergeRecord], List[SplitCheckRecord]]:
    """
    Find optimal category grouping for a predictor.
    
    This implements the full merge-split cycle until convergence.
    Per main.tex, alpha_split < alpha_merge ensures convergence.
    
    Args:
        X: Predictor values
        Y: Dependent variable values
        y_categories: Unique Y categories
        predictor_type: Type of predictor
        alpha_merge: Threshold for merging
        alpha_split: Threshold for splitting (must be < alpha_merge)
        ordered_categories: For ordinal/floating, the category order
        floating_category: For floating predictors, the floating category
        
    Returns:
        Tuple of (final_groups, merge_history, split_history)
    """
    all_merge_history: List[MergeRecord] = []
    all_split_history: List[SplitCheckRecord] = []
    
    # Step 1: Merge categories
    groups, merge_history = merge_categories(
        X, Y, y_categories, predictor_type,
        alpha_merge, ordered_categories, floating_category
    )
    all_merge_history.extend(merge_history)
    
    # Step 2: Check splits and iterate if needed
    max_iterations = 100  # Safety limit
    iteration = 0
    
    while iteration < max_iterations:
        groups, split_history, any_splits = check_splits(
            groups, X, Y, y_categories, predictor_type,
            alpha_split, ordered_categories, floating_category
        )
        all_split_history.extend(split_history)
        
        if not any_splits:
            break
        
        # If splits were made, need to re-merge
        groups, merge_history = merge_categories(
            X, Y, y_categories, predictor_type,
            alpha_merge, ordered_categories, floating_category
        )
        all_merge_history.extend(merge_history)
        
        iteration += 1
    
    return groups, all_merge_history, all_split_history
