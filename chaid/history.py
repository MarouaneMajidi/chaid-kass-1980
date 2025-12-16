"""
History tracking for CHAID algorithm.

This module provides data structures for complete traceability of:
- Every tested split
- Categories before and after merging
- Test statistics, degrees of freedom, p-values
- Selected vs rejected splits
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, FrozenSet
import numpy as np

from .types import PredictorType


@dataclass
class MergeRecord:
    """
    Record of a single merge operation during category fusion.
    
    Attributes:
        categories_before: Tuple of category groups before merge
        categories_after: Tuple of category groups after merge
        merged_pair: The two groups that were merged
        chi_square: Chi-square statistic for the pair comparison
        degrees_of_freedom: Degrees of freedom for the test
        p_value: P-value of the pair comparison
        was_merged: Whether the merge was actually performed
        reason: Reason for merging or not merging
    """
    categories_before: Tuple[FrozenSet, ...]
    categories_after: Tuple[FrozenSet, ...]
    merged_pair: Tuple[FrozenSet, FrozenSet]
    chi_square: float
    degrees_of_freedom: int
    p_value: float
    was_merged: bool
    reason: str


@dataclass
class SplitCheckRecord:
    """
    Record of a split check operation (verifying if merged groups should be split).
    
    Attributes:
        compound_category: The compound category being checked
        best_split: The best binary split found
        chi_square: Chi-square statistic for the best split
        degrees_of_freedom: Degrees of freedom
        p_value: P-value of the best split
        was_split: Whether the split was implemented
        reason: Reason for splitting or not splitting
    """
    compound_category: FrozenSet
    best_split: Optional[Tuple[FrozenSet, FrozenSet]]
    chi_square: float
    degrees_of_freedom: int
    p_value: float
    was_split: bool
    reason: str


@dataclass
class PredictorEvaluation:
    """
    Complete evaluation record for a predictor at a node.
    
    Attributes:
        predictor_name: Name of the predictor variable
        predictor_type: Type of predictor (nominal, ordinal, floating)
        original_categories: Original categories of the predictor
        final_groups: Final groups after merging
        merge_history: List of all merge operations
        split_check_history: List of all split check operations
        final_chi_square: Chi-square for the final grouping
        final_degrees_of_freedom: Degrees of freedom
        final_p_value: Raw p-value
        bonferroni_multiplier: Bonferroni multiplier used
        adjusted_p_value: Bonferroni-adjusted p-value
        was_selected: Whether this predictor was selected for splitting
    """
    predictor_name: str
    predictor_type: PredictorType
    original_categories: Tuple
    final_groups: Tuple[FrozenSet, ...]
    merge_history: List[MergeRecord] = field(default_factory=list)
    split_check_history: List[SplitCheckRecord] = field(default_factory=list)
    final_chi_square: float = 0.0
    final_degrees_of_freedom: int = 0
    final_p_value: float = 1.0
    bonferroni_multiplier: Optional[float] = None
    adjusted_p_value: Optional[float] = None
    was_selected: bool = False


@dataclass
class NodeHistory:
    """
    Complete history of all operations at a node.
    
    Attributes:
        node_id: Unique identifier for the node
        depth: Depth in the tree (root = 0)
        n_observations: Number of observations at this node
        predictor_evaluations: Evaluation records for all predictors
        selected_predictor: Name of the selected predictor (or None if leaf)
        selected_groups: Final groups used for splitting
        reason_for_selection: Why this predictor was selected
        is_leaf: Whether this node is a terminal node
        leaf_reason: If leaf, why it became a leaf
    """
    node_id: int
    depth: int
    n_observations: int
    predictor_evaluations: Dict[str, PredictorEvaluation] = field(default_factory=dict)
    selected_predictor: Optional[str] = None
    selected_groups: Optional[Tuple[FrozenSet, ...]] = None
    reason_for_selection: str = ""
    is_leaf: bool = False
    leaf_reason: str = ""
    
    def get_summary(self) -> str:
        """Generate human-readable summary of node history."""
        lines = []
        lines.append(f"=" * 60)
        lines.append(f"Node {self.node_id} (depth={self.depth}, n={self.n_observations})")
        lines.append(f"=" * 60)
        
        if self.is_leaf:
            lines.append(f"TERMINAL NODE: {self.leaf_reason}")
        else:
            lines.append(f"Split on: {self.selected_predictor}")
            lines.append(f"Reason: {self.reason_for_selection}")
            if self.selected_groups:
                lines.append(f"Groups: {[set(g) for g in self.selected_groups]}")
        
        lines.append("")
        lines.append("Predictor Evaluations:")
        lines.append("-" * 40)
        
        for name, eval_rec in self.predictor_evaluations.items():
            marker = " [SELECTED]" if eval_rec.was_selected else ""
            lines.append(f"\n  {name} ({eval_rec.predictor_type.value}){marker}")
            lines.append(f"    Original categories: {list(eval_rec.original_categories)}")
            lines.append(f"    Final groups: {[set(g) for g in eval_rec.final_groups]}")
            lines.append(f"    Chi-square: {eval_rec.final_chi_square:.4f}")
            lines.append(f"    DF: {eval_rec.final_degrees_of_freedom}")
            lines.append(f"    p-value (raw): {eval_rec.final_p_value:.6f}")
            if eval_rec.bonferroni_multiplier is not None:
                lines.append(f"    Bonferroni multiplier: {eval_rec.bonferroni_multiplier:.1f}")
                lines.append(f"    p-value (adjusted): {eval_rec.adjusted_p_value:.6f}")
            
            if eval_rec.merge_history:
                lines.append(f"    Merge history ({len(eval_rec.merge_history)} operations):")
                for i, merge in enumerate(eval_rec.merge_history):
                    status = "MERGED" if merge.was_merged else "NOT MERGED"
                    lines.append(f"      {i+1}. {status}: χ²={merge.chi_square:.4f}, p={merge.p_value:.4f}")
                    lines.append(f"         Pair: {set(merge.merged_pair[0])} + {set(merge.merged_pair[1])}")
        
        return "\n".join(lines)


@dataclass
class TreeHistory:
    """
    Complete history of the entire tree construction.
    
    Attributes:
        node_histories: Dictionary mapping node_id to NodeHistory
        construction_order: Order in which nodes were processed
        parameters: Parameters used for tree construction
    """
    node_histories: Dict[int, NodeHistory] = field(default_factory=dict)
    construction_order: List[int] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def add_node_history(self, node_history: NodeHistory):
        """Add a node history record."""
        self.node_histories[node_history.node_id] = node_history
        self.construction_order.append(node_history.node_id)
    
    def get_node_history(self, node_id: int) -> Optional[NodeHistory]:
        """Get history for a specific node."""
        return self.node_histories.get(node_id)
    
    def get_full_summary(self) -> str:
        """Generate complete summary of tree construction."""
        lines = []
        lines.append("=" * 70)
        lines.append("CHAID TREE CONSTRUCTION HISTORY")
        lines.append("=" * 70)
        lines.append(f"\nParameters: {self.parameters}")
        lines.append(f"Total nodes: {len(self.node_histories)}")
        lines.append("")
        
        for node_id in self.construction_order:
            if node_id in self.node_histories:
                lines.append(self.node_histories[node_id].get_summary())
                lines.append("")
        
        return "\n".join(lines)
