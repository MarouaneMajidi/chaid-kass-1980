"""
CHAID (Chi-squared Automatic Interaction Detection)

A production-quality Python implementation of the CHAID algorithm
as described by G.V. Kass (1980).

This implementation follows the exact algorithm specification from:
- Kass, G.V. (1980). An exploratory technique for investigating large
  quantities of categorical data. Applied Statistics, 29(2), 119-127.
"""

from .tree import CHAIDTree, PredictorConfig
from .node import CHAIDNode
from .statistics import chi_square_test, bonferroni_multiplier
from .types import PredictorType
from .visualization import (
    TreeVisualizer, 
    visualize_tree, 
    pairwise_chi_square_table,
    get_merge_pairwise_table,
    get_node_merge_history_detailed,
    get_all_pairwise_at_step,
    get_successive_merges_table,
    get_predictor_summary_table
)
from .history import TreeHistory, NodeHistory, PredictorEvaluation

__version__ = "1.0.0"
__author__ = "Marouane Majidi & Walid EL Majdi & Naim Chadia"

__all__ = [
    "CHAIDTree",
    "CHAIDNode",
    "PredictorConfig",
    "chi_square_test",
    "bonferroni_multiplier",
    "PredictorType",
    "TreeVisualizer",
    "visualize_tree",
    "pairwise_chi_square_table",
    "get_merge_pairwise_table",
    "get_node_merge_history_detailed",
    "get_all_pairwise_at_step",
    "get_successive_merges_table",
    "get_predictor_summary_table",
    "TreeHistory",
    "NodeHistory",
    "PredictorEvaluation",
]
