"""
Visualization module for CHAID trees.

Provides text-based and matplotlib visualizations of CHAID trees:

- Node labels must include split variable
- Merged categories shown
- Test statistic (χ²)
- p-value
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union
import warnings

from .node import CHAIDNode
from .tree import CHAIDTree
from .statistics import compute_expected_frequencies, compute_chi_square_statistic
from scipy import stats


def format_group(group: frozenset) -> str:
    """Format a group of categories for display."""
    if len(group) == 1:
        return str(list(group)[0])
    else:
        items = sorted([str(x) for x in group])
        if len(items) <= 3:
            return "{" + ", ".join(items) + "}"
        else:
            remaining = len(items) - 2
            return "{" + ", ".join(items[:2]) + f", ...+{remaining}" + "}"


def format_distribution(dist: Dict[Any, int], n: int) -> str:
    """Format Y distribution for display."""
    parts = []
    for cat, count in sorted(dist.items(), key=lambda x: str(x[0])):
        pct = 100 * count / n if n > 0 else 0
        parts.append(f"{cat}: {count} ({pct:.0f}%)")
    return " | ".join(parts)


class TreeVisualizer:
    """
    Visualizer for CHAID trees.
    
    Supports:
    - Text-based ASCII tree visualization
    - Matplotlib tree plots
    - Summary tables
    """
    
    def __init__(self, tree: CHAIDTree):
        """
        Initialize visualizer with a fitted tree.
        
        Args:
            tree: A fitted CHAIDTree
        """
        if tree.root is None:
            raise ValueError("Tree must be fitted before visualization")
        self.tree = tree
    
    def text_tree(
        self, 
        show_distribution: bool = True,
        show_statistics: bool = True,
        max_width: int = 80
    ) -> str:
        """
        Generate text-based ASCII tree visualization.
        
        Args:
            show_distribution: Show Y distribution at each node
            show_statistics: Show chi-square and p-values
            max_width: Maximum line width
            
        Returns:
            String representation of tree
        """
        lines = []
        lines.append("=" * max_width)
        lines.append("CHAID DECISION TREE")
        lines.append("=" * max_width)
        lines.append(f"Nodes: {len(self.tree.nodes)}")
        lines.append(f"Depth: {self.tree.get_depth()}")
        lines.append(f"Leaves: {len(self.tree.get_leaves())}")
        lines.append("=" * max_width)
        lines.append("")
        
        self._add_node_text(
            self.tree.root, lines, "", True, 
            show_distribution, show_statistics
        )
        
        return "\n".join(lines)
    
    def _add_node_text(
        self,
        node: CHAIDNode,
        lines: List[str],
        prefix: str,
        is_last: bool,
        show_dist: bool,
        show_stats: bool
    ):
        """Recursively add node text to lines."""
        # Determine connector
        connector = "└── " if is_last else "├── "
        
        # Node header
        if node.is_leaf:
            node_marker = "LEAF"
        else:
            node_marker = "SPLIT"
        
        header = f"{prefix}{connector}[Node {node.node_id}] {node_marker} (n={node.n_observations})"
        lines.append(header)
        
        # Extension prefix for children
        extension = "    " if is_last else "│   "
        child_prefix = prefix + extension
        
        # Show distribution
        if show_dist:
            modal = node.modal_category
            modal_count = node.y_distribution.get(modal, 0)
            modal_pct = 100 * modal_count / node.n_observations if node.n_observations > 0 else 0
            lines.append(f"{child_prefix}Modal: {modal} ({modal_count}/{node.n_observations} = {modal_pct:.1f}%)")
            
            dist_str = format_distribution(node.y_distribution, node.n_observations)
            if len(dist_str) < 60:
                lines.append(f"{child_prefix}Dist: {dist_str}")
        
        # Show split info
        if not node.is_leaf and node.split_variable:
            lines.append(f"{child_prefix}Split variable: {node.split_variable} ({node.split_variable_type.value if node.split_variable_type else 'unknown'})")
            
            if show_stats:
                lines.append(f"{child_prefix}χ² = {node.split_chi_square:.4f}, df = {node.split_degrees_of_freedom}")
                lines.append(f"{child_prefix}p-value = {node.split_p_value:.6f}")
                if node.split_p_value_adjusted is not None:
                    lines.append(f"{child_prefix}p-value (adjusted) = {node.split_p_value_adjusted:.6f} (×{node.split_bonferroni_multiplier:.0f})")
            
            if node.split_groups:
                lines.append(f"{child_prefix}Groups:")
                for i, group in enumerate(node.split_groups):
                    group_str = format_group(group)
                    if i < len(node.children):
                        child_n = node.children[i].n_observations
                        lines.append(f"{child_prefix}  [{i+1}] {group_str} → Node {node.children[i].node_id} (n={child_n})")
                    else:
                        lines.append(f"{child_prefix}  [{i+1}] {group_str}")
        
        lines.append(f"{child_prefix}")
        
        # Recurse to children
        for i, child in enumerate(node.children):
            is_child_last = (i == len(node.children) - 1)
            self._add_node_text(child, lines, child_prefix, is_child_last, show_dist, show_stats)
    
    def plot_tree(
        self,
        figsize: Tuple[int, int] = (20, 14),
        font_size: int = 8,
        title: str = "CHAID Decision Tree",
        show_bars: bool = False
    ):
        """
        Create matplotlib visualization of tree in CHAID style.
        
        Style matches classical CHAID output with:
        - Node boxes showing category distributions as text
        - Split variable and statistics below parent nodes
        - Edge labels showing category groups
        
        Args:
            figsize: Figure size (width, height)
            font_size: Font size for labels
            title: Plot title
            show_bars: If True, show mini distribution bars in nodes
            
        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.patches import FancyBboxPatch, Rectangle
        except ImportError:
            warnings.warn("matplotlib not available. Install it for graphical visualization.")
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Calculate positions for nodes using hierarchical layout
        positions = self._calculate_hierarchical_positions()
        
        if not positions:
            ax.text(0.5, 0.5, "Empty tree", ha='center', va='center', fontsize=14)
            return fig
        
        # Get Y categories for consistent ordering
        y_categories = sorted(self.tree.y_categories, key=str)
        
        # Color map for categories
        colors = plt.cm.Set1(np.linspace(0, 1, len(y_categories)))
        color_map = {cat: colors[i] for i, cat in enumerate(y_categories)}
        
        # Node dimensions
        node_width = 0.10
        node_height = 0.08
        
        # Draw edges first (under nodes)
        for node in self.tree.nodes.values():
            if node.children:
                x1, y1 = positions[node.node_id]
                y1_bottom = y1 - node_height / 2
                
                for i, child in enumerate(node.children):
                    x2, y2 = positions[child.node_id]
                    y2_top = y2 + node_height / 2
                    
                    # Draw connection line (elbow style)
                    mid_y = (y1_bottom + y2_top) / 2
                    ax.plot([x1, x1], [y1_bottom, mid_y], 'k-', linewidth=1, zorder=1)
                    ax.plot([x1, x2], [mid_y, mid_y], 'k-', linewidth=1, zorder=1)
                    ax.plot([x2, x2], [mid_y, y2_top], 'k-', linewidth=1, zorder=1)
                    
                    # Edge label (category group)
                    if node.split_groups and i < len(node.split_groups):
                        group_str = format_group(node.split_groups[i])
                        label_y = mid_y + 0.015
                        ax.text(x2, label_y, group_str, fontsize=font_size-1,
                               ha='center', va='bottom', fontweight='bold')
        
        # Draw split info between parent and children
        for node in self.tree.nodes.values():
            if not node.is_leaf and node.children:
                x, y = positions[node.node_id]
                y_below = y - node_height / 2 - 0.025
                
                # Split variable info
                split_text = f"{node.split_variable}"
                if node.split_groups:
                    split_text += f" ({len(node.split_groups)} cat.)"
                ax.text(x, y_below, split_text, fontsize=font_size-1,
                       ha='center', va='top', style='italic')
                
                # Chi-square and p-value
                if node.split_p_value_adjusted is not None:
                    stats_text = f"Adj. P-value={node.split_p_value_adjusted:.4f}, Chi-square={node.split_chi_square:.4f}, df={node.split_degrees_of_freedom}"
                else:
                    stats_text = f"P-value={node.split_p_value:.4f}, Chi-square={node.split_chi_square:.4f}, df={node.split_degrees_of_freedom}"
                ax.text(x, y_below - 0.018, stats_text, fontsize=font_size-2,
                       ha='center', va='top', color='gray')
        
        # Draw nodes
        for node in self.tree.nodes.values():
            x, y = positions[node.node_id]
            
            # Node box
            rect = FancyBboxPatch(
                (x - node_width/2, y - node_height/2),
                node_width, node_height,
                boxstyle="round,pad=0.005,rounding_size=0.01",
                facecolor='white',
                edgecolor='black',
                linewidth=1.5,
                zorder=2
            )
            ax.add_patch(rect)
            
            # Build node content
            lines = []
            lines.append(f"Node {node.node_id}")
            lines.append("-" * 20)
            lines.append(f"{'Category':<10} {'%':>6} {'n':>5}")
            
            total_n = node.n_observations
            for cat in y_categories:
                count = node.y_distribution.get(cat, 0)
                pct = 100 * count / total_n if total_n > 0 else 0
                lines.append(f"{str(cat):<10} {pct:>5.2f} {count:>5}")
            
            lines.append("-" * 20)
            pct_total = 100 * total_n / self.tree.root.n_observations
            lines.append(f"Total  ({pct_total:>5.2f}) {total_n}")
            
            # Draw text
            text = "\n".join(lines)
            ax.text(x, y + node_height/2 - 0.008, text, fontsize=font_size-1,
                   ha='center', va='top', family='monospace', zorder=3)
            
            # Draw mini distribution bar if requested
            if show_bars:
                bar_y = y - node_height/2 + 0.012
                bar_height = 0.008
                bar_width_total = node_width * 0.8
                bar_x_start = x - bar_width_total / 2
                
                cumulative_x = bar_x_start
                for cat in y_categories:
                    count = node.y_distribution.get(cat, 0)
                    pct = count / total_n if total_n > 0 else 0
                    bar_w = bar_width_total * pct
                    if bar_w > 0:
                        bar_rect = Rectangle(
                            (cumulative_x, bar_y),
                            bar_w, bar_height,
                            facecolor=color_map[cat],
                            edgecolor='none',
                            zorder=3
                        )
                        ax.add_patch(bar_rect)
                        cumulative_x += bar_w
                
                # Bar outline
                bar_outline = Rectangle(
                    (bar_x_start, bar_y),
                    bar_width_total, bar_height,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=4
                )
                ax.add_patch(bar_outline)
        
        # Add legend for Y categories
        legend_handles = [mpatches.Patch(color=color_map[cat], label=str(cat)) 
                         for cat in y_categories]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=font_size,
                 title='Categories', framealpha=0.9)
        
        plt.tight_layout()
        return fig
    
    def _calculate_hierarchical_positions(self) -> Dict[int, Tuple[float, float]]:
        """Calculate hierarchical x, y positions for CHAID-style layout."""
        positions = {}
        
        if self.tree.root is None:
            return positions
        
        # Calculate subtree widths for proper spacing
        subtree_widths = {}
        
        def calc_width(node):
            if node.is_leaf:
                subtree_widths[node.node_id] = 1
                return 1
            width = sum(calc_width(child) for child in node.children)
            subtree_widths[node.node_id] = max(width, 1)
            return width
        
        calc_width(self.tree.root)
        
        # Assign positions recursively
        max_depth = self.tree.get_depth()
        
        def assign_positions(node, x_start, x_end, depth):
            x = (x_start + x_end) / 2
            y = 0.92 - (depth / (max_depth + 1)) * 0.85
            positions[node.node_id] = (x, y)
            
            if node.children:
                total_width = subtree_widths[node.node_id]
                current_x = x_start
                for child in node.children:
                    child_width = subtree_widths[child.node_id]
                    child_x_end = current_x + (child_width / total_width) * (x_end - x_start)
                    assign_positions(child, current_x, child_x_end, depth + 1)
                    current_x = child_x_end
        
        assign_positions(self.tree.root, 0.05, 0.95, 0)
        
        return positions
        
        return positions
    
    def summary_table(self) -> str:
        """
        Generate summary table of all nodes.
        
        Returns:
            Formatted table string
        """
        lines = []
        
        # Header
        header = f"{'Node':^6} | {'Type':^8} | {'n':^8} | {'Modal':^10} | {'Split Var':^12} | {'χ²':^10} | {'p-value':^12}"
        separator = "-" * len(header)
        
        lines.append(separator)
        lines.append(header)
        lines.append(separator)
        
        # Sort nodes by ID
        for node_id in sorted(self.tree.nodes.keys()):
            node = self.tree.nodes[node_id]
            
            node_type = "LEAF" if node.is_leaf else "SPLIT"
            modal = str(node.modal_category)[:10]
            
            if node.is_leaf:
                split_var = "-"
                chi_sq = "-"
                p_val = "-"
            else:
                split_var = str(node.split_variable)[:12]
                chi_sq = f"{node.split_chi_square:.4f}"
                if node.split_p_value_adjusted is not None:
                    p_val = f"{node.split_p_value_adjusted:.6f}"
                else:
                    p_val = f"{node.split_p_value:.6f}"
            
            line = f"{node_id:^6} | {node_type:^8} | {node.n_observations:^8} | {modal:^10} | {split_var:^12} | {chi_sq:^10} | {p_val:^12}"
            lines.append(line)
        
        lines.append(separator)
        
        return "\n".join(lines)
    
    def rules_table(self) -> str:
        """
        Generate table of decision rules for each leaf.
        
        Returns:
            Formatted rules table
        """
        lines = []
        leaves = self.tree.get_leaves()
        
        lines.append("=" * 80)
        lines.append("DECISION RULES")
        lines.append("=" * 80)
        
        for leaf in sorted(leaves, key=lambda x: x.node_id):
            lines.append(f"\nNode {leaf.node_id} → {leaf.modal_category} (n={leaf.n_observations})")
            lines.append(f"  Rule: {leaf.get_rule()}")
            
            # Show distribution
            dist_parts = []
            for cat, count in sorted(leaf.y_distribution.items(), key=str):
                pct = 100 * count / leaf.n_observations if leaf.n_observations > 0 else 0
                dist_parts.append(f"{cat}: {pct:.1f}%")
            lines.append(f"  Distribution: {', '.join(dist_parts)}")
        
        return "\n".join(lines)


def visualize_tree(
    tree: CHAIDTree,
    method: str = "text",
    **kwargs
) -> Any:
    """
    Convenience function to visualize a CHAID tree.
    
    Args:
        tree: Fitted CHAIDTree
        method: Visualization method - 'text', 'plot', 'summary', or 'rules'
        **kwargs: Additional arguments for the visualization method
        
    Returns:
        Visualization output (string or matplotlib Figure)
    """
    viz = TreeVisualizer(tree)
    
    if method == "text":
        return viz.text_tree(**kwargs)
    elif method == "plot":
        return viz.plot_tree(**kwargs)
    elif method == "summary":
        return viz.summary_table()
    elif method == "rules":
        return viz.rules_table()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'text', 'plot', 'summary', or 'rules'.")


def pairwise_chi_square_table(
    X: Union[np.ndarray, pd.Series, List],
    y: Union[np.ndarray, pd.Series, List],
    category_labels: Optional[List[str]] = None,
    predictor_type: str = "nominal",
    ordered_categories: Optional[List] = None,
    floating_category: Optional[any] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Compute pairwise chi-square statistics and p-values between pairs of categories.
    
    This generates a table like Table 5/9 in the thesis:
    - Chi-squares are above the diagonal
    - P-values are below the diagonal
    - For ORDINAL: only adjacent pairs are computed
    - For FLOATING: ordinal pairs + floating category can pair with any
    - For NOMINAL: all pairs are computed
    
    Args:
        X: Predictor variable (categorical)
        y: Target variable (categorical)
        category_labels: Optional labels for categories (uses unique values if not provided)
        predictor_type: "nominal", "ordinal", or "floating"
        ordered_categories: For ordinal/floating, the natural order of categories
        floating_category: For floating type, the category that can merge with any
        
    Returns:
        Tuple of (DataFrame with pairwise values, formatted string table)
    """
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Get unique categories
    if ordered_categories is not None:
        # Use provided order, filtering to those present in data
        present_cats = set(X)
        categories = [c for c in ordered_categories if c in present_cats]
        # Add any categories not in ordered list (shouldn't happen normally)
        for c in sorted(present_cats, key=str):
            if c not in categories:
                categories.append(c)
    else:
        categories = sorted(set(X), key=str)
    
    n_cats = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    
    if category_labels is None:
        category_labels = [str(c) for c in categories]
    
    # Initialize result matrices with NaN to indicate "not computed"
    chi2_matrix = np.full((n_cats, n_cats), np.nan)
    pval_matrix = np.full((n_cats, n_cats), np.nan)
    
    # Get unique Y categories
    y_categories = sorted(set(y), key=str)
    
    # Determine which pairs to compute based on predictor type
    # Normalize predictor_type to string for comparison
    if hasattr(predictor_type, 'value'):
        ptype = predictor_type.value
    else:
        ptype = str(predictor_type).lower()
    
    def get_valid_pairs():
        """Get list of (i, j) pairs that should be computed."""
        if ptype == "nominal":
            # All pairs
            return [(i, j) for i in range(n_cats) for j in range(i + 1, n_cats)]
        
        elif ptype == "ordinal":
            # Only adjacent pairs
            return [(i, i + 1) for i in range(n_cats - 1)]
        
        elif ptype == "floating":
            # Adjacent ordinal pairs + floating can pair with anyone
            pairs = []
            floating_idx = cat_to_idx.get(floating_category) if floating_category else None
            
            for i in range(n_cats):
                for j in range(i + 1, n_cats):
                    # If either is the floating category, allow the pair
                    if floating_idx is not None and (i == floating_idx or j == floating_idx):
                        pairs.append((i, j))
                    # Otherwise, only allow adjacent pairs (ordinal rule)
                    elif j == i + 1:
                        # But skip if floating is between them
                        if floating_idx is None or not (i < floating_idx < j):
                            pairs.append((i, j))
            return pairs
        
        return []
    
    valid_pairs = get_valid_pairs()
    
    # Compute chi-square tests for valid pairs only
    for i, j in valid_pairs:
        cat_i = categories[i]
        cat_j = categories[j]
        
        # Filter to only these two categories
        mask = (X == cat_i) | (X == cat_j)
        X_subset = X[mask]
        y_subset = y[mask]
        
        # Build 2 x d contingency table
        contingency = np.zeros((2, len(y_categories)))
        for k, y_cat in enumerate(y_categories):
            contingency[0, k] = np.sum((X_subset == cat_i) & (y_subset == y_cat))
            contingency[1, k] = np.sum((X_subset == cat_j) & (y_subset == y_cat))
        
        # Compute chi-square
        expected = compute_expected_frequencies(contingency)
        
        # Avoid division by zero
        valid_mask = expected > 0
        if np.any(valid_mask):
            chi2 = np.sum(
                (contingency[valid_mask] - expected[valid_mask])**2 / expected[valid_mask]
            )
        else:
            chi2 = 0.0
        
        # Degrees of freedom: (2-1) * (d-1) = d-1
        df = len(y_categories) - 1
        
        # P-value
        if df > 0:
            p_value = 1 - stats.chi2.cdf(chi2, df)
        else:
            p_value = 1.0
        
        # Store in matrices (chi2 above diagonal, p-value below)
        chi2_matrix[i, j] = chi2
        pval_matrix[j, i] = p_value
    
    # Create combined matrix for display
    combined_matrix = np.full((n_cats, n_cats), np.nan)
    for i in range(n_cats):
        combined_matrix[i, i] = 0  # Diagonal is 0
        for j in range(n_cats):
            if i < j and not np.isnan(chi2_matrix[i, j]):
                combined_matrix[i, j] = chi2_matrix[i, j]
            elif i > j and not np.isnan(pval_matrix[i, j]):
                combined_matrix[i, j] = pval_matrix[i, j]
    
    # Create DataFrame
    df_result = pd.DataFrame(
        combined_matrix,
        index=category_labels,
        columns=category_labels
    )
    
    # Generate formatted string table (like Table 9 in thesis)
    lines = []
    # Handle both string and PredictorType enum
    if hasattr(predictor_type, 'value'):
        type_str = predictor_type.value.upper()
    else:
        type_str = str(predictor_type).upper()
    lines.append(f"Table: Chi-squares and p-values by pair of categories ({type_str} predictor)")
    lines.append("=" * (12 + 10 * n_cats))
    
    # Header row with category labels
    header = f"{'':>8}"
    for label in category_labels:
        # Truncate long labels
        lbl = str(label)[:8]
        header += f"{lbl:>10}"
    lines.append(header)
    lines.append("-" * (8 + 10 * n_cats))
    
    # Data rows
    for i in range(n_cats):
        row_label = str(category_labels[i])[:8]
        row = f"{row_label:>8}"
        for j in range(n_cats):
            if i == j:
                row += f"{'0':>10}"
            elif i < j:
                # Chi-square (above diagonal)
                if np.isnan(chi2_matrix[i, j]):
                    row += f"{'':>10}"  # Empty for non-adjacent ordinal
                else:
                    row += f"{chi2_matrix[i, j]:>10.2f}"
            else:
                # P-value (below diagonal)
                if np.isnan(pval_matrix[i, j]):
                    row += f"{'':>10}"  # Empty for non-adjacent ordinal
                else:
                    row += f"{pval_matrix[i, j]:>10.3f}"
        lines.append(row)
    
    lines.append("-" * (8 + 10 * n_cats))
    lines.append("Chi-squares are above the diagonal and p-values below the diagonal.")
    
    if predictor_type == "ordinal":
        lines.append("(Only adjacent category pairs are computed for ordinal predictors)")
    elif predictor_type == "floating":
        lines.append(f"(Adjacent pairs + floating category '{floating_category}' pairs are computed)")
    
    # Find minimum chi-square among valid pairs
    if valid_pairs:
        min_chi2 = float('inf')
        min_pair = None
        for i, j in valid_pairs:
            if chi2_matrix[i, j] < min_chi2:
                min_chi2 = chi2_matrix[i, j]
                min_pair = (i, j)
        
        if min_pair:
            min_pval = pval_matrix[min_pair[1], min_pair[0]]
            lines.append(f"\n→ Most similar pair: [{category_labels[min_pair[0]]}] and [{category_labels[min_pair[1]]}] (χ²={min_chi2:.2f}, p={min_pval:.4f})")
    
    table_str = "\n".join(lines)
    
    return df_result, table_str


def get_merge_pairwise_table(
    tree: CHAIDTree,
    node_id: int = 0,
    predictor: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Get the pairwise chi-square/p-value table for a specific node and predictor.
    
    This is useful for examining which categories could be merged at each step.
    
    Args:
        tree: Fitted CHAIDTree
        node_id: Node ID to examine (default: root)
        predictor: Predictor variable name (if None, uses the selected split variable)
        
    Returns:
        Tuple of (DataFrame, formatted string) or (None, error message)
    """
    if tree.history is None:
        return None, "Tree has no history. Fit with history enabled."
    
    node_history = tree.history.get_node_history(node_id)
    if node_history is None:
        return None, f"No history found for node {node_id}"
    
    # Find the predictor evaluation
    if predictor is None:
        predictor = node_history.selected_predictor
    
    if predictor is None:
        return None, f"Node {node_id} has no split variable"
    
    # Get the node's data
    node = tree.nodes.get(node_id)
    if node is None:
        return None, f"Node {node_id} not found"
    
    # Find predictor evaluation in history
    pred_eval = node_history.predictor_evaluations.get(predictor)
    
    if pred_eval is None:
        return None, f"No evaluation found for predictor '{predictor}' at node {node_id}"
    
    # Generate table from merge history
    lines = []
    lines.append(f"Pairwise Merge Analysis for Node {node_id}, Predictor: {predictor}")
    lines.append("=" * 70)
    lines.append(f"Original categories: {list(pred_eval.original_categories)}")
    lines.append(f"Final groups: {[list(g) for g in pred_eval.final_groups]}")
    lines.append("")
    lines.append("Merge steps:")
    
    for i, merge_rec in enumerate(pred_eval.merge_history):
        status = "MERGED" if merge_rec.was_merged else "STOPPED"
        lines.append(f"  {i+1}. {status}: χ²={merge_rec.chi_square:.4f}, p={merge_rec.p_value:.4f}")
        lines.append(f"      Pair: {set(merge_rec.merged_pair[0])} + {set(merge_rec.merged_pair[1])}")
    
    return None, "\n".join(lines)


def get_node_merge_history_detailed(
    tree: CHAIDTree,
    node_id: int = 0,
    predictor: Optional[str] = None
) -> str:
    """
    Get detailed step-by-step merge history for a predictor at a specific node.
    
    This shows the complete evolution of category groupings through the merge process:
    1. Initial state (all original categories)
    2. Each merge step with:
       - Current groups before merge
       - Pairwise chi-square/p-value table for all current groups
       - Which pair was selected (lowest chi-square / highest p-value)
       - Decision (merge if p > alpha, stop otherwise)
    3. Final groups after all merging
    
    Args:
        tree: Fitted CHAIDTree
        node_id: Node ID to examine (default: 0 = root)
        predictor: Predictor variable name (if None, uses the selected split variable)
        
    Returns:
        Formatted string with complete merge history
    """
    if tree.history is None:
        return "Error: Tree has no history. Fit with history enabled."
    
    node_history = tree.history.get_node_history(node_id)
    if node_history is None:
        return f"Error: No history found for node {node_id}"
    
    # Find the predictor
    if predictor is None:
        predictor = node_history.selected_predictor
    
    if predictor is None:
        return f"Error: Node {node_id} has no split variable"
    
    pred_eval = node_history.predictor_evaluations.get(predictor)
    if pred_eval is None:
        return f"Error: No evaluation found for predictor '{predictor}' at node {node_id}"
    
    # Build detailed output
    lines = []
    lines.append("╔" + "═" * 78 + "╗")
    lines.append(f"║{'DETAILED MERGE HISTORY':^78}║")
    lines.append("╠" + "═" * 78 + "╣")
    lines.append(f"║  Node: {node_id:<10} Predictor: {predictor:<20} Type: {pred_eval.predictor_type.value:<15}║")
    lines.append(f"║  Alpha (merge): {tree.alpha_merge:<10}                                              ║")
    lines.append("╚" + "═" * 78 + "╝")
    lines.append("")
    
    # Show original categories
    orig_cats = list(pred_eval.original_categories)
    lines.append(f"INITIAL STATE: {len(orig_cats)} categories")
    lines.append("-" * 60)
    for i, cat in enumerate(orig_cats):
        lines.append(f"  [{i+1}] {cat}")
    lines.append("")
    
    if not pred_eval.merge_history:
        lines.append("No merge operations were performed.")
        lines.append(f"(Only {len(orig_cats)} categories - no pairs to merge)")
        return "\n".join(lines)
    
    # Track current groups through iterations
    current_groups = [frozenset([cat]) for cat in orig_cats]
    
    # Process each merge step
    for step_num, merge_rec in enumerate(pred_eval.merge_history, 1):
        lines.append("=" * 78)
        lines.append(f"STEP {step_num}: Testing merge possibilities")
        lines.append("=" * 78)
        
        # Show current groups before this step
        groups_before = merge_rec.categories_before
        n_groups = len(groups_before)
        
        lines.append(f"\nCurrent groups ({n_groups}):")
        group_labels = {}
        for i, grp in enumerate(groups_before):
            label = i + 1
            group_labels[grp] = label
            if len(grp) == 1:
                lines.append(f"  [{label}] {list(grp)[0]}")
            else:
                lines.append(f"  [{label}] {{{', '.join(str(x) for x in sorted(grp, key=str))}}}")
        
        # Show the pair that was tested
        pair1, pair2 = merge_rec.merged_pair
        label1 = group_labels.get(pair1, "?")
        label2 = group_labels.get(pair2, "?")
        
        lines.append(f"\nMost similar pair (lowest χ²):")
        lines.append(f"  Groups [{label1}] and [{label2}]")
        lines.append("")
        
        # Show the test result in a nice box
        lines.append("┌" + "─" * 50 + "┐")
        lines.append(f"│  Chi-square test for pair [{label1}] vs [{label2}]" + " " * (50 - 38 - len(str(label1)) - len(str(label2))) + "│")
        lines.append("├" + "─" * 50 + "┤")
        lines.append(f"│  χ² = {merge_rec.chi_square:>10.4f}                              │")
        lines.append(f"│  df = {merge_rec.degrees_of_freedom:>10}                              │")
        lines.append(f"│  p  = {merge_rec.p_value:>10.6f}                              │")
        lines.append("├" + "─" * 50 + "┤")
        
        if merge_rec.was_merged:
            lines.append(f"│  DECISION: MERGE (p = {merge_rec.p_value:.4f} > α = {tree.alpha_merge})     │")
            lines.append("│  → Categories are statistically similar           │")
            lines.append("└" + "─" * 50 + "┘")
            
            # Show the merged result
            merged_group = pair1 | pair2
            lines.append(f"\n  Result: [{label1}] + [{label2}] → {{{', '.join(str(x) for x in sorted(merged_group, key=str))}}}")
            
            # Update current groups for next iteration
            current_groups = list(merge_rec.categories_after)
        else:
            lines.append(f"│  DECISION: STOP (p = {merge_rec.p_value:.4f} ≤ α = {tree.alpha_merge})      │")
            lines.append("│  → Categories are significantly different         │")
            lines.append("└" + "─" * 50 + "┘")
            lines.append(f"\n  No merge performed. Merging process complete.")
        
        lines.append("")
    
    # Show final groups
    lines.append("=" * 78)
    lines.append("FINAL RESULT")
    lines.append("=" * 78)
    lines.append(f"\nFinal groups after merging ({len(pred_eval.final_groups)}):")
    for i, grp in enumerate(pred_eval.final_groups):
        if len(grp) == 1:
            lines.append(f"  Group {i+1}: {list(grp)[0]}")
        else:
            merged_cats = sorted([str(x) for x in grp])
            lines.append(f"  Group {i+1}: {{{', '.join(merged_cats)}}} (merged)")
    
    lines.append("")
    lines.append(f"Final chi-square: {pred_eval.final_chi_square:.4f}")
    lines.append(f"Degrees of freedom: {pred_eval.final_degrees_of_freedom}")
    lines.append(f"Raw p-value: {pred_eval.final_p_value:.6f}")
    if pred_eval.bonferroni_multiplier:
        lines.append(f"Bonferroni multiplier: {pred_eval.bonferroni_multiplier:.1f}")
        lines.append(f"Adjusted p-value: {pred_eval.adjusted_p_value:.6f}")
    
    if pred_eval.was_selected:
        lines.append("\n✓ This predictor was SELECTED for splitting at this node.")
    else:
        lines.append("\n✗ This predictor was NOT selected (another had lower p-value).")
    
    return "\n".join(lines)


def get_all_pairwise_at_step(
    tree: CHAIDTree,
    node_id: int,
    predictor: str,
    step: int = 0
) -> str:
    """
    Get the pairwise chi-square table at a specific step of the merge process.
    
    Args:
        tree: Fitted CHAIDTree
        node_id: Node ID
        predictor: Predictor variable name
        step: Step number (0 = initial state before any merging)
        
    Returns:
        Formatted pairwise table string
    """
    if tree.history is None:
        return "Error: No history available"
    
    node_history = tree.history.get_node_history(node_id)
    if node_history is None:
        return f"Error: No history for node {node_id}"
    
    pred_eval = node_history.predictor_evaluations.get(predictor)
    if pred_eval is None:
        return f"Error: No evaluation for predictor '{predictor}'"
    
    # Get the groups at this step
    if step == 0:
        groups = [frozenset([cat]) for cat in pred_eval.original_categories]
        title = "Initial pairwise table (before any merging)"
    elif step <= len(pred_eval.merge_history):
        merge_rec = pred_eval.merge_history[step - 1]
        groups = list(merge_rec.categories_after if merge_rec.was_merged else merge_rec.categories_before)
        title = f"Pairwise table after step {step}"
    else:
        return f"Error: Step {step} does not exist. Max step: {len(pred_eval.merge_history)}"
    
    # Get node data
    node = tree.nodes.get(node_id)
    if node is None:
        return f"Error: Node {node_id} not found"
    
    indices = node.indices
    X_col = tree._X.loc[indices, predictor].values
    y_subset = tree._Y[indices]
    y_categories = tree.y_categories
    
    n_groups = len(groups)
    predictor_type = pred_eval.predictor_type
    
    # Determine valid pairs based on predictor type
    # Import the helper function logic
    def get_valid_pairs_for_groups():
        """Get valid pairs based on predictor type."""
        if predictor_type.value == "nominal":
            return [(i, j) for i in range(n_groups) for j in range(i + 1, n_groups)]
        elif predictor_type.value == "ordinal":
            # Only adjacent groups
            return [(i, i + 1) for i in range(n_groups - 1)]
        elif predictor_type.value == "floating":
            # Get floating category from predictor config
            floating_cat = None
            if hasattr(tree, 'predictor_configs') and predictor in tree.predictor_configs:
                floating_cat = tree.predictor_configs[predictor].floating_category
            
            pairs = []
            
            # Find which group contains floating category
            floating_group_idx = None
            if floating_cat:
                for idx, grp in enumerate(groups):
                    if floating_cat in grp:
                        floating_group_idx = idx
                        break
            
            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    if floating_group_idx is not None and (i == floating_group_idx or j == floating_group_idx):
                        pairs.append((i, j))
                    elif j == i + 1:
                        pairs.append((i, j))
            return pairs
        return []
    
    valid_pairs = get_valid_pairs_for_groups()
    
    # Compute pairwise chi-squares for valid pairs only
    chi2_matrix = np.full((n_groups, n_groups), np.nan)
    pval_matrix = np.full((n_groups, n_groups), np.nan)
    
    for i, j in valid_pairs:
        grp_i = groups[i]
        grp_j = groups[j]
        
        # Build 2 x d contingency table
        contingency = np.zeros((2, len(y_categories)))
        for k, y_cat in enumerate(y_categories):
            # Count for group i
            for cat in grp_i:
                contingency[0, k] += np.sum((X_col == cat) & (y_subset == y_cat))
            # Count for group j
            for cat in grp_j:
                contingency[1, k] += np.sum((X_col == cat) & (y_subset == y_cat))
        
        # Compute chi-square
        expected = compute_expected_frequencies(contingency)
        valid = expected > 0
        if np.any(valid):
            chi2 = np.sum((contingency[valid] - expected[valid])**2 / expected[valid])
        else:
            chi2 = 0.0
        
        df = len(y_categories) - 1
        p_val = 1 - stats.chi2.cdf(chi2, df) if df > 0 else 1.0
        
        chi2_matrix[i, j] = chi2
        pval_matrix[j, i] = p_val
    
    # Format output
    lines = []
    lines.append(title)
    lines.append("=" * (15 + 12 * n_groups))
    
    # Group labels
    lines.append("\nGroups:")
    for i, grp in enumerate(groups):
        if len(grp) == 1:
            lines.append(f"  [{i+1}] {list(grp)[0]}")
        else:
            lines.append(f"  [{i+1}] {{{', '.join(str(x) for x in sorted(grp, key=str))}}}")
    lines.append("")
    
    # Header
    header = f"{'':>6}"
    for j in range(1, n_groups + 1):
        header += f"{j:>12}"
    lines.append(header)
    lines.append("-" * (6 + 12 * n_groups))
    
    # Data rows
    for i in range(n_groups):
        row = f"{i+1:>6}"
        for j in range(n_groups):
            if i == j:
                row += f"{'—':>12}"
            elif i < j:
                if np.isnan(chi2_matrix[i, j]):
                    row += f"{'':>12}"  # Empty for non-valid pairs
                else:
                    row += f"{chi2_matrix[i, j]:>12.2f}"
            else:
                if np.isnan(pval_matrix[i, j]):
                    row += f"{'':>12}"  # Empty for non-valid pairs
                else:
                    row += f"{pval_matrix[i, j]:>12.4f}"
        lines.append(row)
    
    lines.append("-" * (6 + 12 * n_groups))
    lines.append("Upper triangle: χ² values | Lower triangle: p-values")
    
    if predictor_type.value == "ordinal":
        lines.append("(Only adjacent pairs computed for ORDINAL predictor)")
    elif predictor_type.value == "floating":
        lines.append("(Adjacent + floating category pairs computed for FLOATING predictor)")
    
    # Highlight minimum chi-square pair among valid pairs
    if valid_pairs:
        min_chi2 = float('inf')
        min_pair = None
        for i, j in valid_pairs:
            if not np.isnan(chi2_matrix[i, j]) and chi2_matrix[i, j] < min_chi2:
                min_chi2 = chi2_matrix[i, j]
                min_pair = (i + 1, j + 1)
        
        if min_pair:
            min_pval = pval_matrix[min_pair[1] - 1, min_pair[0] - 1]
            lines.append(f"\n→ Most similar pair: [{min_pair[0]}] and [{min_pair[1]}] (χ²={min_chi2:.2f}, p={min_pval:.4f})")
    
    return "\n".join(lines)


def get_successive_merges_table(
    tree: CHAIDTree,
    node_id: int = 0,
    predictor: Optional[str] = None,
    include_splits: bool = True
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Generate a "Successive merges" table like Table 6 in the thesis.
    
    Shows each iteration of the merge/split process with:
    - Iteration number
    - What was merged or split
    - Chi-square statistic
    - p-value
    - Decision (merged/stopped or split/kept)
    
    Args:
        tree: Fitted CHAIDTree
        node_id: Node ID to examine (default: 0 = root)
        predictor: Predictor variable name (if None, uses the selected split variable)
        include_splits: Whether to include split check iterations
        
    Returns:
        Tuple of (DataFrame with the data, formatted string for display)
    """
    if tree.history is None:
        return None, "Error: Tree has no history. Fit with history enabled."
    
    node_history = tree.history.get_node_history(node_id)
    if node_history is None:
        return None, f"Error: No history found for node {node_id}"
    
    # Find the predictor
    if predictor is None:
        predictor = node_history.selected_predictor
    
    if predictor is None:
        return None, f"Error: Node {node_id} has no split variable"
    
    pred_eval = node_history.predictor_evaluations.get(predictor)
    if pred_eval is None:
        return None, f"Error: No evaluation found for predictor '{predictor}' at node {node_id}"
    
    # Build the data
    rows = []
    iteration = 0
    
    # Track original category labels (1-indexed like the thesis)
    orig_cats = list(pred_eval.original_categories)
    cat_to_label = {cat: i + 1 for i, cat in enumerate(orig_cats)}
    
    def format_group_labels(group: frozenset) -> str:
        """Format a group using original category labels."""
        labels = sorted([cat_to_label.get(cat, cat) for cat in group], key=lambda x: (isinstance(x, str), x))
        if len(labels) == 1:
            return str(labels[0])
        else:
            return "{" + ",".join(str(l) for l in labels) + "}"
    
    def format_merge_pair(grp1: frozenset, grp2: frozenset) -> str:
        """Format a merge as {labels1, labels2}."""
        lbl1 = format_group_labels(grp1)
        lbl2 = format_group_labels(grp2)
        # If both are single, format as {a,b}
        if len(grp1) == 1 and len(grp2) == 1:
            return "{" + lbl1 + "," + lbl2 + "}"
        # Otherwise show as {lbl1, lbl2}
        return "{" + lbl1 + "," + lbl2 + "}"
    
    # Process merge history
    for merge_rec in pred_eval.merge_history:
        iteration += 1
        pair1, pair2 = merge_rec.merged_pair
        merge_str = format_merge_pair(pair1, pair2)
        
        rows.append({
            'Iteration': iteration,
            'Operation': 'Merge' if merge_rec.was_merged else 'Stop',
            'Categories': merge_str,
            'Chi-square': merge_rec.chi_square,
            'p-value': merge_rec.p_value,
            'Decision': 'Merged' if merge_rec.was_merged else 'Stopped'
        })
    
    # Process split check history (if any and if requested)
    if include_splits and pred_eval.split_check_history:
        for split_rec in pred_eval.split_check_history:
            iteration += 1
            compound_str = format_group_labels(split_rec.compound_category)
            
            if split_rec.best_split:
                split1, split2 = split_rec.best_split
                operation_str = f"Split {compound_str}"
            else:
                operation_str = f"Check {compound_str}"
            
            rows.append({
                'Iteration': iteration,
                'Operation': 'Split' if split_rec.was_split else 'Check',
                'Categories': compound_str,
                'Chi-square': split_rec.chi_square,
                'p-value': split_rec.p_value,
                'Decision': 'Split' if split_rec.was_split else 'Kept'
            })
    
    if not rows:
        return None, "No merge or split operations were performed."
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Build formatted string output (like Table 6)
    lines = []
    lines.append("╔" + "═" * 76 + "╗")
    lines.append(f"║{'TABLE: Successive Merges':^76}║")
    lines.append(f"║{'Predictor: ' + predictor + ' at Node ' + str(node_id):^76}║")
    lines.append("╠" + "═" * 76 + "╣")
    
    # Header
    lines.append(f"║ {'Iteration':^10} │ {'Merge/Split':^20} │ {'Chi-square':^12} │ {'p-value':^12} │ {'Decision':^10} ║")
    lines.append("╠" + "═" * 76 + "╣")
    
    # Data rows
    for _, row in df.iterrows():
        chi2_str = f"{row['Chi-square']:.2f}"
        pval_str = f"{row['p-value']:.2%}" if row['p-value'] < 1 else f"{row['p-value']:.4f}"
        
        lines.append(f"║ {row['Iteration']:^10} │ {row['Categories']:^20} │ {chi2_str:^12} │ {pval_str:^12} │ {row['Decision']:^10} ║")
    
    lines.append("╚" + "═" * 76 + "╝")
    
    # Add legend
    lines.append("")
    lines.append("Category Labels:")
    for cat, label in cat_to_label.items():
        lines.append(f"  [{label}] = {cat}")
    
    # Add summary
    n_merges = sum(1 for r in rows if r['Decision'] == 'Merged')
    n_splits = sum(1 for r in rows if r['Decision'] == 'Split')
    n_stopped = sum(1 for r in rows if r['Decision'] in ('Stopped', 'Kept'))
    
    lines.append("")
    lines.append(f"Summary: {n_merges} merge(s), {n_splits} split(s), {n_stopped} stop(s)")
    lines.append(f"Final groups: {len(pred_eval.final_groups)}")
    
    # Show final groups
    lines.append("")
    lines.append("Final category groupings:")
    for i, grp in enumerate(pred_eval.final_groups):
        grp_lbl = format_group_labels(grp)
        cats_str = ", ".join(str(c) for c in sorted(grp, key=str))
        lines.append(f"  Group {i+1}: {grp_lbl} = {{{cats_str}}}")
    
    return df, "\n".join(lines)


def get_predictor_summary_table(
    tree: CHAIDTree,
    node_id: int = 0,
    return_dataframe: bool = False
) -> Union[str, Tuple[pd.DataFrame, str]]:
    """
    Generate a summary table of all predictor evaluations at a node.
    
    Similar to Table 10 in the thesis: Summary of possible first level splits.
    Shows all predictors evaluated, their original categories, final groups
    after merging, chi-square statistics, degrees of freedom, p-values,
    and which predictor was selected for splitting.
    
    Args:
        tree: Fitted CHAIDTree
        node_id: Node ID to examine (default: 0 = root)
        return_dataframe: If True, return (DataFrame, str) tuple
        
    Returns:
        Formatted string table, or (DataFrame, string) if return_dataframe=True
        
    Example:
        >>> print(get_predictor_summary_table(tree, node_id=0))
        ╔════════════════════════════════════════════════════════════════════════════╗
        ║              TABLE 10: Summary of Possible First Level Splits              ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║ Predictor       │ Type     │ #cat │ #grp │  Chi-sq │ df │    p-value │ ★ ║
        ╠────────────────────────────────────────────────────────────────────────────╣
        ║ failures_cat    │ ORDINAL  │    4 │    2 │  102.34 │  1 │ 0.0000000  │ ★ ║
        ║ higher          │ NOMINAL  │    2 │    2 │   62.25 │  1 │ 0.0000000  │   ║
        ...
    """
    if tree.history is None:
        return "Error: Tree has no history. Fit with history enabled."
    
    node_history = tree.history.get_node_history(node_id)
    if node_history is None:
        return f"Error: No history found for node {node_id}"
    
    if not node_history.predictor_evaluations:
        return f"Error: No predictor evaluations found for node {node_id}"
    
    # Collect data for each predictor
    rows = []
    for pred_name, eval_rec in node_history.predictor_evaluations.items():
        n_categories = len(eval_rec.original_categories)
        n_splits = len(eval_rec.final_groups)
        chi_sq = eval_rec.final_chi_square
        df = eval_rec.final_degrees_of_freedom
        p_value = eval_rec.final_p_value
        adj_p = eval_rec.adjusted_p_value
        pred_type = eval_rec.predictor_type.value.upper()
        was_selected = eval_rec.was_selected
        
        rows.append({
            'Predictor': pred_name,
            'Type': pred_type,
            '#categories': n_categories,
            '#groups': n_splits,
            'Chi-square': chi_sq,
            'df': df,
            'p-value': p_value,
            'adj. p-value': adj_p if adj_p else p_value,
            'Selected': was_selected
        })
    
    # Sort by adjusted p-value (lowest = most significant = best)
    rows.sort(key=lambda x: x['adj. p-value'])
    
    # Create DataFrame
    df_result = pd.DataFrame(rows)
    
    # Create formatted output
    lines = []
    lines.append("╔" + "═" * 100 + "╗")
    lines.append("║" + " Summary of Possible First Level Splits ".center(100) + "║")
    lines.append("╠" + "═" * 100 + "╣")
    
    # Header
    header = f"║ {'Predictor':<20} │ {'Type':<8} │ {'#cat':>5} │ {'#grp':>5} │ {'Chi-sq':>10} │ {'df':>3} │ {'p-value':>14} │ {'Selected':>8} ║"
    lines.append(header)
    lines.append("╠" + "─" * 100 + "╣")
    
    # Data rows
    for row in rows:
        p_str = f"{row['adj. p-value']:.10f}" if row['adj. p-value'] < 0.0001 else f"{row['adj. p-value']:.6f}"
        selected_str = "★" if row['Selected'] else ""
        line = f"║ {row['Predictor']:<20} │ {row['Type']:<8} │ {row['#categories']:>5} │ {row['#groups']:>5} │ {row['Chi-square']:>10.2f} │ {row['df']:>3} │ {p_str:>14} │ {selected_str:>8} ║"
        lines.append(line)
    
    lines.append("╚" + "═" * 100 + "╝")
    
    # Add explanation
    selected_pred = node_history.selected_predictor
    if selected_pred:
        lines.append(f"\n★ Selected predictor: {selected_pred}")
        lines.append(f"  Selection reason: {node_history.reason_for_selection}")
        
        # Show final groups for selected predictor
        if selected_pred in node_history.predictor_evaluations:
            eval_rec = node_history.predictor_evaluations[selected_pred]
            lines.append(f"\n  Final groups after merging:")
            for i, grp in enumerate(eval_rec.final_groups, 1):
                cats_str = ", ".join(str(c) for c in sorted(grp, key=str))
                lines.append(f"    Group {i}: {{{cats_str}}}")
    else:
        lines.append(f"\n  No predictor selected (no significant split found)")
        if node_history.leaf_reason:
            lines.append(f"  Reason: {node_history.leaf_reason}")
    
    result_str = "\n".join(lines)
    
    if return_dataframe:
        return df_result, result_str
    return result_str

