"""
CHAID Tree implementation.

This is the main class implementing the CHAID algorithm as described by Kass (1980).
The algorithm is documented in main.tex with the following key components:

Algorithm Overview:
1. Fusion (Merging): For each predictor, iteratively merge similar categories
2. Division (Splitting): Verify if merged groups should be split
3. Evaluation: Calculate significance of each optimized predictor
4. Selection: Choose predictor with lowest p-value
5. Recursion: Repeat for each child node

Stopping Rules:
- α_split threshold: Don't partition if adjusted p-value > α_split
- Maximum depth: Limit tree depth
- Minimum parent size: Don't split nodes with < n_parent observations
- Minimum child size: Only consider splits where each child has ≥ n_child observations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, FrozenSet
from dataclasses import dataclass, field

from .types import PredictorType
from .node import CHAIDNode
from .statistics import chi_square_test, bonferroni_multiplier, build_contingency_table
from .merging import optimize_predictor, get_group_y_values
from .history import (
    TreeHistory, NodeHistory, PredictorEvaluation,
    MergeRecord, SplitCheckRecord
)


@dataclass
class PredictorConfig:
    """
    Configuration for a predictor variable.
    
    Attributes:
        name: Column name in the data
        predictor_type: NOMINAL, ORDINAL, or FLOATING
        ordered_categories: For ordinal/floating, the category order
        floating_category: For floating, the floating category value
    """
    name: str
    predictor_type: PredictorType = PredictorType.NOMINAL
    ordered_categories: Optional[List] = None
    floating_category: Optional[Any] = None


class CHAIDTree:
    """
    CHAID (Chi-squared Automatic Interaction Detection) Decision Tree.
    
    This implementation follows the exact algorithm from Kass (1980) as
    documented in main.tex. Key features:
    
    - Multi-way splits (not binary)
    - Chi-square test for independence
    - Category merging heuristic
    - Optional Bonferroni correction
    - Complete traceability of all operations
    
    Parameters:
        alpha_merge: Significance level for merging categories (default 0.05)
        alpha_split: Significance level for checking splits (default 0.049)
                     Must be < alpha_merge for convergence
        alpha_select: Significance level for selecting split variable (default 0.05)
        max_depth: Maximum tree depth (default 5)
        min_parent_size: Minimum observations to attempt split (default 100)
        min_child_size: Minimum observations per child node (default 50)
        apply_bonferroni: Whether to apply Bonferroni correction (default True)
    
    Attributes:
        root: Root node of the tree
        nodes: Dictionary of all nodes by node_id
        history: Complete construction history
        y_categories: Unique categories of dependent variable
    """
    
    def __init__(
        self,
        alpha_merge: float = 0.05,
        alpha_split: float = 0.049,
        alpha_select: float = 0.05,
        max_depth: int = 5,
        min_parent_size: int = 100,
        min_child_size: int = 50,
        apply_bonferroni: bool = True
    ):
        # Validate parameters
        if not (0 < alpha_merge <= 1):
            raise ValueError("alpha_merge must be in (0, 1]")
        if not (0 < alpha_split <= 1):
            raise ValueError("alpha_split must be in (0, 1]")
        if not (0 < alpha_select <= 1):
            raise ValueError("alpha_select must be in (0, 1]")
        
        # Per main.tex Section 4.2.3: alpha_split < alpha_merge for convergence
        # However, allowing equality as noted with 0.049 < 0.05 example
        if alpha_split > alpha_merge:
            raise ValueError("alpha_split must be ≤ alpha_merge for algorithm convergence")
        
        if max_depth < 1:
            raise ValueError("max_depth must be ≥ 1")
        if min_parent_size < 1:
            raise ValueError("min_parent_size must be ≥ 1")
        if min_child_size < 1:
            raise ValueError("min_child_size must be ≥ 1")
        
        self.alpha_merge = alpha_merge
        self.alpha_split = alpha_split
        self.alpha_select = alpha_select
        self.max_depth = max_depth
        self.min_parent_size = min_parent_size
        self.min_child_size = min_child_size
        self.apply_bonferroni = apply_bonferroni
        
        # Tree structure
        self.root: Optional[CHAIDNode] = None
        self.nodes: Dict[int, CHAIDNode] = {}
        self._node_counter = 0
        
        # Data
        self._X: Optional[pd.DataFrame] = None
        self._Y: Optional[np.ndarray] = None
        self.y_categories: Optional[np.ndarray] = None
        self.predictor_configs: Dict[str, PredictorConfig] = {}
        
        # History
        self.history = TreeHistory()
        self.history.parameters = {
            "alpha_merge": alpha_merge,
            "alpha_split": alpha_split,
            "alpha_select": alpha_select,
            "max_depth": max_depth,
            "min_parent_size": min_parent_size,
            "min_child_size": min_child_size,
            "apply_bonferroni": apply_bonferroni,
        }
    
    def _get_next_node_id(self) -> int:
        """Generate next unique node ID."""
        node_id = self._node_counter
        self._node_counter += 1
        return node_id
    
    def _compute_y_distribution(
        self, 
        indices: np.ndarray
    ) -> Tuple[Dict[Any, int], Dict[Any, float]]:
        """
        Compute distribution of Y for given indices.
        
        Args:
            indices: Observation indices
            
        Returns:
            Tuple of (counts dict, proportions dict)
        """
        y_subset = self._Y[indices]
        distribution = {}
        proportions = {}
        n = len(y_subset)
        
        for cat in self.y_categories:
            count = int(np.sum(y_subset == cat))
            distribution[cat] = count
            proportions[cat] = count / n if n > 0 else 0.0
        
        return distribution, proportions
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        predictor_types: Optional[Dict[str, Union[str, PredictorType, PredictorConfig]]] = None
    ) -> 'CHAIDTree':
        """
        Fit the CHAID tree to data.
        
        Args:
            X: Predictor variables (DataFrame or 2D array)
            y: Dependent variable (categorical)
            predictor_types: Optional dict mapping column names to predictor types
                            Values can be:
                            - String: 'nominal', 'ordinal', 'floating'
                            - PredictorType enum
                            - PredictorConfig object (for full configuration)
                            
        Returns:
            self (fitted tree)
        """
        # Convert inputs to standard format
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        
        self._X = X.copy()
        self._Y = np.array(y)
        self.y_categories = np.unique(self._Y)
        
        # Configure predictors
        self.predictor_configs = {}
        for col in self._X.columns:
            if predictor_types and col in predictor_types:
                config = predictor_types[col]
                if isinstance(config, PredictorConfig):
                    self.predictor_configs[col] = config
                elif isinstance(config, PredictorType):
                    self.predictor_configs[col] = PredictorConfig(
                        name=col, predictor_type=config
                    )
                elif isinstance(config, str):
                    ptype = PredictorType(config.lower())
                    self.predictor_configs[col] = PredictorConfig(
                        name=col, predictor_type=ptype
                    )
                else:
                    raise ValueError(f"Invalid predictor config for {col}")
            else:
                # Default to nominal
                self.predictor_configs[col] = PredictorConfig(
                    name=col, predictor_type=PredictorType.NOMINAL
                )
        
        # Build tree starting from root
        all_indices = np.arange(len(self._Y))
        self.root = self._build_node(all_indices, depth=0, parent=None)
        
        return self
    
    def _build_node(
        self,
        indices: np.ndarray,
        depth: int,
        parent: Optional[CHAIDNode]
    ) -> CHAIDNode:
        """
        Recursively build a node and its children.
        
        Args:
            indices: Observation indices for this node
            depth: Current depth in tree
            parent: Parent node (None for root)
            
        Returns:
            The constructed node
        """
        node_id = self._get_next_node_id()
        y_dist, y_prop = self._compute_y_distribution(indices)
        
        # Create node
        node = CHAIDNode(
            node_id=node_id,
            depth=depth,
            indices=indices,
            y_distribution=y_dist,
            y_proportions=y_prop,
            n_observations=len(indices),
            parent=parent,
            is_leaf=True  # Default to leaf, will update if split found
        )
        
        self.nodes[node_id] = node
        
        # Initialize history
        node_history = NodeHistory(
            node_id=node_id,
            depth=depth,
            n_observations=len(indices)
        )
        
        # Check stopping conditions (Section 4.5)
        stop_reason = self._check_stopping_conditions(node)
        
        if stop_reason:
            node_history.is_leaf = True
            node_history.leaf_reason = stop_reason
            node.history = node_history
            self.history.add_node_history(node_history)
            return node
        
        # Find best split
        best_split = self._find_best_split(indices, node_history)
        
        if best_split is None:
            node_history.is_leaf = True
            node_history.leaf_reason = "No significant predictor found"
            node.history = node_history
            self.history.add_node_history(node_history)
            return node
        
        # Extract split information
        predictor_name, groups, chi_sq, df, p_value, p_adj, mult, pred_type = best_split
        
        # Verify child sizes before splitting
        valid_groups = []
        for group in groups:
            mask = np.isin(self._X.loc[indices, predictor_name].values, list(group))
            child_size = np.sum(mask)
            if child_size >= self.min_child_size:
                valid_groups.append(group)
        
        if len(valid_groups) < 2:
            node_history.is_leaf = True
            node_history.leaf_reason = "Child nodes would be too small"
            node.history = node_history
            self.history.add_node_history(node_history)
            return node
        
        # Perform split
        node.is_leaf = False
        node.split_variable = predictor_name
        node.split_variable_type = pred_type
        node.split_groups = tuple(valid_groups)
        node.split_chi_square = chi_sq
        node.split_degrees_of_freedom = df
        node.split_p_value = p_value
        node.split_p_value_adjusted = p_adj
        node.split_bonferroni_multiplier = mult
        
        # Update history
        node_history.selected_predictor = predictor_name
        node_history.selected_groups = tuple(valid_groups)
        node_history.reason_for_selection = f"Lowest adjusted p-value: {p_adj:.6f}" if p_adj else f"Lowest p-value: {p_value:.6f}"
        
        # Create child nodes
        X_col = self._X.loc[indices, predictor_name].values
        
        for group in valid_groups:
            mask = np.isin(X_col, list(group))
            child_indices = indices[mask]
            
            if len(child_indices) > 0:
                child = self._build_node(
                    child_indices,
                    depth=depth + 1,
                    parent=node
                )
                node.children.append(child)
        
        node.history = node_history
        self.history.add_node_history(node_history)
        
        return node
    
    def _check_stopping_conditions(self, node: CHAIDNode) -> Optional[str]:
        """
        Check if node should be a leaf based on stopping rules.
        
        From main.tex Section 4.5:
        - Maximum depth reached
        - Node too small (< min_parent_size)
        
        Args:
            node: The node to check
            
        Returns:
            Reason for stopping, or None if should continue
        """
        if node.depth >= self.max_depth:
            return f"Maximum depth ({self.max_depth}) reached"
        
        if node.n_observations < self.min_parent_size:
            return f"Node size ({node.n_observations}) < min_parent_size ({self.min_parent_size})"
        
        # Check if Y has only one category
        non_zero_cats = sum(1 for v in node.y_distribution.values() if v > 0)
        if non_zero_cats <= 1:
            return "Node is pure (only one category of Y)"
        
        return None
    
    def _find_best_split(
        self,
        indices: np.ndarray,
        node_history: NodeHistory
    ) -> Optional[Tuple[str, List[FrozenSet], float, int, float, Optional[float], Optional[float], PredictorType]]:
        """
        Find the best predictor and grouping for splitting.
        
        Algorithm:
        1. For each predictor, find optimal grouping (merge/split)
        2. Calculate chi-square and p-value for optimal grouping
        3. Apply Bonferroni correction if enabled
        4. Select predictor with lowest adjusted p-value
        
        Args:
            indices: Observation indices
            node_history: History record to update
            
        Returns:
            Tuple of (predictor_name, groups, chi_sq, df, p_value, p_adj, multiplier, pred_type)
            or None if no significant split found
        """
        best_predictor = None
        best_groups = None
        best_p_value = float('inf')
        best_chi_sq = 0.0
        best_df = 0
        best_p_raw = 1.0
        best_mult = None
        best_type = None
        
        Y_subset = self._Y[indices]
        
        for predictor_name, config in self.predictor_configs.items():
            X_col = self._X.loc[indices, predictor_name].values
            
            # Get unique values in proper order
            # For ordinal/floating, use ordered_categories if provided
            data_unique = set(np.unique(X_col))
            if config.ordered_categories is not None:
                # Use provided order, filtering to only values present in data
                unique_vals = [cat for cat in config.ordered_categories if cat in data_unique]
                unique_vals = np.array(unique_vals)
            else:
                unique_vals = np.unique(X_col)
            
            # Skip if predictor has only one unique value
            if len(unique_vals) < 2:
                continue
            
            # Step 1 & 2: Optimize predictor (merge and split check)
            # Pass unique_vals as ordered_categories since it's already in correct order
            groups, merge_history, split_history = optimize_predictor(
                X=X_col,
                Y=Y_subset,
                y_categories=self.y_categories,
                predictor_type=config.predictor_type,
                alpha_merge=self.alpha_merge,
                alpha_split=self.alpha_split,
                ordered_categories=list(unique_vals),
                floating_category=config.floating_category
            )
            
            # Skip if all categories merged into one group
            if len(groups) < 2:
                continue
            
            # Step 3: Evaluate significance of optimal grouping
            # Build contingency table for final grouping
            group_labels = np.zeros(len(X_col), dtype=int)
            for i, group in enumerate(groups):
                for val in group:
                    group_labels[X_col == val] = i
            
            contingency = build_contingency_table(
                group_labels, Y_subset, self.y_categories
            )
            
            # Calculate chi-square
            c_original = len(unique_vals)
            g_final = len(groups)
            
            result = chi_square_test(
                contingency,
                predictor_type=config.predictor_type,
                c_original=c_original,
                g_final=g_final,
                apply_bonferroni=self.apply_bonferroni
            )
            
            # Get p-value for comparison
            p_for_comparison = result.p_value_adjusted if self.apply_bonferroni else result.p_value
            
            # Record evaluation
            eval_record = PredictorEvaluation(
                predictor_name=predictor_name,
                predictor_type=config.predictor_type,
                original_categories=tuple(unique_vals),
                final_groups=tuple(groups),
                merge_history=merge_history,
                split_check_history=split_history,
                final_chi_square=result.statistic,
                final_degrees_of_freedom=result.degrees_of_freedom,
                final_p_value=result.p_value,
                bonferroni_multiplier=result.bonferroni_multiplier,
                adjusted_p_value=result.p_value_adjusted,
                was_selected=False
            )
            node_history.predictor_evaluations[predictor_name] = eval_record
            
            # Step 4: Track best predictor
            if p_for_comparison < best_p_value:
                best_p_value = p_for_comparison
                best_predictor = predictor_name
                best_groups = groups
                best_chi_sq = result.statistic
                best_df = result.degrees_of_freedom
                best_p_raw = result.p_value
                best_mult = result.bonferroni_multiplier
                best_type = config.predictor_type
        
        # Check if best predictor is significant
        if best_predictor is None or best_p_value > self.alpha_select:
            return None
        
        # Mark selected predictor
        if best_predictor in node_history.predictor_evaluations:
            node_history.predictor_evaluations[best_predictor].was_selected = True
        
        p_adj = best_p_value if self.apply_bonferroni else None
        
        return (
            best_predictor, best_groups, best_chi_sq, best_df,
            best_p_raw, p_adj, best_mult, best_type
        )
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Predictor variables
            
        Returns:
            Array of predicted class labels (modal category at leaf)
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted. Call fit() first.")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        
        predictions = []
        
        for idx in range(len(X)):
            node = self.root
            row = X.iloc[idx]
            
            while not node.is_leaf:
                child_found = False
                if node.split_variable and node.split_groups:
                    value = row[node.split_variable]
                    for i, group in enumerate(node.split_groups):
                        if value in group:
                            if i < len(node.children):
                                node = node.children[i]
                                child_found = True
                                break
                
                if not child_found:
                    # Value not in any group, use modal category of current node
                    break
            
            predictions.append(node.modal_category)
        
        return np.array(predictions)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Predictor variables
            
        Returns:
            2D array of shape (n_samples, n_classes) with probabilities
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted. Call fit() first.")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        
        n_classes = len(self.y_categories)
        cat_to_idx = {cat: i for i, cat in enumerate(self.y_categories)}
        probas = np.zeros((len(X), n_classes))
        
        for idx in range(len(X)):
            node = self.root
            row = X.iloc[idx]
            
            while not node.is_leaf:
                child_found = False
                if node.split_variable and node.split_groups:
                    value = row[node.split_variable]
                    for i, group in enumerate(node.split_groups):
                        if value in group:
                            if i < len(node.children):
                                node = node.children[i]
                                child_found = True
                                break
                
                if not child_found:
                    break
            
            for cat, prop in node.y_proportions.items():
                if cat in cat_to_idx:
                    probas[idx, cat_to_idx[cat]] = prop
        
        return probas
    
    def get_tree_structure(self) -> Dict[str, Any]:
        """
        Get complete tree structure as dictionary.
        
        Returns:
            Nested dictionary representation of tree
        """
        if self.root is None:
            return {}
        
        def node_to_dict(node: CHAIDNode) -> Dict[str, Any]:
            result = node.to_dict()
            if node.children:
                result["children_data"] = [node_to_dict(c) for c in node.children]
            return result
        
        return {
            "parameters": self.history.parameters,
            "y_categories": list(self.y_categories) if self.y_categories is not None else [],
            "n_nodes": len(self.nodes),
            "tree": node_to_dict(self.root)
        }
    
    def get_split_history(self) -> TreeHistory:
        """
        Get complete history of all splits and evaluations.
        
        Returns:
            TreeHistory object with full traceability
        """
        return self.history
    
    def print_tree(self, node: Optional[CHAIDNode] = None, indent: str = "") -> None:
        """
        Print tree structure to console.
        
        Args:
            node: Starting node (defaults to root)
            indent: Current indentation string
        """
        if node is None:
            if self.root is None:
                print("Tree not fitted.")
                return
            node = self.root
        
        # Print node info
        node_type = "Leaf" if node.is_leaf else "Node"
        print(f"{indent}{node_type} {node.node_id} (n={node.n_observations})")
        
        if node.is_leaf:
            print(f"{indent}   Modal: {node.modal_category} ({node.modal_count}/{node.n_observations})")
        else:
            print(f"{indent}   Split: {node.split_variable}")
            print(f"{indent}   χ²={node.split_chi_square:.2f}, p={node.split_p_value:.4f}")
            if node.split_p_value_adjusted is not None:
                print(f"{indent}   p_adj={node.split_p_value_adjusted:.4f} (×{node.split_bonferroni_multiplier:.0f})")
        
        # Print children
        for i, child in enumerate(node.children):
            if node.split_groups and i < len(node.split_groups):
                group = node.split_groups[i]
                if len(group) == 1:
                    group_str = str(list(group)[0])
                else:
                    group_str = str(set(group))
                print(f"{indent}   ├─ {node.split_variable} = {group_str}")
            
            self.print_tree(child, indent + "   │  ")
    
    def get_leaves(self) -> List[CHAIDNode]:
        """
        Get all leaf nodes.
        
        Returns:
            List of leaf nodes
        """
        return [node for node in self.nodes.values() if node.is_leaf]
    
    def get_depth(self) -> int:
        """
        Get actual depth of tree.
        
        Returns:
            Maximum depth of any node
        """
        if not self.nodes:
            return 0
        return max(node.depth for node in self.nodes.values())
