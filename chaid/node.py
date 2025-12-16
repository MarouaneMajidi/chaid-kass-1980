"""
CHAID Node representation.

A node in the CHAID tree contains:
- Subset of data (observation indices)
- Split information (if not a leaf)
- Distribution of dependent variable
- Children nodes
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, FrozenSet, Tuple
import numpy as np

from .types import PredictorType
from .history import NodeHistory


@dataclass
class CHAIDNode:
    """
    A node in the CHAID decision tree.
    
    Attributes:
        node_id: Unique identifier for this node
        depth: Depth in the tree (root = 0)
        indices: Indices of observations belonging to this node
        y_distribution: Distribution of dependent variable {category: count}
        y_proportions: Proportions of each category
        n_observations: Total number of observations
        is_leaf: Whether this is a terminal node
        
        # Split information (if not leaf)
        split_variable: Name of variable used for splitting
        split_variable_type: Type of the split variable
        split_groups: Tuple of category groups for each branch
        split_chi_square: Chi-square statistic for the split
        split_degrees_of_freedom: Degrees of freedom
        split_p_value: Raw p-value
        split_p_value_adjusted: Bonferroni-adjusted p-value
        split_bonferroni_multiplier: Bonferroni multiplier used
        
        # Tree structure
        children: List of child nodes
        parent: Reference to parent node (None for root)
        
        # History
        history: Complete history of operations at this node
    """
    node_id: int
    depth: int
    indices: np.ndarray
    y_distribution: Dict[Any, int] = field(default_factory=dict)
    y_proportions: Dict[Any, float] = field(default_factory=dict)
    n_observations: int = 0
    is_leaf: bool = True
    
    # Split information
    split_variable: Optional[str] = None
    split_variable_type: Optional[PredictorType] = None
    split_groups: Optional[Tuple[FrozenSet, ...]] = None
    split_chi_square: float = 0.0
    split_degrees_of_freedom: int = 0
    split_p_value: float = 1.0
    split_p_value_adjusted: Optional[float] = None
    split_bonferroni_multiplier: Optional[float] = None
    
    # Tree structure
    children: List['CHAIDNode'] = field(default_factory=list)
    parent: Optional['CHAIDNode'] = None
    
    # History
    history: Optional[NodeHistory] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        self.n_observations = len(self.indices)
    
    @property
    def modal_category(self) -> Any:
        """Return the most frequent category (modal class)."""
        if not self.y_distribution:
            return None
        return max(self.y_distribution, key=self.y_distribution.get)
    
    @property
    def modal_count(self) -> int:
        """Return count of the modal category."""
        if not self.y_distribution:
            return 0
        return max(self.y_distribution.values())
    
    def get_child_for_value(self, value: Any) -> Optional['CHAIDNode']:
        """
        Get the child node for a given predictor value.
        
        Args:
            value: Value of the split variable
            
        Returns:
            Child node containing this value, or None if not found
        """
        if self.is_leaf or self.split_groups is None:
            return None
        
        for i, group in enumerate(self.split_groups):
            if value in group:
                if i < len(self.children):
                    return self.children[i]
        return None
    
    def get_summary(self) -> str:
        """
        Generate human-readable summary of this node.
        
        Returns:
            Formatted string describing the node
        """
        lines = []
        
        # Header
        node_type = "LEAF" if self.is_leaf else "INTERNAL"
        lines.append(f"Node {self.node_id} [{node_type}] (n={self.n_observations}, depth={self.depth})")
        
        # Distribution
        if self.y_distribution:
            dist_str = ", ".join([f"{k}: {v} ({self.y_proportions.get(k, 0)*100:.1f}%)" 
                                  for k, v in sorted(self.y_distribution.items(), key=str)])
            lines.append(f"  Distribution: {dist_str}")
            lines.append(f"  Modal category: {self.modal_category}")
        
        # Split information
        if not self.is_leaf and self.split_variable:
            lines.append(f"  Split on: {self.split_variable} ({self.split_variable_type.value if self.split_variable_type else 'unknown'})")
            lines.append(f"  χ² = {self.split_chi_square:.4f}, df = {self.split_degrees_of_freedom}")
            lines.append(f"  p-value (raw) = {self.split_p_value:.6f}")
            if self.split_p_value_adjusted is not None:
                lines.append(f"  p-value (adjusted) = {self.split_p_value_adjusted:.6f}")
                lines.append(f"  Bonferroni multiplier = {self.split_bonferroni_multiplier:.1f}")
            if self.split_groups:
                lines.append(f"  Groups: {[set(g) for g in self.split_groups]}")
        
        return "\n".join(lines)
    
    def get_rule(self) -> str:
        """
        Get the rule that defines this node (path from root).
        
        Returns:
            String representation of the path
        """
        if self.parent is None:
            return "Root"
        
        rules = []
        node = self
        while node.parent is not None:
            parent = node.parent
            # Find which group this node belongs to
            for i, child in enumerate(parent.children):
                if child is node:
                    if parent.split_groups and i < len(parent.split_groups):
                        group = parent.split_groups[i]
                        if len(group) == 1:
                            rules.append(f"{parent.split_variable} = {list(group)[0]}")
                        else:
                            rules.append(f"{parent.split_variable} in {set(group)}")
                    break
            node = parent
        
        return " AND ".join(reversed(rules))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to dictionary representation.
        
        Returns:
            Dictionary containing all node information
        """
        result = {
            "node_id": self.node_id,
            "depth": self.depth,
            "n_observations": self.n_observations,
            "is_leaf": self.is_leaf,
            "y_distribution": dict(self.y_distribution),
            "y_proportions": dict(self.y_proportions),
            "modal_category": self.modal_category,
            "rule": self.get_rule(),
        }
        
        if not self.is_leaf:
            result.update({
                "split_variable": self.split_variable,
                "split_variable_type": self.split_variable_type.value if self.split_variable_type else None,
                "split_groups": [list(g) for g in self.split_groups] if self.split_groups else None,
                "split_chi_square": self.split_chi_square,
                "split_degrees_of_freedom": self.split_degrees_of_freedom,
                "split_p_value": self.split_p_value,
                "split_p_value_adjusted": self.split_p_value_adjusted,
                "split_bonferroni_multiplier": self.split_bonferroni_multiplier,
                "children": [child.node_id for child in self.children],
            })
        
        return result
