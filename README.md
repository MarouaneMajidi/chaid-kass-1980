# CHAID - Chi-squared Automatic Interaction Detection

A Python implementation of the CHAID algorithm based on Kass, G.V. (1980) "An exploratory technique for investigating large quantities of categorical data."

## Installation

```bash
pip install numpy pandas scipy matplotlib
```

## Quick Start

```python
import pandas as pd
from chaid import CHAIDTree, PredictorConfig, PredictorType

# Prepare data
data = pd.DataFrame({
    'education': ['High', 'Low', 'Medium', 'High', ...],
    'income': ['<30k', '30-60k', '>60k', ...],
    'outcome': ['Yes', 'No', 'Yes', ...]
})

# Configure predictor types
configs = {
    'education': PredictorConfig('education', PredictorType.ORDINAL, ['Low', 'Medium', 'High']),
    'income': PredictorConfig('income', PredictorType.NOMINAL)
}

# Train tree
tree = CHAIDTree(alpha_merge=0.05, alpha_split=0.05, max_depth=3)
tree.fit(data[['education', 'income']], data['outcome'], predictor_types=configs)

# Visualize
print(tree.print_tree())
```

## Predictor Types

| Type       | Description                 | Pairwise Comparisons         |
| ---------- | --------------------------- | ---------------------------- |
| `NOMINAL`  | Unordered categories        | All pairs                    |
| `ORDINAL`  | Ordered categories          | Adjacent pairs only          |
| `FLOATING` | Ordinal + floating category | Adjacent + floating with all |

### FLOATING Example

```python
# Missing values as floating category
configs = {
    'income': PredictorConfig(
        name='income',
        predictor_type=PredictorType.FLOATING,
        ordered_categories=['<30k', '30-60k', '>60k', 'missing'],
        floating_category='missing'
    )
}
```

## Visualization Functions

```python
from chaid import (
    visualize_tree,
    pairwise_chi_square_table,
    get_successive_merges_table,
    get_predictor_summary_table,
    get_all_pairwise_at_step
)

# Tree visualization
print(tree.print_tree())                           # Text tree
fig = visualize_tree(tree, method="plot")          # Plot
print(visualize_tree(tree, method="rules"))        # Decision rules

# Analysis tables
print(get_predictor_summary_table(tree, node_id=0))                    # Table 10
print(get_all_pairwise_at_step(tree, 0, 'education', step=0))          # Table 3/5/9
_, table = get_successive_merges_table(tree, 0, 'education')           # Table 6
print(table)
```

## Parameters

| Parameter          | Default | Description                                      |
| ------------------ | ------- | ------------------------------------------------ |
| `alpha_merge`      | 0.05    | Significance level for merging (p > α → merge)   |
| `alpha_split`      | 0.05    | Significance level for splitting (p < α → split) |
| `max_depth`        | None    | Maximum tree depth                               |
| `min_parent_size`  | 30      | Minimum samples to attempt split                 |
| `min_child_size`   | 10      | Minimum samples per child node                   |
| `apply_bonferroni` | True    | Apply Bonferroni correction                      |

## Project Structure

```
chaid/
├── __init__.py          # Package exports
├── tree.py              # CHAIDTree, PredictorConfig
├── node.py              # CHAIDNode
├── types.py             # PredictorType enum
├── statistics.py        # Chi-square tests, Bonferroni
├── visualization.py     # All visualization functions
├── history.py           # TreeHistory, NodeHistory
└── merging.py           # Category merging logic
```

## Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- matplotlib

## References

Kass, G.V. (1980). An exploratory technique for investigating large quantities of categorical data. _Applied Statistics_, 29(2), 119-127.

## License

MIT
