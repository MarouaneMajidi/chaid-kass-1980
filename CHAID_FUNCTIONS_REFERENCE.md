# CHAID Package - Complete Function Reference

> **Implementation based on Kass, G.V. (1980)**  
> _An exploratory technique for investigating large quantities of categorical data_

---

## Table of Contents

1. [Core Classes](#core-classes)
   - [CHAIDTree](#chaidtree)
   - [PredictorType](#predictortype)
   - [PredictorConfig](#predictorconfig)
2. [Visualization Functions](#visualization-functions)
   - [visualize_tree()](#1-visualize_tree)
   - [pairwise_chi_square_table()](#2-pairwise_chi_square_table)
   - [get_successive_merges_table()](#3-get_successive_merges_table)
   - [get_node_merge_history_detailed()](#4-get_node_merge_history_detailed)
   - [get_all_pairwise_at_step()](#5-get_all_pairwise_at_step)
   - [get_predictor_summary_table()](#6-get_predictor_summary_table)
   - [get_merge_pairwise_table()](#7-get_merge_pairwise_table)
3. [History & Traceability](#history--traceability)
4. [Quick Reference Table](#quick-reference-table)
5. [Statistics Functions](#statistics-functions)
   - [chi_square_test()](#chi_square_test)
   - [bonferroni_multiplier()](#bonferroni_multiplier)
6. [Complete Usage Example](#complete-usage-example)

---

## Core Classes

### `CHAIDTree`

The main class for building CHAID decision trees.

#### Constructor Parameters

| Parameter          | Type     | Default | Description                                                 |
| ------------------ | -------- | ------- | ----------------------------------------------------------- |
| `alpha_merge`      | float    | 0.05    | Threshold for merging categories (p > α → merge)            |
| `alpha_split`      | float    | 0.049   | Threshold for splitting compound categories (p < α → split) |
| `alpha_select`     | float    | 0.05    | Threshold for selecting a predictor (p < α → significant)   |
| `max_depth`        | int/None | None    | Maximum tree depth (None = unlimited)                       |
| `min_parent_size`  | int      | 30      | Minimum observations to attempt split                       |
| `min_child_size`   | int      | 10      | Minimum observations per child node                         |
| `apply_bonferroni` | bool     | True    | Apply Bonferroni correction                                 |

#### Methods

```python
from chaid import CHAIDTree, PredictorType, PredictorConfig

# Initialize
tree = CHAIDTree(
    alpha_merge=0.05,
    alpha_split=0.05,
    max_depth=3,
    min_parent_size=30,
    min_child_size=10
)

# Fit the tree (simple - just predictor types)
tree.fit(X, y, predictor_types={
    'education': PredictorType.NOMINAL,
    'age': PredictorType.ORDINAL
})

# Fit the tree (advanced - with PredictorConfig for ordering)
tree.fit(X, y, predictor_types={
    'study_time': PredictorConfig(
        name='study_time',
        predictor_type=PredictorType.ORDINAL,
        ordered_categories=['<2h', '2-5h', '5-10h', '>10h']
    ),
    'absence': PredictorConfig(
        name='absence',
        predictor_type=PredictorType.FLOATING,
        ordered_categories=['None', 'Low', 'Medium', 'High', 'miss'],
        floating_category='miss'
    )
})

# Predictions
predictions = tree.predict(X_test)           # Returns predicted classes
probabilities = tree.predict_proba(X_test)   # Returns probability matrix

# Tree information
tree.get_depth()           # Returns max depth
tree.get_leaves()          # Returns list of leaf nodes
tree.print_tree()          # Prints tree structure
tree.get_tree_structure()  # Returns dict with full tree info
tree.get_split_history()   # Returns TreeHistory object
```

---

### `PredictorType`

Enum for specifying predictor variable types.

```python
from chaid import PredictorType

PredictorType.NOMINAL   # Any pair can be merged (e.g., colors, regions)
PredictorType.ORDINAL   # Only adjacent categories can be merged (e.g., education levels)
PredictorType.FLOATING  # Ordinal with one "floating" category that can merge with any (e.g., "missing")
```

#### Comparison of Predictor Types

| Type       | Mergeable Pairs          | Use Case                              | Example                   |
| ---------- | ------------------------ | ------------------------------------- | ------------------------- |
| `NOMINAL`  | All pairs (i, j)         | Unordered categories                  | {Red, Blue, Green}        |
| `ORDINAL`  | Adjacent pairs (i, i+1)  | Ordered categories                    | {Low, Medium, High}       |
| `FLOATING` | Adjacent + floating pair | Ordinal with a special "any" category | {Low, Medium, High, miss} |

---

### `PredictorConfig`

Dataclass for detailed predictor configuration, especially for ORDINAL and FLOATING types.

```python
from chaid import PredictorConfig, PredictorType

# NOMINAL predictor (simple)
config_nominal = PredictorConfig(
    name='gender',
    predictor_type=PredictorType.NOMINAL
)

# ORDINAL predictor with explicit category order
config_ordinal = PredictorConfig(
    name='study_time',
    predictor_type=PredictorType.ORDINAL,
    ordered_categories=['<2h', '2-5h', '5-10h', '>10h']  # Natural order!
)

# FLOATING predictor with floating category
config_floating = PredictorConfig(
    name='absence_level',
    predictor_type=PredictorType.FLOATING,
    ordered_categories=['None', 'Low', 'Medium', 'High', 'miss'],
    floating_category='miss'  # Can merge with ANY category
)
```

#### Parameters

| Parameter            | Type          | Required | Description                                           |
| -------------------- | ------------- | -------- | ----------------------------------------------------- |
| `name`               | str           | Yes      | Column name in the data                               |
| `predictor_type`     | PredictorType | Yes      | NOMINAL, ORDINAL, or FLOATING                         |
| `ordered_categories` | List          | No\*     | Category order for ORDINAL/FLOATING (\*recommended)   |
| `floating_category`  | Any           | No\*     | The floating category value (\*required for FLOATING) |

#### Why Use PredictorConfig?

Without `PredictorConfig`, ordinal categories are sorted **alphabetically**, which may not be correct:

- Alphabetical: `['2-5h', '5-10h', '<2h', '>10h']` ❌
- Natural order: `['<2h', '2-5h', '5-10h', '>10h']` ✓

```python
# ❌ Without PredictorConfig - alphabetical order
predictor_types = {'study_time': PredictorType.ORDINAL}

# ✓ With PredictorConfig - correct natural order
predictor_types = {
    'study_time': PredictorConfig(
        name='study_time',
        predictor_type=PredictorType.ORDINAL,
        ordered_categories=['<2h', '2-5h', '5-10h', '>10h']
    )
}
```

---

## Visualization Functions

### 1. `visualize_tree()`

**Main visualization function** - generates text or plot representations.

#### Parameters

| Parameter           | Type      | Default  | Description                                   |
| ------------------- | --------- | -------- | --------------------------------------------- |
| `tree`              | CHAIDTree | required | Fitted CHAID tree                             |
| `method`            | str       | "text"   | `"text"`, `"summary"`, `"rules"`, or `"plot"` |
| `show_distribution` | bool      | True     | Show Y distribution at nodes (text mode)      |
| `show_statistics`   | bool      | True     | Show χ², df, p-value (text mode)              |
| `figsize`           | tuple     | (16, 12) | Figure size for plot mode                     |
| `title`             | str       | None     | Title for plot                                |
| `show_bars`         | bool      | True     | Show distribution bars in plot nodes          |

#### Returns

- **Text modes**: `str`
- **Plot mode**: `matplotlib.Figure`

#### Usage

```python
from chaid import visualize_tree

# Method 1: Text-based ASCII tree
text = visualize_tree(tree, method="text", show_distribution=True, show_statistics=True)
print(text)

# Method 2: Summary table
summary = visualize_tree(tree, method="summary")
print(summary)

# Method 3: Decision rules
rules = visualize_tree(tree, method="rules")
print(rules)

# Method 4: Matplotlib plot (CHAID-style)
fig = visualize_tree(tree, method="plot", figsize=(16, 12),
                     title="CHAID Decision Tree", show_bars=False)
fig.savefig("tree.png", dpi=150, bbox_inches='tight')
```

#### Example Result - `method="text"`

```
================================================================================
CHAID DECISION TREE
================================================================================
Nodes: 9
Depth: 2
Leaves: 6
================================================================================

└── [Node 0] 📊 SPLIT (n=400)
    Modal: Success (222/400 = 55.5%)
    Dist: Fail: 178 (44%) | Success: 222 (56%)
    Split variable: education (nominal)
    χ² = 39.6593, df = 2
    p-value = 0.000000
    p-value (adjusted) = 0.000000 (×6)
    Groups:
      [1] {General, Technical} → Node 1 (n=235)
      [2] None → Node 5 (n=61)
      [3] Vocational → Node 6 (n=104)

    ├── [Node 1] 📊 SPLIT (n=235)
    │   Modal: Success (158/235 = 67.2%)
    │   Split variable: study_hours (ordinal)
    │   χ² = 12.2382, df = 2
    │   ...
```

#### Example Result - `method="rules"`

```
DECISION RULES
==============

Rule 1: IF education IN {General, Technical} AND study_hours IN {10-20h, 20-30h}
        THEN outcome = Success (confidence: 68.0%, n=153)

Rule 2: IF education IN {General, Technical} AND study_hours = <10h
        THEN outcome = Fail (confidence: 54.1%, n=37)

Rule 3: IF education IN {General, Technical} AND study_hours = >30h
        THEN outcome = Success (confidence: 82.2%, n=45)

Rule 4: IF education = None
        THEN outcome = Fail (confidence: 75.4%, n=61)
...
```

---

### 2. `pairwise_chi_square_table()`

**Generates Table 3/5/9 from the thesis** - Pairwise χ² values (upper triangle) and p-values (lower triangle).

Supports all three predictor types:

- **NOMINAL**: All pairs computed
- **ORDINAL**: Only adjacent pairs computed (others blank)
- **FLOATING**: Adjacent pairs + floating category pairs with all

#### Parameters

| Parameter            | Type              | Default   | Description                               |
| -------------------- | ----------------- | --------- | ----------------------------------------- |
| `X`                  | array-like        | required  | Predictor variable values                 |
| `y`                  | array-like        | required  | Dependent variable values                 |
| `predictor_type`     | str/PredictorType | "nominal" | Type: "nominal", "ordinal", or "floating" |
| `ordered_categories` | List              | None      | Category order for ordinal/floating       |
| `floating_category`  | Any               | None      | The floating category value               |
| `category_labels`    | dict              | None      | Optional mapping of categories to labels  |

#### Returns

`Tuple[pd.DataFrame, str]` - DataFrame with data and formatted string for display

#### Usage

```python
from chaid import pairwise_chi_square_table, PredictorType

# NOMINAL predictor - all pairs computed
df, table_str = pairwise_chi_square_table(
    X=data['education'],
    y=data['outcome'],
    predictor_type=PredictorType.NOMINAL
)
print(table_str)

# ORDINAL predictor - only adjacent pairs computed
df, table_str = pairwise_chi_square_table(
    X=data['study_time'],
    y=data['outcome'],
    predictor_type=PredictorType.ORDINAL,
    ordered_categories=['<2h', '2-5h', '5-10h', '>10h']
)
print(table_str)

# FLOATING predictor - adjacent + floating category pairs
df, table_str = pairwise_chi_square_table(
    X=data['absence_level'],
    y=data['outcome'],
    predictor_type=PredictorType.FLOATING,
    ordered_categories=['None', 'Low', 'Medium', 'High', 'miss'],
    floating_category='miss'
)
print(table_str)
```

#### Example Result - NOMINAL

```
Table: Chi-squares and p-values by pair of categories (NOMINAL predictor)
==========================================================================

         General      None  Technical  Vocational
General        0     37.80       0.83        9.88
   None   0.0000         0      16.12       10.02
Technical 0.3633    0.0001          0        8.22
Vocational 0.0071   0.0067     0.0042           0

Chi-squares are above the diagonal and p-values below the diagonal.

→ Most similar pair: [General] and [Technical] (χ²=0.83, p=0.3633)
```

#### Example Result - ORDINAL (Only Adjacent Pairs)

```
Table: Chi-squares and p-values by pair of categories (ORDINAL predictor)
==========================================================================

           <2h      2-5h     5-10h      >10h
   <2h       0      6.67
  2-5h  0.0098         0     19.73
 5-10h            0.0000         0      0.29
  >10h                      0.5927         0

Chi-squares are above the diagonal and p-values below the diagonal.

→ Most similar pair: [5-10h] and [>10h] (χ²=0.29, p=0.5927)
```

#### Example Result - FLOATING (Table 9 Style)

```
Table: Chi-squares and p-values by pair of categories (FLOATING predictor)
==========================================================================

          None       Low    Medium      High      miss
  None       0      0.09                          1.47
   Low  0.7652         0      2.05                0.95
Medium            0.1518         0      0.39      0.07
  High                      0.5310         0      0.64
  miss  0.2247    0.3309    0.7917    0.4239         0

Chi-squares are above the diagonal and p-values below the diagonal.

→ Most similar pair: [Medium] and [miss] (χ²=0.07, p=0.7917)
```

**Key Difference**: Notice how in FLOATING mode, the `miss` row/column has values for ALL categories (it can merge with any), while other categories only have values for adjacent pairs.

---

### 3. `get_successive_merges_table()`

**Generates Table 6 from the thesis** - Shows each iteration of the merge/split process.

#### Parameters

| Parameter        | Type      | Default  | Description                                |
| ---------------- | --------- | -------- | ------------------------------------------ |
| `tree`           | CHAIDTree | required | Fitted CHAID tree                          |
| `node_id`        | int       | 0        | Node to examine (0 = root)                 |
| `predictor`      | str       | None     | Predictor name (None = use split variable) |
| `include_splits` | bool      | True     | Include split check iterations             |

#### Returns

`Tuple[pd.DataFrame, str]` - DataFrame with data and formatted string for display

#### Usage

```python
from chaid import get_successive_merges_table

df, table_str = get_successive_merges_table(
    tree,
    node_id=0,
    predictor='education',
    include_splits=True
)

print(table_str)
```

#### Example Result

```
╔════════════════════════════════════════════════════════════════════════════╗
║                          TABLE: Successive Merges                          ║
║                       Predictor: education at Node 0                       ║
╠════════════════════════════════════════════════════════════════════════════╣
║ Iteration  │     Merge/Split      │  Chi-square  │   p-value    │  Decision  ║
╠════════════════════════════════════════════════════════════════════════════╣
║     1      │        {1,3}         │     0.83     │    36.33%    │   Merged   ║
║     2      │        {2,4}         │     8.22     │    0.42%     │  Stopped   ║
╚════════════════════════════════════════════════════════════════════════════╝

Category Labels:
  [1] = General
  [2] = None
  [3] = Technical
  [4] = Vocational

Summary: 1 merge(s), 0 split(s), 1 stop(s)
Final groups: 3

Final category groupings:
  Group 1: {1,3} = {General, Technical}
  Group 2: 2 = {None}
  Group 3: 4 = {Vocational}
```

#### Example with Split Checks (Node 6)

```
╔════════════════════════════════════════════════════════════════════════════╗
║                          TABLE: Successive Merges                          ║
║                      Predictor: study_hours at Node 6                      ║
╠════════════════════════════════════════════════════════════════════════════╣
║ Iteration  │     Merge/Split      │  Chi-square  │   p-value    │  Decision  ║
╠════════════════════════════════════════════════════════════════════════════╣
║     1      │        {1,2}         │     0.16     │    68.96%    │   Merged   ║
║     2      │      {{1,2},3}       │     0.99     │    32.07%    │   Merged   ║
║     3      │     {{1,2,3},4}      │    13.49     │    0.02%     │  Stopped   ║
║     4      │       {1,2,3}        │     0.99     │    32.07%    │    Kept    ║
╚════════════════════════════════════════════════════════════════════════════╝

Category Labels:
  [1] = 10-20h
  [2] = 20-30h
  [3] = <10h
  [4] = >30h

Summary: 2 merge(s), 0 split(s), 2 stop(s)
Final groups: 2

Final category groupings:
  Group 1: {1,2,3} = {10-20h, 20-30h, <10h}
  Group 2: 4 = {>30h}
```

**Decision Types**:

- **Merged**: p > α_merge → categories are similar, merge them
- **Stopped**: p ≤ α_merge → categories are different, stop merging
- **Split**: p < α_split → compound category should be split
- **Kept**: p ≥ α_split → compound category stays merged

---

### 4. `get_node_merge_history_detailed()`

**Step-by-step merge process** with full details at each step.

#### Parameters

| Parameter   | Type      | Default  | Description                                |
| ----------- | --------- | -------- | ------------------------------------------ |
| `tree`      | CHAIDTree | required | Fitted CHAID tree                          |
| `node_id`   | int       | 0        | Node to examine                            |
| `predictor` | str       | None     | Predictor name (None = use split variable) |

#### Returns

`str` - Detailed formatted output

#### Usage

```python
from chaid import get_node_merge_history_detailed

output = get_node_merge_history_detailed(tree, node_id=0, predictor='education')
print(output)
```

#### Example Result

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           DETAILED MERGE HISTORY                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Node: 0          Predictor: education            Type: nominal              ║
║  Alpha (merge): 0.05                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

INITIAL STATE: 4 categories
------------------------------------------------------------
  [1] General
  [2] None
  [3] Technical
  [4] Vocational

==============================================================================
STEP 1: Testing merge possibilities
==============================================================================

Current groups (4):
  [1] General
  [2] None
  [3] Technical
  [4] Vocational

Most similar pair (lowest χ²):
  Groups [1] and [3]

┌──────────────────────────────────────────────────────┐
│  Chi-square test for pair [1] vs [3]                 │
├──────────────────────────────────────────────────────┤
│  χ² =     0.8265                                     │
│  df =          1                                     │
│  p  =   0.363300                                     │
├──────────────────────────────────────────────────────┤
│  DECISION: MERGE (p = 0.3633 > α = 0.05)             │
│  → Categories are statistically similar              │
└──────────────────────────────────────────────────────┘

  Result: [1] + [3] → {General, Technical}

==============================================================================
STEP 2: Testing merge possibilities
==============================================================================

Current groups (3):
  [1] {General, Technical}
  [2] None
  [3] Vocational

Most similar pair (lowest χ²):
  Groups [2] and [3]

┌──────────────────────────────────────────────────────┐
│  Chi-square test for pair [2] vs [3]                 │
├──────────────────────────────────────────────────────┤
│  χ² =     8.2164                                     │
│  df =          1                                     │
│  p  =   0.004200                                     │
├──────────────────────────────────────────────────────┤
│  DECISION: STOP (p = 0.0042 ≤ α = 0.05)              │
│  → All remaining pairs significantly different       │
└──────────────────────────────────────────────────────┘

==============================================================================
FINAL RESULT
==============================================================================
Converged after 2 iterations

Final groups (3):
  [1] {General, Technical}
  [2] None
  [3] Vocational
```

---

### 5. `get_all_pairwise_at_step()`

**Pairwise table at a specific merge step** - see how the table evolves during the merge process.

Respects predictor types:

- **NOMINAL**: Shows all pairs
- **ORDINAL**: Shows only adjacent pairs
- **FLOATING**: Shows adjacent pairs + all pairs involving the floating category

#### Parameters

| Parameter   | Type      | Default  | Description                     |
| ----------- | --------- | -------- | ------------------------------- |
| `tree`      | CHAIDTree | required | Fitted CHAID tree               |
| `node_id`   | int       | 0        | Node ID                         |
| `predictor` | str       | required | Predictor name                  |
| `step`      | int       | 0        | Step number (0 = initial state) |

#### Returns

`str` - Pairwise table at that step

#### Usage

```python
from chaid import get_all_pairwise_at_step

# Initial state (before any merging)
print(get_all_pairwise_at_step(tree, node_id=0, predictor='education', step=0))

# After first merge
print(get_all_pairwise_at_step(tree, node_id=0, predictor='education', step=1))
```

#### Example Result - ORDINAL Step 0 (Initial)

```
Initial pairwise table (before any merging)
===============================================================

Groups:
  [1] <2h
  [2] 2-5h
  [3] 5-10h
  [4] >10h

                 1           2           3           4
------------------------------------------------------
     1           —       10.02
     2      0.0015           —       16.14
     3                  0.0001           —        0.05
     4                              0.8288           —
------------------------------------------------------
Upper triangle: χ² values | Lower triangle: p-values
(Only adjacent pairs computed for ORDINAL predictor)

→ Most similar pair: [3] and [4] (χ²=0.05, p=0.8288)
```

#### Example Result - FLOATING Step 0 (Initial)

```
Initial pairwise table (before any merging)
===========================================================================

Groups:
  [1] None
  [2] Low
  [3] Medium
  [4] High
  [5] miss

                 1           2           3           4           5
------------------------------------------------------------------
     1           —        0.09                                1.47
     2      0.7652           —        2.05                    0.95
     3                  0.1518           —        0.39        0.07
     4                              0.5310           —        0.64
     5      0.2247      0.3309      0.7917      0.4239           —
------------------------------------------------------------------
Upper triangle: χ² values | Lower triangle: p-values
(Adjacent + floating category pairs computed for FLOATING predictor)

→ Most similar pair: [3] and [5] (χ²=0.07, p=0.7917)
```

#### Example Result - Step 1 (After first merge)

```
Pairwise table after step 1
============================================

Groups:
  [1] {General, Technical}
  [2] None
  [3] Vocational

           1           2           3
----------------------------------------
     1     —       39.66        8.22
     2  0.0000         —        8.22
     3  0.0042    0.0042           —
----------------------------------------
Upper triangle: χ² values | Lower triangle: p-values

→ Most similar pair: [2] and [3] (χ²=8.22, p=0.0042)
```

---

### 6. `get_predictor_summary_table()`

**Generates Table 10 from the thesis** - Summary of all predictors evaluated at a node, showing which one was selected for splitting and why.

#### Parameters

| Parameter          | Type      | Default  | Description                           |
| ------------------ | --------- | -------- | ------------------------------------- |
| `tree`             | CHAIDTree | required | Fitted CHAID tree                     |
| `node_id`          | int       | 0        | Node to examine (0 = root)            |
| `return_dataframe` | bool      | False    | If True, also return pandas DataFrame |

#### Returns

- **Default**: `str` - Formatted table string
- **With `return_dataframe=True`**: `Tuple[pd.DataFrame, str]`

#### Usage

```python
from chaid import get_predictor_summary_table

# Get formatted table
print(get_predictor_summary_table(tree, node_id=0))

# Get DataFrame for further analysis
df, table_str = get_predictor_summary_table(tree, node_id=0, return_dataframe=True)
print(df)
```

#### Example Result

```
╔════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                          TABLE 10: Summary of Possible First Level Splits                          ║
╠════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ Predictor            │ Type     │  #cat │  #grp │     Chi-sq │  df │        p-value │ Selected ║
╠────────────────────────────────────────────────────────────────────────────────────────────────────╣
║ failures_cat         │ ORDINAL  │     4 │     2 │     102.34 │   1 │   0.0000000000 │        ★ ║
║ higher               │ NOMINAL  │     2 │     2 │      62.25 │   1 │   0.0000000000 │          ║
║ study_time           │ ORDINAL  │     4 │     3 │      19.25 │   2 │       0.000198 │          ║
║ mother_education     │ ORDINAL  │     4 │     2 │      10.09 │   1 │       0.004474 │          ║
║ internet             │ NOMINAL  │     2 │     2 │       5.05 │   1 │       0.024620 │          ║
║ sex                  │ NOMINAL  │     2 │     2 │       3.97 │   1 │       0.046289 │          ║
║ absence_level        │ FLOATING │     5 │     2 │       5.60 │   1 │       0.125502 │          ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════╝

★ Selected predictor: failures_cat
  Selection reason: Lowest p-value: 0.000000

  Final groups after merging:
    Group 1: {0}
    Group 2: {1, 2, 3+}
```

#### DataFrame Output

```python
df, _ = get_predictor_summary_table(tree, node_id=0, return_dataframe=True)
print(df.to_string(index=False))
```

```
       Predictor     Type  #categories  #groups  Chi-square  df      p-value  adj. p-value  Selected
    failures_cat  ORDINAL            4        2  102.341750   1 0.000000e+00  0.000000e+00      True
          higher  NOMINAL            2        2   62.251385   1 2.997602e-15  2.997602e-15     False
      study_time  ORDINAL            4        3   19.251296   2 6.601373e-05  1.980412e-04     False
mother_education  ORDINAL            4        2   10.089299   1 1.491329e-03  4.473986e-03     False
        internet  NOMINAL            2        2    5.050400   1 2.462026e-02  2.462026e-02     False
             sex  NOMINAL            2        2    3.971035   1 4.628932e-02  4.628932e-02     False
   absence_level FLOATING            5        2    5.603092   1 1.792881e-02  1.255017e-01     False
```

**Table Columns Explained**:

| Column    | Description                                   |
| --------- | --------------------------------------------- |
| Predictor | Variable name                                 |
| Type      | NOMINAL, ORDINAL, or FLOATING                 |
| #cat      | Original number of categories                 |
| #grp      | Final number of groups after merging          |
| Chi-sq    | Chi-square statistic for the final grouping   |
| df        | Degrees of freedom                            |
| p-value   | Adjusted p-value (with Bonferroni if enabled) |
| Selected  | ★ if this predictor was chosen for the split  |

---

### 7. `get_merge_pairwise_table()`

**Quick merge analysis summary** - shows merge steps for a predictor at a node.

#### Parameters

| Parameter   | Type      | Default  | Description                                |
| ----------- | --------- | -------- | ------------------------------------------ |
| `tree`      | CHAIDTree | required | Fitted CHAID tree                          |
| `node_id`   | int       | 0        | Node to examine                            |
| `predictor` | str       | None     | Predictor name (None = use split variable) |

#### Returns

`Tuple[None, str]` - (None, formatted string)

#### Usage

```python
from chaid import get_merge_pairwise_table

_, output = get_merge_pairwise_table(tree, node_id=0, predictor='education')
print(output)
```

#### Example Result

```
Pairwise Merge Analysis for Node 0, Predictor: education
======================================================================
Original categories: ['General', 'None', 'Technical', 'Vocational']
Final groups: [['General', 'Technical'], ['None'], ['Vocational']]

Merge steps:
  1. MERGED: χ²=0.8265, p=0.3633
      Pair: {'General'} + {'Technical'}
  2. STOPPED: χ²=8.2164, p=0.0042
      Pair: {'None'} + {'Vocational'}
```

---

## History & Traceability

### `TreeHistory`

Access via `tree.get_split_history()`.

```python
history = tree.get_split_history()

# Properties
history.parameters          # Dict of tree parameters used
history.construction_order  # Order nodes were processed

# Methods
node_hist = history.get_node_history(node_id=0)  # Get history for specific node
summary = history.get_full_summary()              # Complete summary string
```

### `NodeHistory`

Access via `history.get_node_history(node_id)`.

```python
node_hist = history.get_node_history(0)

# Properties
node_hist.node_id                    # Node ID
node_hist.depth                      # Depth in tree
node_hist.n_observations             # Sample size at this node
node_hist.predictor_evaluations      # Dict[str, PredictorEvaluation]
node_hist.selected_predictor         # Name of chosen predictor (or None)
node_hist.selected_groups            # Final groups used for split
node_hist.is_leaf                    # True if terminal node
node_hist.leaf_reason                # Why it became a leaf

# Methods
node_hist.get_summary()              # Human-readable summary
```

### `PredictorEvaluation`

Access via `node_hist.predictor_evaluations['predictor_name']`.

```python
eval_rec = node_hist.predictor_evaluations['education']

# Properties
eval_rec.predictor_name          # Variable name
eval_rec.predictor_type          # NOMINAL, ORDINAL, or FLOATING
eval_rec.original_categories     # Tuple of original categories
eval_rec.final_groups            # Tuple of FrozenSets after merging
eval_rec.merge_history           # List[MergeRecord] - all merge operations
eval_rec.split_check_history     # List[SplitCheckRecord] - split checks
eval_rec.final_chi_square        # χ² for final grouping
eval_rec.final_degrees_of_freedom # df
eval_rec.final_p_value           # Raw p-value
eval_rec.bonferroni_multiplier   # Correction factor (m)
eval_rec.adjusted_p_value        # Corrected p-value (p × m)
eval_rec.was_selected            # True if this predictor was chosen for split
```

### `MergeRecord`

Each merge operation is recorded:

```python
for merge in eval_rec.merge_history:
    merge.categories_before   # Groups before merge
    merge.categories_after    # Groups after merge
    merge.merged_pair         # The two groups that were tested
    merge.chi_square          # χ² statistic
    merge.degrees_of_freedom  # df
    merge.p_value             # p-value
    merge.was_merged          # True if merge happened
    merge.reason              # Explanation string
```

### `SplitCheckRecord`

Each split check on compound categories:

```python
for split in eval_rec.split_check_history:
    split.compound_category   # The compound category checked
    split.best_split          # Best dichotomy found (or None)
    split.chi_square          # χ² for best split
    split.degrees_of_freedom  # df
    split.p_value             # p-value
    split.was_split           # True if split was implemented
    split.reason              # Explanation string
```

---

## Quick Reference Table

| Function                            | Purpose                              | Returns                     |
| ----------------------------------- | ------------------------------------ | --------------------------- |
| `visualize_tree(method="text")`     | ASCII tree with statistics           | `str`                       |
| `visualize_tree(method="plot")`     | CHAID-style tree plot                | `Figure`                    |
| `visualize_tree(method="summary")`  | Node summary table                   | `str`                       |
| `visualize_tree(method="rules")`    | Decision rules                       | `str`                       |
| `pairwise_chi_square_table()`       | **Table 3/5/9**: χ²/p-value matrix   | `(DataFrame, str)`          |
| `get_successive_merges_table()`     | **Table 6**: Iteration history       | `(DataFrame, str)`          |
| `get_node_merge_history_detailed()` | Step-by-step merge details           | `str`                       |
| `get_all_pairwise_at_step()`        | Pairwise table at step N             | `str`                       |
| `get_predictor_summary_table()`     | **Table 10**: Predictor comparison   | `str` or `(DataFrame, str)` |
| `get_merge_pairwise_table()`        | Quick merge analysis summary         | `(None, str)`               |
| `chi_square_test()`                 | Chi-square test on contingency table | `ChiSquareResult`           |
| `bonferroni_multiplier()`           | Compute Bonferroni correction factor | `float`                     |
| `tree.predict()`                    | Class predictions                    | `ndarray`                   |
| `tree.predict_proba()`              | Probability matrix                   | `ndarray`                   |
| `tree.get_split_history()`          | Full traceability object             | `TreeHistory`               |

---

## Statistics Functions

### `chi_square_test()`

**Perform chi-square test of independence** on a contingency table.

#### Parameters

| Parameter           | Type          | Default  | Description                                   |
| ------------------- | ------------- | -------- | --------------------------------------------- |
| `contingency_table` | np.ndarray    | required | c × d matrix of observed frequencies          |
| `predictor_type`    | PredictorType | None     | Type of predictor (for Bonferroni correction) |
| `c_original`        | int           | None     | Original number of categories before merging  |
| `g_final`           | int           | None     | Final number of groups after merging          |
| `apply_bonferroni`  | bool          | False    | Whether to apply Bonferroni correction        |

#### Returns

`ChiSquareResult` - NamedTuple containing:

- `statistic`: Chi-square test statistic
- `degrees_of_freedom`: Degrees of freedom
- `p_value`: Raw p-value
- `p_value_adjusted`: Bonferroni-adjusted p-value (if requested)
- `expected_frequencies`: Expected frequency matrix
- `observed_frequencies`: Observed frequency matrix
- `bonferroni_multiplier`: The multiplier used (if applied)

#### Usage

```python
import numpy as np
from chaid import chi_square_test, PredictorType

# Create contingency table (2x3: 2 groups × 3 outcomes)
table = np.array([
    [50, 30, 20],
    [30, 40, 30]
])

# Basic test
result = chi_square_test(table)
print(f"χ² = {result.statistic:.4f}")
print(f"df = {result.degrees_of_freedom}")
print(f"p-value = {result.p_value:.6f}")

# With Bonferroni correction
result = chi_square_test(
    table,
    predictor_type=PredictorType.ORDINAL,
    c_original=5,
    g_final=2,
    apply_bonferroni=True
)
print(f"Adjusted p-value = {result.p_value_adjusted:.6f}")
print(f"Multiplier = {result.bonferroni_multiplier}")
```

---

### `bonferroni_multiplier()`

**Compute the Bonferroni correction factor** based on the number of possible partitions.

The multiplier equals the number of ways to partition `c` categories into `g` groups, respecting the predictor type constraints (as defined in Kass 1980).

#### Parameters

| Parameter        | Type          | Required | Description                   |
| ---------------- | ------------- | -------- | ----------------------------- |
| `c`              | int           | required | Original number of categories |
| `g`              | int           | required | Final number of groups        |
| `predictor_type` | PredictorType | required | Type of predictor             |

#### Formulas

| Predictor Type | Formula                                       | Description                 |
| -------------- | --------------------------------------------- | --------------------------- |
| NOMINAL        | $S(c, g)$                                     | Stirling number of 2nd kind |
| ORDINAL        | $\binom{c-1}{g-1}$                            | Contiguous partitions       |
| FLOATING       | $\binom{c-2}{g-2} + g \cdot \binom{c-2}{g-1}$ | Ordinal + floating category |

#### Returns

`float` - The Bonferroni multiplier

#### Usage

```python
from chaid import bonferroni_multiplier, PredictorType

# Nominal: 4 categories into 2 groups
m = bonferroni_multiplier(4, 2, PredictorType.NOMINAL)
print(f"NOMINAL: S(4,2) = {m}")  # 7

# Ordinal: 4 categories into 2 groups
m = bonferroni_multiplier(4, 2, PredictorType.ORDINAL)
print(f"ORDINAL: C(3,1) = {m}")  # 3

# Floating: 5 categories (4 ordinal + 1 floating) into 2 groups
m = bonferroni_multiplier(5, 2, PredictorType.FLOATING)
print(f"FLOATING: C(3,0) + 2×C(3,1) = {m}")  # 1 + 6 = 7
```

---

### All Exports

```python
from chaid import (
    # Core classes
    CHAIDTree,
    CHAIDNode,
    PredictorConfig,
    PredictorType,

    # Visualization functions
    TreeVisualizer,
    visualize_tree,
    pairwise_chi_square_table,
    get_merge_pairwise_table,
    get_node_merge_history_detailed,
    get_all_pairwise_at_step,
    get_successive_merges_table,
    get_predictor_summary_table,

    # History classes
    TreeHistory,
    NodeHistory,
    PredictorEvaluation,

    # Statistics
    chi_square_test,
    bonferroni_multiplier
)
```

---

## Complete Usage Example

```python
"""
Complete CHAID Example with All Predictor Types
"""
import numpy as np
import pandas as pd
from chaid import (
    CHAIDTree,
    PredictorType,
    PredictorConfig,
    visualize_tree,
    pairwise_chi_square_table,
    get_successive_merges_table,
    get_node_merge_history_detailed,
    get_all_pairwise_at_step,
    get_predictor_summary_table
)

# =============================================================================
# 1. PREPARE DATA
# =============================================================================
np.random.seed(42)
n = 500

data = pd.DataFrame({
    # NOMINAL predictor
    'education': np.random.choice(['General', 'Technical', 'Vocational', 'None'], n),
    # ORDINAL predictor
    'study_time': np.random.choice(['<2h', '2-5h', '5-10h', '>10h'], n),
    # FLOATING predictor (ordinal + missing)
    'absences': np.random.choice(['None', 'Low', 'Medium', 'High', 'miss'], n),
    # Outcome
    'outcome': np.random.choice(['Pass', 'Fail'], n)
})

X = data[['education', 'study_time', 'absences']]
y = data['outcome']

# =============================================================================
# 2. CONFIGURE PREDICTORS WITH PredictorConfig
# =============================================================================
predictor_configs = {
    # NOMINAL - no ordering needed
    'education': PredictorConfig(
        name='education',
        predictor_type=PredictorType.NOMINAL
    ),

    # ORDINAL - specify natural order
    'study_time': PredictorConfig(
        name='study_time',
        predictor_type=PredictorType.ORDINAL,
        ordered_categories=['<2h', '2-5h', '5-10h', '>10h']
    ),

    # FLOATING - ordinal with floating "miss" category
    'absences': PredictorConfig(
        name='absences',
        predictor_type=PredictorType.FLOATING,
        ordered_categories=['None', 'Low', 'Medium', 'High', 'miss'],
        floating_category='miss'
    )
}

# =============================================================================
# 3. TRAIN CHAID TREE
# =============================================================================
tree = CHAIDTree(
    alpha_merge=0.05,
    alpha_split=0.05,
    max_depth=3,
    min_parent_size=30,
    min_child_size=15,
    apply_bonferroni=True
)

tree.fit(X, y, predictor_types=predictor_configs)

print(f"Tree trained: {len(tree.nodes)} nodes, depth {tree.get_depth()}")

# =============================================================================
# 4. VISUALIZE TREE
# =============================================================================

# Text visualization
print(tree.print_tree())

# Plot (save to file)
fig = visualize_tree(tree, method="plot", figsize=(16, 12),
                     title="CHAID Tree", show_bars=False)
fig.savefig("chaid_tree.png", dpi=150, bbox_inches='tight')

# Decision rules
print(visualize_tree(tree, method="rules"))

# =============================================================================
# 5. PREDICTOR SUMMARY TABLE (Table 10)
# =============================================================================
print(get_predictor_summary_table(tree, node_id=0))

# Get as DataFrame for analysis
df_summary, _ = get_predictor_summary_table(tree, node_id=0, return_dataframe=True)
print(df_summary)

# =============================================================================
# 6. PAIRWISE CHI-SQUARE TABLES (Table 3/5/9)
# =============================================================================

# NOMINAL predictor - all pairs
df, table = pairwise_chi_square_table(
    data['education'], data['outcome'],
    predictor_type=PredictorType.NOMINAL
)
print(table)

# ORDINAL predictor - only adjacent pairs
df, table = pairwise_chi_square_table(
    data['study_time'], data['outcome'],
    predictor_type=PredictorType.ORDINAL,
    ordered_categories=['<2h', '2-5h', '5-10h', '>10h']
)
print(table)

# FLOATING predictor - adjacent + floating pairs
df, table = pairwise_chi_square_table(
    data['absences'], data['outcome'],
    predictor_type=PredictorType.FLOATING,
    ordered_categories=['None', 'Low', 'Medium', 'High', 'miss'],
    floating_category='miss'
)
print(table)

# =============================================================================
# 7. SUCCESSIVE MERGES TABLE (Table 6)
# =============================================================================
df, table = get_successive_merges_table(tree, node_id=0)
print(table)

# =============================================================================
# 8. PAIRWISE TABLE AT SPECIFIC STEP
# =============================================================================
# Initial state
print(get_all_pairwise_at_step(tree, node_id=0, predictor='education', step=0))

# After first merge
print(get_all_pairwise_at_step(tree, node_id=0, predictor='education', step=1))

# =============================================================================
# 9. DETAILED MERGE HISTORY
# =============================================================================
print(get_node_merge_history_detailed(tree, node_id=0))

# =============================================================================
# 10. PREDICTIONS
# =============================================================================
test_data = pd.DataFrame({
    'education': ['General', 'None', 'Technical'],
    'study_time': ['>10h', '<2h', '2-5h'],
    'absences': ['Low', 'miss', 'High']
})

predictions = tree.predict(test_data)
probabilities = tree.predict_proba(test_data)

print("\nPredictions:")
for i in range(len(test_data)):
    print(f"  {dict(test_data.iloc[i])} → {predictions[i]}")
    print(f"    Probabilities: {dict(zip(tree.y_categories, probabilities[i]))}")
```

---

## Output Files

When running the examples, the following files are generated:

| File                       | Description                                |
| -------------------------- | ------------------------------------------ |
| `chaid_tree_plot.png`      | CHAID-style tree without distribution bars |
| `chaid_tree_with_bars.png` | CHAID-style tree with distribution bars    |
| `chaid_split_history.png`  | Predictor significance comparison          |
| `chaid_distributions.png`  | Outcome distributions by leaf              |
| `chaid_chi_square.png`     | Chi-square values comparison               |

---

## Algorithm Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MERGE PHASE (Step 1)                         │
│─────────────────────────────────────────────────────────────────────│
│ 1. Start with each category as its own group                        │
│ 2. For all valid pairs:                                             │
│    - Compute χ² for 2×d subtable                                    │
│    - Find pair with MINIMUM χ² (most similar)                       │
│ 3. If p-value > α_merge: MERGE the pair, go to step 2               │
│    Else: STOP merging, proceed to Split Phase                       │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        SPLIT PHASE (Step 2)                         │
│─────────────────────────────────────────────────────────────────────│
│ For each compound category with k ≥ 3 original categories:          │
│ 1. Examine all valid dichotomies (binary splits)                    │
│ 2. Find split with MAXIMUM χ² (most divergent)                      │
│ 3. If p-value < α_split: Implement split, return to MERGE PHASE     │
│    Else: Keep the merged group                                      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                           CONVERGENCE                               │
│─────────────────────────────────────────────────────────────────────│
│ The cycle terminates when no more merges or splits are possible.    │
│ α_split < α_merge ensures convergence (per Kass 1980).              │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Insight**:

- **MERGE**: Looking for SIMILARITY (low χ² = categories behave alike)
- **SPLIT**: Looking for DIVERGENCE (high χ² = categories behave differently)
