"""
Predictor type enumeration for CHAID.

Based on Section 4.1 of main.tex:
- Nominal (free): Any combination of categories can be merged
- Ordinal (monotonic): Only contiguous categories can be merged
- Floating: Ordinal with one floating category (typically missing values)
"""

from enum import Enum


class PredictorType(Enum):
    """
    Enumeration of predictor types in CHAID.
    
    Each type imposes different constraints on category merging:
    
    - NOMINAL: No ordering constraint. Any categories can be merged.
              Number of partitions into g groups: S(c,g) (Stirling number of 2nd kind)
              
    - ORDINAL: Categories have natural order. Only contiguous categories can merge.
              Number of partitions into g groups: C(c-1, g-1)
              
    - FLOATING: Ordinal with one special "floating" category (e.g., missing values)
               that can be merged with any group regardless of position.
               Number of partitions into g groups: C(c-2, g-2) + g * C(c-2, g-1)
    """
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    FLOATING = "floating"
