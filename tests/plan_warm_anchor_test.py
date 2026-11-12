#!/usr/bin/env python3
"""Ensure model order is swapped while both setups have a warm anchor."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from benchmark_plan_selection import reversed_plan_order

items=['atbsr','other_baseline','auto_model','auto_vector_model']
assert reversed_plan_order(items,False)==['auto_vector_model','auto_model','other_baseline','atbsr']
anchored=reversed_plan_order(items,True)
assert anchored==['atbsr','auto_vector_model','auto_model','other_baseline']
assert items[0]==anchored[0] and items.index('auto_model')<items.index('auto_vector_model')
assert anchored.index('auto_vector_model')<anchored.index('auto_model')
print('warm setup anchor ordering passed')
