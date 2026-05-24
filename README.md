# Goat-Mule Climbing Optimization (GMCO) - Mealpy Extension

An academic and production-ready Python implementation of the **Goat-Mule Climbing Optimization (GMCO)** algorithm, structured as a native extension framework for the popular `mealpy` optimization library.

## 📌 Features
- [cite_start]**Goat Model:** Lévy Flight driven global exploration engine[cite: 245].
- [cite_start]**Mule Model:** Tangent-constrained exploitation engine utilizing N-Dimensional Givens hyperplane rotation to eliminate vector dimension mismatches[cite: 245, 527, 534].
- [cite_start]Fully compliant with the standard `mealpy.Optimizer` pipeline.

## 🚀 Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

   python main_run.py

   Integration Example
   from mealpy_wrapper import GoatMuleClimbingOptimization
from mealpy import Problem

# Deploy GMCO directly onto your native mealpy problem architecture
model = GoatMuleClimbingOptimization(epoch=150, pop_size=50, theta_max=15.0)
best_agent = model.solve(your_problem_struct)
