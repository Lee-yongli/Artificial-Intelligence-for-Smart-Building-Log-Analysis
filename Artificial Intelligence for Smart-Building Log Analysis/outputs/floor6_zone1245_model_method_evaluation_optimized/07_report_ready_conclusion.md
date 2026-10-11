## Report Conclusion and Final Model Choice

For the overall project, the recommended final model is the **V3 operation-aware statistical baseline**. It should be used as the main model for the later analysis, interpretation, event explanation, and report narrative.

This choice is supported by the budget-controlled proxy evaluation. The **V3 budgeted baseline** ranks first at all tested anomaly budgets (**3%, 5%, and 9%**) in both the quantitative proxy ranking and the practical selection ranking. Its mean quantitative rank is **1.0**, and its mean practical rank is **1.0**.

Among the two unsupervised machine-learning methods, **Optimized Isolation Forest** is the better detector. It ranks second at all three budgets, ahead of LOF, with a mean quantitative rank of **2.0** and a mean practical rank of **2.0**. LOF ranks third across the tested budgets, with a mean quantitative rank of **3.0**.

The recommended modelling strategy is therefore:

1. Use **V3 operation-aware statistical baseline** as the project's final/main model.
2. Use **Optimized Isolation Forest** as a supporting unsupervised ML benchmark or secondary candidate detector.
3. Do not choose **LOF** as the main project model, because it is less stable and less useful for event-level interpretation in this comparison.

This conclusion is based on equal anomaly-budget comparisons at **3%, 5%, and 9%**. The V3 full-baseline overlap is reported only as a reference diagnostic and is not used as a scored criterion in the quantitative proxy ranking, which avoids giving the V3 method an unfair advantage against IF and LOF.

Because there is no human-confirmed ground-truth fault label, this is a **proxy model-selection evaluation**, not an accuracy, precision, recall, or F1-score evaluation.
