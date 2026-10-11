## Final Modelling Strategy

The final model for the project is the **V3 operation-aware statistical baseline**. It is used to define the final anomaly labels, final anomaly events, event severity, variable-level explanations, and subsequent analysis outputs.

The unsupervised machine-learning models are retained as supporting evidence rather than replacing the final model. **Optimized Isolation Forest** is the primary ML support model, because it was the strongest unsupervised ML method in the budget-controlled method evaluation. **LOF** is kept as an additional benchmark.

In the final V3 event table, there are **475** V3 anomaly events. **115** events have meaningful Isolation Forest support, **121** events have meaningful LOF support, and **20** events have support from both IF and LOF. The remaining **259** events are still final anomalies because the final model is V3, but they should be interpreted as V3-primary events without strong ML corroboration.

This combined design satisfies the project aim of using AI for smart-building log analysis while keeping the final outputs interpretable. V3 provides the operational explanation; IF and LOF provide AI/ML-based cross-model evidence and benchmarking.
