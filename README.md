# Credit Scoring Business Understanding

This section outlines the business context and considerations for developing a credit risk model for Bati Bank’s buy-now-pay-later service, focusing on regulatory compliance, proxy variables, and model selection trade-offs.

1. **Basel II Accord and Model Interpretability**  
   The Basel II Capital Accord emphasizes accurate risk measurement and regulatory oversight, requiring banks to maintain sufficient capital for credit risk. This necessitates interpretable models, such as Logistic Regression with Weight of Evidence (WoE), which allow clear explanation of feature contributions to risk scores. Well-documented models are critical for regulatory audits, ensuring compliance and enabling transparent loan approval decisions, thus minimizing the risk of penalties.

2. **Necessity and Risks of a Proxy Variable**  
   Since the dataset lacks a direct “default” label, a proxy variable (`is_high_risk`) based on Recency, Frequency, and Monetary (RFM) metrics is essential to classify customers as high or low risk for model training. However, an inaccurate proxy may mislabel customers (e.g., labeling engaged customers as high-risk), leading to biased predictions. This could result in unfair loan denials, increased defaults, or revenue loss, highlighting the need for careful proxy design and validation.

3. **Trade-offs Between Simple and Complex Models**  
   In a regulated financial context, simple models like Logistic Regression with WoE offer high interpretability, making them easier to explain to regulators and align with Basel II requirements. However, they may sacrifice predictive power on complex data. Complex models like Gradient Boosting provide superior accuracy by capturing non-linear patterns but are less interpretable, posing challenges for regulatory approval. Prioritizing interpretability ensures compliance and trust, though balancing performance is key for effective risk assessment.