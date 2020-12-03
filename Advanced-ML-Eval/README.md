# Advanced-ML-Evaluation-Techniques

Originally forked from [this repository](https://github.com/dssg/student-early-warning)

In this repository, I am implemeting Sections 5 & 6 from the paper [A Machine Learning FrameWork to Identify Students at Risk of Adverse Academic Outcomes](https://dl.acm.org/doi/10.1145/2783258.2788620)

**Repository Still Under Construction**

**Implementations**:
 - Quantifying 'risk' as being the probability of the positive class
 - Mean Empirical Risk Curves
 - Precision/Recall Curves at Top K
 - Probability of Mistake per model per Frequent Pattern Using FP-growth technique
   * Use percentiles(distribution) to "itemize"/"categorize" values inside column
 - Jaccard Similarity curves between model pairs
 - All models used here are shallow models - Could incorporate Deep Learning Models soon.
 - SMOTE used to oversample **training data** before testing (DO NOT APPLY SMOTE ON THE WHOLE DATA THEN DO A TRAIN-TEST SPLIY)
