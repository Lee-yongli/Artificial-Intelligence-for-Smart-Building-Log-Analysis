# Speaker Notes: AI Smart-Building Log Analysis

Target length: under 10 minutes.

Suggested pace: title slide 20-30 seconds; each content slide 45-60 seconds.

## Slide 1 - Title

Good morning. Our project is called AI Smart-Building Log Analysis. In this presentation, I will explain the problem, the dataset, our planned pipeline, the anomaly-detection methods, and how we will evaluate the prototype. The main idea is simple: we want to turn complex building logs into useful evidence for human review.

## Slide 2 - Why AI Is Useful

The motivation is that smart buildings produce many signals. These signals can show energy use, indoor temperature, humidity and lighting conditions. However, normal behaviour is not the same everywhere in a building. A fixed rule may miss important patterns, and manual checking is not realistic. So our goal is decision support. We want to highlight unusual behaviour, but we do not want the system to make automatic operational decisions.

## Slide 3 - Dataset

For the project, we will use the CU-BEMS Energy Research Data from the Chamchuri 5 building. It includes minute-level measurements from seven floors over about eighteen months. The signals include three power-system groups and three indoor-environment groups. This gives us a reproducible dataset. But there is also a limit: it does not represent every Smart Facilities SA system, and it does not include confirmed incident labels.

## Slide 4 - Scope

The project scope is important. We are not building a production monitoring platform, and we are not controlling physical equipment. We are also not installing new sensors or confirming every detected anomaly as a real incident. Instead, the project focuses on data preparation, feature engineering, anomaly analysis, visualisation and prioritisation. This makes the work achievable within the ten-week timeframe.

## Slide 5 - Pipeline

Our pipeline has five main stages. First, we import the CU-BEMS data and keep the raw data unchanged. Second, we clean and integrate the files into a consistent structure. Third, we create features, such as rolling statistics and time-based patterns. Fourth, we run anomaly-detection methods. Finally, we explain the results with visualisations. This structure is important because a reviewer should be able to understand how each result was produced.

## Slide 6 - Methods

We will compare two method families. The first family is interpretable statistical baselines, such as thresholds, z-scores, robust scores and rolling deviations. These are easy to explain and give us a clear benchmark. The second family is unsupervised machine learning, such as Isolation Forest and Local Outlier Factor. These methods may detect more complex patterns, but they must be tuned carefully. We will compare usefulness, stability and interpretability.

## Slide 7 - Evaluation

A key challenge is that the dataset does not have trustworthy labels for true faults. So we cannot rely on normal classification metrics as the main measure. Instead, we will use a multi-part evaluation. We will inject controlled synthetic anomalies, test stability across data subsets, check alert volume, and review representative cases. This is more honest for this project, because an anomaly is not automatically a real equipment failure.

## Slide 8 - Risks

This slide shows the main risks from the proposal. The highest risk is the lack of trustworthy anomaly labels. The next major risk is data quality, including missing or noisy measurements. Integration across floors, zones and measurement types is also important. Our mitigation strategy is to do the data audit early, start with interpretable baselines, compare methods, and clearly document uncertainty.

## Slide 9 - Deliverables

The proposal defines seven deliverables. The first group is build artifacts: the data-preparation pipeline, pattern analysis and model comparison. The second group is interpretation artifacts: visualisation and anomaly prioritisation. The final group is handover artifacts: the final technical report and reproducible package. The important point is that success is not only about producing a model. It is also about producing evidence that another technical reviewer can understand and reproduce.

## Slide 10 - Plan And Close

The project runs over ten teaching weeks. The first two weeks focus on requirements and data confidence. Weeks two to four focus on preparation and pattern discovery. Weeks four to six focus on anomaly detection. Weeks six to eight focus on evaluation and visualisation. The last two weeks are for prototype review, documentation and handover. Our closing message is that the project should provide a useful and reproducible prototype, while clearly explaining its limits.

## Optional Closing Sentence

Thank you. I am happy to answer questions about the dataset, the anomaly-detection methods, or the evaluation plan.

## References Used

- Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. (2000). LOF: Identifying density-based local outliers.
- Chandola, V., Banerjee, A., and Kumar, V. (2009). Anomaly detection: A survey.
- Himeur, Y., Ghanem, K., Alsalemi, A., Bensaali, F., and Amira, A. (2021). Artificial intelligence based anomaly detection of energy consumption in buildings.
- Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008). Isolation Forest.
- Pipattanasomporn, M. et al. (2020). CU-BEMS, smart building electricity consumption and indoor environmental sensor datasets.
