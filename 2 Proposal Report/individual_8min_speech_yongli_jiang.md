# Individual 8-Minute Speech Script - Yongli Jiang (Lee)

Target speaker: Yongli Jiang (Lee)
Role: Data Engineer
Target duration: about 5 to 7 minutes with pauses
Style: clear academic English, easy for a non-native speaker

## Slide 1 - Title

Good morning. My name is Yongli Jiang, and my preferred name is Lee.

Today I present AI Smart-Building Log Analysis.

The goal is to use AI to find useful patterns and possible abnormal behaviour in smart-building data.

My role is Data Engineer.

## Slide 2 - Why AI Is Useful Here

First, I will explain why this problem matters.

Smart buildings produce large data from energy, temperature, humidity, and lighting systems.

Manual checking is not practical.

Normal behaviour also changes by time, floor, zone, and device type.

Our prototype supports human decision-making.

## Slide 3 - Dataset

Next, I will introduce the dataset.

We use the public CU-BEMS dataset from Chamchuri 5.

It has minute-level data from seven floors, from July 2018 to December 2019.

It has no confirmed fault labels, so anomalies are review candidates, not confirmed failures.

## Slide 4 - Scope

Now I will define the project scope.

This is a research prototype.

In scope are data cleaning, integration, feature engineering, pattern analysis, anomaly detection, visualisation, and prioritisation.

Out of scope are automatic control, new hardware, cybersecurity testing, and confirming every anomaly.

## Slide 5 - Reproducible Pipeline

This slide is the most important for my own role as Data Engineer.

First, I will analyse the structure and quality of smart-building operational and sensor data: timestamps, missing values, inconsistent records, variable names, floors, zones, and measurement types.

Second, I will help design an analysis-ready data structure for measurements across floors, zones, and systems.

Third, I will support cleaning, transformation, and timestamp processing.

I will also create features for pattern analysis and anomaly detection, such as time variables, rolling statistics, rate of change, and energy-environment relationships.

This matters because anomaly detection depends on reliable input data. If the data foundation is weak, the model output may be misleading.

## Slide 6 - Methods

After the data pipeline, we compare methods.

We compare two method types: statistical baselines and unsupervised machine learning.

Baselines are easier to explain.

Machine learning may find more complex patterns.

We compare stability, alert volume, explanation, and usefulness.

## Slide 7 - Evaluation

After that, we need to evaluate the results.

Evaluation is difficult because there are no trusted fault labels.

So we cannot only use accuracy or F1-score.

We will use synthetic anomalies, stability checks, alert volume, and example cases.

This shows what the prototype can and cannot prove.

## Slide 8 - Risks

Now I will discuss the main risks.

The main risks are data quality, integration, and evaluation.

The biggest risk is the lack of trusted anomaly labels.

My role reduces data risks through audit, timestamp standardisation, missing-data checks, variable mapping, and documentation.

The team also controls false positives and scope.

## Slide 9 - Deliverables

Next, I will summarise the deliverables.

The proposal has seven deliverables.

The main deliverables are a data pipeline, pattern analysis, model comparison, visualisations, anomaly prioritisation, final report, and handover package.

My main contribution is the data pipeline and pattern analysis.

## Slide 10 - Plan And Close

Finally, I will close with the plan and my learning goal.

The plan starts with requirements and data audit.

Then we prepare data, create features, test methods, evaluate results, and prepare handover.

My learning goal is better data-engineering for building operations.

To conclude, reliable data is the foundation of this project. Thank you.

I am happy to answer questions.

## Short Q&A Preparation

If asked "What is your role?", answer:

My role is Data Engineer. I analyse data structure and quality, integrate measurements across floors, zones, and systems, clean timestamps and missing values, and create features for pattern analysis and anomaly detection.

If asked "Why is your role important?", answer:

Anomaly detection depends on reliable data. If the data is not clean and well structured, even a strong model can produce misleading results.

If asked "Your dataset mainly contains numerical sensor time-series rather than event or system logs. How does it represent the log-analysis problem in the original brief?", answer:

The alignment is not one-to-one. This is not a traditional text-based log dataset. However, it contains time-stamped operational records from building systems, so we use it as a measurable representation of smart-building operations. It supports our core aim: finding normal patterns and unusual behaviour for human review. We also recognise this limitation, and future work should validate the prototype with real Smart Facilities SA logs, maintenance records, or event records before production use.

Chinese reminder: 重点是先承认它不是传统 text-based event logs，然后解释它仍然是 time-stamped operational records，可以代表建筑系统的可测量运行状态，最后说明未来需要真实日志或维护记录验证。
