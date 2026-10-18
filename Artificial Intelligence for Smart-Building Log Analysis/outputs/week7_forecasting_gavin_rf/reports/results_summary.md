# Gavin forecasting results

选定模型：rf_2（训练集拟合，验证集 MAE 选参）。

目标：同一区域未来 15 分钟平均 AC 功率。
本版输出表格、CSV/JSON 报告和模型文件；如存在 2018-09-06 actual table，会额外输出该日 actual vs predicted。

- persistence: Test MAE = 1.1009 kW; RMSE = 3.8355 kW; Bias = 0.0000 kW.
- rf: Test MAE = 0.3122 kW; RMSE = 1.0807 kW; Bias = 0.1129 kW.
- ML 相对 baseline 的 MAE 改善率：71.64%；RMSE 改善率：71.82%（负数表示退步）。
- 2018-09-06 persistence: MAE = 1.5832 kW; RMSE = 4.6333 kW; Bias = -0.0504 kW.
- 2018-09-06 rf: MAE = 0.4771 kW; RMSE = 1.3329 kW; Bias = 0.0784 kW.
- 按 zone 分组，ML 最低 MAE：5（0.2085 kW，n=12945）；最高 MAE：4（0.4839 kW，n=12945）。
- 按 period 分组，ML 最低 MAE：night_off（0.0001 kW，n=10020）；最高 MAE：ramp_up（1.0744 kW，n=2880）。
ML 的 MAE 高于 persistence 的时段：evening_off, night_off, weekend。整体改善不代表所有场景都更好。

时段按目标窗口截止时间划分，属于固定日历分组，不代表实测设备启停状态。
总体指标为各区域分钟记录等权；逐行预测对比保存在 data/，摘要对比预览保存在 reports/。
结果仅代表当前楼层与留出时段；缺失目标已由上游排除，不含节假日识别或统计显著性检验。

## 产物
- data/: validation/test 逐行真实值、预测值、误差与时间分组；如存在 sep6 actual table，也包含该日预测。
- reports/: 总体指标、验证选参、预测对比预览、分组指标和运行元数据。
- models/: 包含独热编码器的 RF pipeline。
15 分钟平均 target 核对：全部 361,377 个 train/validation/test 目标已由 cleaned source 构建。