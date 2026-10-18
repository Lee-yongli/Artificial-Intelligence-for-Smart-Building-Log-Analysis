# Gavin forecasting results

选定模型：hgb_3（训练集拟合，验证集 MAE 选参）。

目标：同一区域 15 分钟后的 AC 功率。

- persistence: Test MAE = 1.6958 kW; RMSE = 5.7687 kW; Bias = -0.0000 kW.
- hgb: Test MAE = 0.6259 kW; RMSE = 2.0983 kW; Bias = 0.0614 kW.
- ML 相对 baseline 的 MAE 改善率：63.10%；RMSE 改善率：63.63%（负数表示退步）。
- 按 zone 分组，ML 最低 MAE：2（0.4295 kW，n=12945）；最高 MAE：4（1.1174 kW，n=12945）。
- 按 period 分组，ML 最低 MAE：night_off（0.0163 kW，n=10020）；最高 MAE：core_work（2.1436 kW，n=11520）。
ML 的 MAE 高于 persistence 的时段：evening_off, night_off, ramp_down, weekend。整体改善不代表所有场景都更好。
完整原始 AC 通道测试子集（n=51691）：persistence MAE=1.6932 kW，RMSE=5.7631 kW。
完整原始 AC 通道测试子集（n=51691）：hgb MAE=0.6254 kW，RMSE=2.0957 kW。

时段按目标时间划分，属于固定日历分组，不代表实测设备启停状态。
真实值对比图排除部分 AC 通道缺失目标，图中三条序列使用同一子集。小时均值仅用于趋势图，指标使用原始逐分钟记录。总体指标为各区域分钟记录等权。
结果仅代表当前楼层与留出时段；缺失目标已由上游排除，不含节假日识别或统计显著性检验。

## 产物
- data/: validation/test 逐行真实值、预测值、误差与时间分组。
- reports/: 总体指标、验证选参、分组指标和运行元数据。
- figures/: 趋势图、48 小时细节图、误差分布与分组比较图。
- models/: 包含独热编码器的 HGB pipeline。
原始 AC 核对：全部 361,433 个目标与对应区域 AC 通道和一致；其中 1,182 个目标存在部分通道缺失。