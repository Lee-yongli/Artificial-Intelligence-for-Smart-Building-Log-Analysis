# 2018-09-12 新日期验证

目标时间：2018-09-12 07:58:00 至 2018-09-12 08:25:00；提前 15 分钟滚动预测。
有效目标：112/112；四个区域各 28 分钟。

- persistence: MAE=6.1905 kW，RMSE=11.0205 kW，Bias=3.6720 kW。
- hgb: MAE=3.4935 kW，RMSE=5.1922 kW，Bias=0.9786 kW。
- HGB 相对 persistence 的 MAE 改善率：43.57%（负数表示退步）。
- Zone 1：persistence 的 MAE 更低。
- Zone 2：persistence 的 MAE 更低。
- Zone 4：hgb 的 MAE 更低。
- Zone 5：hgb 的 MAE 更低。

只使用各预测起点及以前的信息；未重新训练；真实值为对应区域全部 AC 通道之和。
短窗口结果不能推广为长期性能结论，特别是突变和启停行为需要更多日期检验。