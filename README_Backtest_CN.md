# BTC分类模型回测补充包

`run_tree_study.py` 只运行训练、分类评价和特征分析，不自动回测。

## 回测入口

`backtest_classification.py` 识别全部模型与3个筛选版本，读取已有实验预测后生成回测结果。

示例假设：预测上涨则全仓持有BTC，否则持现金；无杠杆、不做空；每2h检查信号，仅仓位变化时交易；单边手续费0.1%，初次买入与末尾清仓也扣费；未建模滑点、价差和延迟。这是历史理想化模拟，不是未来预期收益。

## 在自己的项目运行

1. 确认 `backtest_classification.py` 位于Bitcoin项目根目录，并使用支持筛选版本的新版入口。已有预测与训练结果不用替换。
2. 在VS Code的项目终端执行下面命令；将 `实际实验编号` 换成你的真实输出目录名：

```powershell
.\.venv\Scripts\python.exe backtest_classification.py --predictions-dir ".\outputs\tree_2h_实际实验编号" --fee 0.001
```

`--predictions-dir` 指向包含 `predictions.csv` 和 `run.json` 的实验目录。
如果你本地运行的是Top 10，该命令就回测你本地Top-10版本；不用重新训练。
0.001是本示例假设的单边0.1%手续费，不是查询得到的账户实际费率。

运行后终端打印 `Completed backtest: 完整路径`，结果位于该实验目录下面的 `backtest` 文件夹。

## 看哪个文件

| 文件 | 用途 |
| --- | --- |
| `summary.md` | 中文策略说明和已格式化为百分比的收益表，最容易阅读 |
| `metrics.csv` | 各策略的累计毛收益、累计净收益、年化净收益、回撤和交易次数 |
| `net_equity.png` | 扣费净值曲线 |
| `gross_equity.png` | 不扣费净值曲线 |
| `equity_curve.csv` | 每个2h时段的仓位和净值 |

`gross_return` 是不扣手续费的累计收益，`net_return` 是扣手续费后的累计收益，`annualized_return` 是历史净收益的年化换算。
CSV收益率存小数：0.15=15%，-0.40=-40%。`summary.md` 已显示为百分比。
`buy_hold` 是同区间买入持有BTC，`cash` 是一直持现金。

## 原入口为什么要更新

原脚本只读取 `run.json.config.models` 中的模型，遗漏了DT、AdaBoost、GBM的 `_selected` 预测版本。
更新入口读取 `active_model_cache` 中的全部版本；对原分类实验仍兼容。回测报告也明确区分全部特征与每日筛选版本。

## 已完成的验证

此前验证使用的2025年Top-5历史示例包含11个模型/预测版本及BTC买入持有、现金，共13条策略。
已核对真实行情标签、收益区间连续性、BTC基准、首尾费用和手工复利样例。
另独立使用对数复利加总和交易事件数，重算全部13个策略的毛/净收益，误差不超过1e-12。

上述验证示例对应原来的Top 5全年预测，不代表你后来本地Top 10的收益。
高频换仓造成大量手续费，因此示例中不少模型毛收益为正但净收益为负。应同时报告毛收益、净收益和交易次数。
