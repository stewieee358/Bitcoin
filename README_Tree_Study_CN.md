# MH6807：树模型、特征重要性和特征筛选

这是在 https://github.com/stewieee358/Bitcoin 的 `a3c74f3fb4e508165d0a1c293703bb7e00975265` 版本上增加的扩展。
新增决策树、AdaBoost、GBM；复用原始数据、21特征、时间划分、分类指标和缓存接口。
新增入口是 `run_tree_study.py`。原入口 `run_classification.py` 和原训练代码保持原样。

## 你负责的内容

1. 全部21特征下，运行决策树、AdaBoost、GBM。
2. 与组员的 Logistic、KNN、RF、Logistic-PCA、KNN-PCA 同条件比较。
3. 分析RF和三个新树模型在每日历史训练窗口中的树内重要性。
4. 用历史验证段的RF置换重要性，每日筛选一组共用特征。
5. 对照三个新模型的全部特征版本与筛选版本。

这是涨跌分类任务，不直接预测具体BTC价格。树重要性不等于解释了多少比例的价格变动。

## 在 Windows / VS Code 中运行

完整代码包可直接解压到新文件夹，例如 `D:\ML_Project\Bitcoin_Tree_Study`，然后在VS Code打开这个文件夹。
这样能保留你原来的Bitcoin文件夹。

打开「终端 → 新建终端」，确认终端位于解压后包含 `run_tree_study.py` 的目录。
你之前已安装Git和uv，此扩展不需要额外安装XGBoost或LightGBM。

逐条执行：

```powershell
uv python install 3.12
uv venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -r requirements_tree_study.txt
```

如果该文件夹已有兼容的 `.venv`，第二条不用再执行。建议Python 3.12；3.14不适合此项目锁定的依赖。
在VS Code按 `Ctrl+Shift+P`，执行 `Python: Select Interpreter`，选择 `.venv\Scripts\python.exe`。

设置线程数：

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:LOKY_MAX_CPU_COUNT = '4'
```

先运行3天的短检查（它不是正式模型效果结论）：

```powershell
.\.venv\Scripts\python.exe run_tree_study.py --evaluation-end 2025-01-04T00:00:00+00:00
```

再运行2025年全年：

```powershell
.\.venv\Scripts\python.exe run_tree_study.py
```

也提供可选PowerShell脚本，默认只跑短检查：

```powershell
.\Run_Tree_Study.ps1 -Smoke
.\Run_Tree_Study.ps1 -Full
```

如果脚本被PowerShell执行策略阻止，直接使用上面的Python命令即可。
所有模型和筛选参数都在 `configs/config_tree_study_2h_2025.json`。
参数是预先固定的初始设置，未宣称为最优参数，未做超参数搜索。

## 固定的共同设定

| 项目 | 设定 |
| --- | --- |
| 交易对 | BTCUSDT现货 |
| K线 | Binance官方原生2h归档，ZIP/CHECKSUM校验 |
| 历史数据 | 2024-09至2025-12 |
| 评价范围 | UTC `[2025-01-01, 2026-01-01)` |
| 输入 | 原项目全部21个特征 |
| 标签 | 下一根2h收盘收益率>0为1，否则为0 |
| 训练窗口 | 90个日历日的有效样本 |
| 重新拟合 | 每24小时一次 |
| 分类规则 | 上涨概率严格>0.5 |
| 公平比较 | 全部模型、全部/筛选版本使用同一组有效样本 |

每日模型固定，每根K线完成后才获取该根特征并预测下一根，不在零点假设当天特征全部可用。
全年完整评价共365次更新、4379条预测；评价右边界不含，因此年末一条目标落在2026-01-01的预测被排除。

## 三种新增模型的初始参数

| 模型 | 设置 |
| --- | --- |
| 决策树 `dt` | max_depth=4，min_samples_leaf=20，random_state=42 |
| AdaBoost `ada` | 100个浅树，学习率0.1，基础树max_depth=1/min_samples_leaf=10 |
| GBM `gbm` | 100棵树，学习率0.05，max_depth=2，min_samples_leaf=20 |

AdaBoost使用锁定版本scikit-learn的SAMME分类实现。
GBM不启用随机早停验证。预处理仍仅在当前训练数据上拟合。
组员原RF保持其原参数，包含 `max_features=1.0`，没有悄悄修改成另一种RF。

## 每日特征筛选的准确流程

在每日截止时刻T：

1. 使用原框架构造 `[T-90天,T)` 的共同有效训练样本，并要求 `target_time<=T`。
2. 内部截止S=T-30天：前60天拟合筛选RF，明确要求该拟合样本的 `target_time<=S`。
3. 后30天为历史验证段，所有验证标签在T已经可用。
4. 在验证段上重复置换每个特征10次，指标为Balanced Accuracy，阈值同样严格>0.5。
5. 按平均下降值排序，选择Top 5；并列按原特征顺序确定，输入仍保留原顺序。
6. 三个筛选版本共享当天的这组特征，并在全部可用90天样本上重新拟合。
7. 如果样本不足、内部仅单类、或全部置换重要性均不为正，则当天使用全部21特征并记录原因。

因此 `_selected` 表示「历史筛选＋必要时保留全部特征」策略，并不是每一天都强行使用5个特征。
`selection_daily.csv` 记录每天的划分、标签边界、入选名单、验证表现和回退原因。
入选频率仅统计真正Top-5的日期，不将回退日的21特征全部计作重要。
`validation_edge_over_majority` 可用于检查筛选RF是否在该历史验证段超过多数类基准；
排名为正不自动意味着存在稳定可交易的信号。

## 去哪里看结果

终端会打印实际输出路径，形如 `outputs/tree_2h_<编号>/`。
每个实验只有一个 `run.json`，保存配置、版本、数据校验、缓存、状态和筛选身份。

| 文件 | 作用 |
| --- | --- |
| `summary.md` | 英文结果表、筛选效果和限制说明 |
| `RESULTS_Tree_Study_CN.md` | 本包附带的中文实际结果说明 |
| `overall_metrics.csv` | 11种模型/策略版本及zero/majority基准，共13行 |
| `all_feature_metrics.csv` | 8个全部21特征模型及majority，供小组主比较 |
| `feature_selection_effect.csv` | DT、AdaBoost、GBM筛选前后的配对变化 |
| `feature_importance_summary.csv` | 平均RF置换重要性、跨日波动、正值比例、入选频率 |
| `tree_importance_summary.csv` | 各树模型平均训练MDI及波动 |
| `feature_importance_by_month.csv` | 历史验证段的重要性随月份变化 |
| `selection_daily.csv` | 每天共用特征名单及时间审计 |
| `permutation_importance_daily.csv` | 每日每个特征的置换均值和置换标准差 |
| `feature_importance_daily.csv` | 每日每个树模型的训练MDI |
| `initial_window_correlation.csv` | 首个历史训练窗口的Spearman相关性 |
| `predictions.csv` | 统一时间、真实涨跌及所有模型的概率和分类 |
| `training_updates.csv` | 每日90天训练窗口边界和样本数 |
| `roc_curve.png` | 各模型ROC |
| `figures/` | 模型比较、筛选效果、重要性、入选频率、时间变化与相关性图 |

树内MDI图只是补充描述；主要筛选依据是历史验证段的置换重要性。
误差线为跨日标准差，不是独立样本置信区间。每日窗口重叠，重要性观察也相互依赖。
相关OHLC/成交量字段可能替代或分摊重要性；单列逐行置换还会破坏原时间依赖，因此不把图解读成因果关系。

## 检查与复现

单元测试（人工小数据仅用于测试，不进入正式实验）：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

真实实验完成后，将下面路径替换为终端实际打印的目录：

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_tree_study --output-dir outputs/tree_2h_实际编号
```

核验器重算时间划分、全部指标、筛选首日、首日全部模型预测，并校验缓存文件哈希。
现有的 `scripts.verify_classification` 不认识这次新增注册及筛选版本，应使用上述专用核验器。
21项测试包括原项目14项，以及这次新增7项。
另外修复原测试中 `logistic_*` 误匹配 `logistic_pca_*` 的缓存定位问题；没有改原模型实现。

## 缓存及与组员合并

重复运行本扩展时，经过身份、SHA256和共用行检查才复用预测与重要性。
改变Top K或筛选参数会产生新的研究目录；筛选版本不能冒充原模型结果。
增加模型不会把其模型名单当作共同数据身份；每个模型仍有自己的缓存身份。
修改已有模型参数时只重训该模型，但该树的MDI也需更新。
源代码、数据、周期、窗口、依赖版本变化会影响缓存身份。

你的电脑Python版本与本包实验版本不同时，程序可能重新训练，这是正常行为。
只有依赖和配置完全吻合且缓存校验通过，才复用组员旧Logistic/KNN/PCA预测。
RF需要提取训练重要性，缺少对应重要性记录时会重新拟合。
旧15m/1d结果不导入此次2h比较，也不覆盖原实验目录。

完整代码包可以独立运行；若要合并回原仓库，复制下列新增内容：

```text
run_tree_study.py
Run_Tree_Study.ps1
requirements_tree_study.txt
README_Tree_Study_CN.md
configs/config_tree_study_2h_2025.json
src/tree_models.py
src/tree_study.py
src/tree_reports.py
scripts/verify_tree_study.py
scripts/create_tree_results_cn.py
tests/test_tree_study.py
```

`tests/test_framework.py` 只有上述测试定位修复；可按包内版本更新该测试文件。
原 `src/models.py`、`src/training.py`、`src/experiment.py` 等核心文件保持原样。
扩展模型只在新入口中注册，因此从原入口直接传 `--models dt` 不会自动识别。
若组员要加入其他分类器，可先在原模型注册表注册，再把名字和参数加入本扩展JSON的 `models`。
这些新增非树模型参与全部21特征主比较；每日Top-5对照仍固定为你负责的DT/AdaBoost/GBM。

## 报告中该怎么表述

主问题：哪些树模型在共同21特征、90天滚动窗口设定下表现更好？
附加问题：在当时可用的历史数据内筛选特征，能否改善三个树模型的随后样本外预测？
重要性问题：哪些变量经常被选中，哪些排名随时间变化？

所有结果都属于已被小组查看的历史上的探索性比较。
分类指标提升不等于交易盈利；本扩展未运行交易回测，没有证明筛选必然有效，也没有统计显著性结论。
如果再根据2025年成绩调整模型或Top K，需使用新的后续数据验证，不能继续称2025年为独立测试集。

## 方法参考

- scikit-learn，Permutation feature importance：https://scikit-learn.org/stable/modules/permutation_importance.html
- scikit-learn，Permutation importance vs Random Forest MDI：https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
- scikit-learn，Correlated features and permutation importance：https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
- scikit-learn，Ensemble methods：https://scikit-learn.org/stable/modules/ensemble.html

中文结果说明可在完整实验和核验后重新生成：

```powershell
.\.venv\Scripts\python.exe -m scripts.create_tree_results_cn --output-dir outputs/tree_2h_实际编号
```
