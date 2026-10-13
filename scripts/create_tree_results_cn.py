"""Create a Chinese result note from a successfully completed tree study."""
import argparse
from pathlib import Path

import pandas as pd
from src.metadata import read_run


def create_report(out):
    document = read_run(out)
    if document['status']['status'] != 'success':
        raise ValueError('Finish the study before writing its result note')
    config, status = document['config'], document['status']
    metrics = pd.read_csv(out/'overall_metrics.csv')
    effect = pd.read_csv(out/'feature_selection_effect.csv')
    importance = pd.read_csv(out/'feature_importance_summary.csv')
    selection = pd.read_csv(out/'selection_daily.csv')
    names = {'dt': '决策树', 'ada': 'AdaBoost', 'gbm': '梯度提升树GBM',
             'dt_selected': '决策树：历史筛选', 'ada_selected': 'AdaBoost：历史筛选',
             'gbm_selected': 'GBM：历史筛选', 'majority': '训练窗口多数类基准', 'zero': '始终非上涨基准'}
    lines = ['# MH6807 BTC树模型项目：实际运行结果', '',
             f'评价区间：UTC [{config["evaluation_start"]}, {config["evaluation_end"]})。',
             f'共用评价样本：{status["n"]:,}条；每日更新：{status["n_updates"]}次。',
             '三个新增模型为决策树、AdaBoost和GBM；同时重现组员的Logistic、KNN、RF及两个PCA变体。',
             '全部特征的主比较均使用相同21个特征；每日历史筛选是另外三组配对对照。', '',
             '## 方法', '',
             '训练使用此前90个日历日，24小时重新拟合，预测下一根2h K线涨跌；上涨概率严格大于0.5时预测上涨。',
             f'每日先在90天历史窗口的前{90-config["tree_study"]["validation_days"]}天拟合筛选RF，'
             f'并在后{config["tree_study"]["validation_days"]}天计算重复{config["tree_study"]["permutation_repeats"]}次的置换重要性。',
             '筛选RF的训练标签在内部截止时刻已知；验证标签在当日更新时刻已知。随后用完整90天历史重训三个模型。',
             f'实际Top-{config["tree_study"]["top_k"]}筛选{status["selected_days"]}天，保留全部特征{status["fallback_days"]}天。',
             '保留全部特征的原因由selection_daily.csv逐日记录；不把回退日的21特征都计作重要。', '',
             '## 分类结果', '',
             '| 模型 | Accuracy | Balanced Accuracy | F1 | AUC | Log Loss | Brier |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in metrics.itertuples():
        lines.append(f'| {names.get(row.model,row.model)} | {row.accuracy:.2%} | {row.balanced_accuracy:.2%} | '
                     f'{row.f1:.2%} | {row.roc_auc:.4f} | {row.log_loss:.4f} | {row.brier_score:.4f} |')
    lines += ['', '## 筛选前后的配对变化', '',
              '以下变化为筛选版本减去全部21特征版本。Balanced Accuracy与AUC越高越好，Log Loss越低越好。', '',
              '| 模型 | Balanced Accuracy变化（百分点） | AUC变化 | Log Loss变化 |',
              '| --- | ---: | ---: | ---: |']
    for row in effect.itertuples():
        lines.append(f'| {names[row.model]} | {100*row.balanced_accuracy_change:+.3f} | '
                     f'{row.roc_auc_change:+.4f} | {row.log_loss_change:+.4f} |')
    lines += ['', '## 特征重要性', '',
              '下表按历史验证段的平均RF置换重要性排序，列出前5个变量。下降值的单位是Balanced Accuracy百分点。',
              '跨日标准差反映描述性波动，不能当作独立观测的置信区间；每个每日窗口相互重叠。', '',
              f'| 特征 | 平均下降（百分点） | 跨日标准差（百分点） | Top-{config["tree_study"]["top_k"]}入选频率 |',
              '| --- | ---: | ---: | ---: |']
    for row in importance.head(5).itertuples():
        lines.append(f'| {row.feature} | {100*row.mean_decrease:.3f} | {100*row.std_across_days:.3f} | {row.top_k_frequency:.2%} |')
    lines += ['', '入选频率只统计真正按Top K筛选的日期。平均重要性较高与入选频率较高是两种不同的统计量。',
              '这些数值描述特定模型对变量的预测依赖，不是BTC价格变动的解释比例，也不是因果贡献。',
              'OHLC与成交量字段存在相关性，可替代或分摊重要性；逐行置换也会打破原有时间依赖。', '',
              '## 可用于报告的结论', '',
              '模型比较与特征筛选效果应分别讨论，不能仅凭一张重要性图认定筛选有效。',
              '不同树模型对同一组筛选特征的反应可能不同，应结合上面的配对变化说明。',
              '主要对照训练窗口多数类概率基准；始终非上涨的零概率基准有极端Log Loss，超过它不代表概率预测有用。',
              '这些结果来自小组已查看过的历史，属于探索性结果。小幅分数差异未做统计显著性检验。',
              '本项目完成的是涨跌分类，没有运行交易回测，不将分类表现解释为交易盈利。', '',
              '## 核验与文件', '',
              '验证记录在代码包validation目录：21项单元测试，以及独立时间边界、指标、缓存SHA256和首日重新拟合核验。',
              '完整每日记录与预测在本实验目录；结果图在figures目录，ROC图在roc_curve.png。',
              '本次开发和实测使用Linux及Python 3.12.14。提供的PowerShell启动脚本未在真实Windows上执行。', '']
    path = out/'RESULTS_Tree_Study_CN.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    create_report(args.output_dir.resolve())
