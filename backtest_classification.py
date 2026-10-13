"""Idealized daily close-to-close long/cash backtest of saved predictions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import load_data
from src.prepare_data import INTERVALS
from src.metadata import read_run, update_run, atomic_json

ROOT = Path(__file__).resolve().parent


def scheduled_positions(times, signals, hours):
    times = pd.DatetimeIndex(times)
    assert times.tz is not None
    times = times.tz_convert('UTC')
    decision = times == times.floor(f'{hours}h')
    # Use only the current completed bar's signal at each UTC decision boundary.
    # Before the first boundary stay in cash; never average future signals.
    positions = pd.Series(np.asarray(signals,dtype=float)).where(decision).ffill().fillna(0).to_numpy()
    changes = positions != np.r_[0.,positions[:-1]]
    assert not (changes & ~decision).any()
    return positions


def simulate(returns, positions, fee, periods_per_day=1):
    positions = np.asarray(positions, dtype=float)
    returns = np.asarray(returns, dtype=float)
    assert len(positions) == len(returns) and len(returns)
    assert np.isin(positions, [0.,1.]).all() and np.isfinite(returns).all()
    turnover = np.abs(positions - np.r_[0., positions[:-1]])
    gross_factor = 1 + positions * returns
    net_factor = (1-fee*turnover) * gross_factor
    # Initial cash; liquidate any final BTC position at the end boundary.
    net_factor[-1] *= 1-fee*positions[-1]
    gross = np.cumprod(gross_factor)
    net = np.cumprod(net_factor)
    peak = np.maximum.accumulate(np.r_[1.,net])[1:]
    return dict(gross=gross, net=net, turnover=turnover,
                gross_return=gross[-1]-1, net_return=net[-1]-1,
                max_drawdown=float((net/peak-1).min()),
                annualized_return=net[-1]**(365*periods_per_day/len(returns))-1,
                exposure=float(positions.mean()),
                buy_count=int(np.sum(positions>np.r_[0.,positions[:-1]])),
                sell_count=int(np.sum(positions<np.r_[0.,positions[:-1]])+positions[-1]),
                fee_events=int(turnover.sum()+positions[-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fee', type=float, default=.001, help='hypothetical one-way fee fraction')
    parser.add_argument('--predictions-dir', type=Path, default=ROOT/'outputs/1d_b9ce570233e0cefd')
    parser.add_argument('--output-dir', type=Path, help='separate result directory for fee scenarios')
    parser.add_argument('--rebalance-hours', type=int, help='hold position between UTC decision boundaries')
    args = parser.parse_args()
    if not 0<=args.fee<1:
        raise ValueError('fee must be 0..1 exclusive')
    predictions_dir = args.predictions_dir.resolve()
    document = read_run(predictions_dir)
    config = document['config']
    bar_step = INTERVALS[config['interval']]
    periods_per_day = int(pd.Timedelta(days=1)/bar_step)
    if args.rebalance_hours is not None:
        if args.rebalance_hours <= 0 or pd.Timedelta(hours=args.rebalance_hours) < bar_step or pd.Timedelta(hours=args.rebalance_hours) % bar_step:
            raise ValueError('Decision interval must be positive and a multiple of the bar interval')
    out = args.output_dir.resolve() if args.output_dir else predictions_dir/'backtest'
    out.mkdir(parents=True, exist_ok=True)
    assert document['status']['status']=='success'
    pred = pd.read_csv(predictions_dir/'predictions.csv', float_precision='round_trip')
    for c in ('feature_time','target_time','train_cutoff'):
        pred[c] = pd.to_datetime(pred[c], utc=True)
    pred = pred.sort_values('feature_time').reset_index(drop=True)
    assert not pred.feature_time.duplicated().any()
    assert (pred.target_time==pred.feature_time+bar_step).all()
    assert (pred.feature_time.diff().dropna()==bar_step).all()
    assert (pred.train_cutoff<=pred.feature_time).all()
    raw,_,_ = load_data(config, ROOT/'data/raw')
    joined = pred.merge(raw[['feature_time','close','target']], on='feature_time', validate='one_to_one')
    assert len(joined)==len(pred)
    np.testing.assert_allclose(joined.actual_return, joined.target, rtol=0, atol=1e-14)
    # Independent hand check of timing, compounding, initial/final fees and cash.
    check = simulate([.1,-.1,.1], [1,0,1], args.fee)
    np.testing.assert_allclose(check['net'][-1], 1.21*(1-args.fee)**4)
    assert check['buy_count']==check['sell_count']==2
    assert simulate([.1,-.1],[0,0],args.fee)['net_return']==0
    if args.rebalance_hours == 2:
        times_check = pd.date_range('2025-04-01',periods=17,freq='15min',tz='UTC')
        signals_check = [1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1]
        np.testing.assert_array_equal(scheduled_positions(times_check,signals_check,2),[1]*8+[0]*8+[1])
        np.testing.assert_array_equal(scheduled_positions(times_check[1:8],signals_check[1:8],2),np.zeros(7))
    equity = pred[['feature_time','target_time','actual_return']].copy()
    results = []
    # Tree-study metadata also records the selected variants.
    names = list(document.get('active_model_cache') or config['models'])+['buy_hold','cash']
    missing = [name for name in names if name not in ('buy_hold', 'cash') and name+'_class' not in pred]
    if missing:
        raise ValueError(f'Missing prediction columns: {missing}')
    for name in names:
        position = (np.ones(len(pred)) if name=='buy_hold' else np.zeros(len(pred)) if name=='cash' else pred[name+'_class'].to_numpy())
        if args.rebalance_hours and name not in ('buy_hold','cash'):
            position = scheduled_positions(pred.feature_time,position,args.rebalance_hours)
        result = simulate(pred.actual_return,position,args.fee,periods_per_day)
        results.append(dict(strategy=name,n=len(pred),fee_one_way=args.fee,
                            **{k:v for k,v in result.items() if k not in ('gross','net','turnover')}))
        equity[name+'_position'] = position
        equity[name+'_gross_equity'] = result['gross']
        equity[name+'_net_equity'] = result['net']
    # BTC benchmark must exactly match cumulative close-to-close price changes.
    btc_price = np.r_[joined.close.iloc[0],joined.close.to_numpy()*(1+pred.actual_return.to_numpy())]
    np.testing.assert_allclose(equity.buy_hold_gross_equity,btc_price[1:]/btc_price[0],rtol=1e-12)
    equity.to_csv(out/'equity_curve.csv',index=False)
    metrics = pd.DataFrame(results)
    metrics.to_csv(out/'metrics.csv',index=False)
    settings = dict(initial_equity=1.,fee_one_way=args.fee,execution='idealized feature-time close reference',
                    position='long BTC for class1, otherwise cash',shorting=False,leverage=1,
                    cash_interest=0,slippage=0,final_liquidation=True,start=pred.feature_time.iloc[0].isoformat(),
                    end=pred.target_time.iloc[-1].isoformat(),n=len(pred),interval=config['interval'],periods_per_day=periods_per_day,
                    rebalance_hours=args.rebalance_hours,model_horizon_minutes=int(bar_step/pd.Timedelta(minutes=1)),strategies=names)
    if out.is_relative_to(predictions_dir) and out != predictions_dir:
        backtests = read_run(predictions_dir).get('backtests', {})
        backtests[out.relative_to(predictions_dir).as_posix()] = {'config': settings}
        update_run(predictions_dir, backtests=backtests)
    else:
        atomic_json(out/'run.json', {'config': settings})
    fig,ax = plt.subplots(figsize=(10,5),constrained_layout=True)
    times = pd.DatetimeIndex([pred.feature_time.iloc[0]]+pred.target_time.tolist())
    for name in names:
        ax.plot(times,np.r_[1.,equity[name+'_net_equity']],label=name)
    timing = f'{args.rebalance_hours}h decisions' if args.rebalance_hours else f'{config["interval"]} decisions'
    ax.set(xlabel='Time (UTC)',ylabel='Net equity (initial = 1)',title=f'{timing}; fee {args.fee:.2%} per side')
    ax.legend(); ax.grid(alpha=.3)
    fig.savefig(out/'net_equity.png',dpi=160)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(10,5),constrained_layout=True)
    for name in names:
        ax.plot(times,np.r_[1.,equity[name+'_gross_equity']],label=name)
    ax.set(xlabel='Time (UTC)',ylabel='Gross equity (initial = 1)',title=f'{timing}: without fees')
    ax.legend(); ax.grid(alpha=.3)
    fig.savefig(out/'gross_equity.png',dpi=160)
    plt.close(fig)
    text = ['# 简易分类回测', '', f'范围：{settings["start"]} 至 {settings["end"]}，共{len(pred)}个连续{config["interval"]}收益区间。',
            f'复用{len(names)-2}个模型/预测版本；训练窗口{config["window_days"]}天，共用相同评价行。候选特征{len(config["feature_columns"])}个，筛选版本使用当日历史筛选的特征；没有根据回测修改参数。',
            '预测上涨则100%持有BTC，非上涨则100%现金；现货、不做空、无杠杆，现金利息为0。',
            (f'仅在UTC每{args.rebalance_hours}小时边界读取当时信号，期间仓位保持不变，期末清仓例外。模型预测期限仍为下一根{config["interval"]}。' if args.rebalance_hours else '每根K线结束后读取该时刻信号，只有仓位变化才交易。'),
            '零收益在交易比较中对应现金；BTC自身价格表现对应buy_hold（买入持有）。',
            f'手续费是假设的单边{args.fee:.2%}，不是账户实际费率；初次买入、每次换仓和末尾清仓均扣费。另给不扣费收益。',
            '费用净值递推：(上一净值)*(1-费率*换仓量)*(1+仓位*该根真实收益率)，最后一根另扣末尾清仓费。', '',
            '| 策略 | 不扣费总收益 | 扣费总收益 | 扣费最大回撤 | 持仓时间比例 | 买入/卖出次数 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in metrics.itertuples():
        text.append(f'| {r.strategy} | {r.gross_return:.2%} | {r.net_return:.2%} | {r.max_drawdown:.2%} | {r.exposure:.2%} | {r.buy_count}/{r.sell_count} |')
    text += ['', '表中收益率/回撤展示为百分比，CSV存储为小数。净值初始为1。年化收益按365日及每根实际时长换算，仅为这段路径换算。',
             '执行假设：K线完整结束后得到特征，理想化地按该结束边界的收盘参考价即时换仓。实际计算/下单有延迟，无法保证成交于这个已知收盘价。',
             '未建模滑点、买卖价差、市场冲击和真实成交；这是信号的理想化历史模拟，不是可执行盈利证明。',
             '全部模型和BTC基准使用相同连续日期；已核对真实标签、基准价格归一化、手续费方向与复利手工样例。',
             '这段历史已被多次查看和尝试，不是新的独立样本；分类准确率高不保证收益高，收益受涨跌幅及持仓时段影响。', '']
    (out/'summary.md').write_text('\n'.join(text),encoding='utf-8')
    print(metrics.to_string(index=False))
    print(f'Completed backtest: {out}', flush=True)


if __name__=='__main__':
    main()
