import csv
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import requests
import yfinance as yf
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ESTAT_APP_ID = os.getenv('ESTAT_APP_ID', '')
INDUSTRIES = {
    '産業計': '',
    'J 金融業，保険業': 'J',
    'C 鉱業，採石業，砂利採取業': 'C',
    'G 情報通信業': 'G',
    'E 製造業': 'E',
}
WAGE_STATS_ID = '0003411955'
BONUS_STATS_ID = '0003411978'


def fetch_estat(stats_id: str, industry: str) -> Dict[str, float]:
    params = {
        'appId': ESTAT_APP_ID,
        'statsDataId': stats_id,
        'metaGetFlg': 'N',
        'cntGetFlg': 'N',
        'explanationGetFlg': 'N',
        'annotationGetFlg': 'N',
    }
    if industry:
        params['cdCat01'] = industry
    url = 'https://api.e-stat.go.jp/rest/3.0/app/getSimpleStatsData'
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    values = data['GET_STATS_DATA']['STATISTICAL_DATA']['DATA_INF']['VALUE']
    return {v['@time']: float(v['$']) for v in values}


def fetch_wage_bonus() -> Dict[str, Dict[str, Dict[str, float]]]:
    out = {}
    for name, code in INDUSTRIES.items():
        wage = fetch_estat(WAGE_STATS_ID, code)
        bonus = fetch_estat(BONUS_STATS_ID, code)
        out[name] = {'wage': wage, 'bonus': bonus}
    with open(RAW_DIR / 'wage_bonus.yml', 'w', encoding='utf-8') as f:
        yaml.dump({'industries': out}, f, allow_unicode=True)
    return out


def fetch_world_returns() -> pd.Series:
    data = yf.download('ACWI', start='2004-01-01', end='2016-01-01', interval='1mo', auto_adjust=True, progress=False)
    data['return'] = data['Close'].pct_change()
    data = data.dropna(subset=['return'])
    data[['return']].to_csv(RAW_DIR / 'world_stock.csv', index_label='date')
    return data['return']


def irr(cashflows):
    low, high = -0.9999, 1.0
    for _ in range(1000):
        mid = (low + high) / 2
        npv = sum(cf / ((1 + mid) ** i) for i, cf in enumerate(cashflows))
        if abs(npv) < 1e-6:
            return mid
        if npv > 0:
            low = mid
        else:
            high = mid
    return mid


def main():
    wage_bonus = fetch_wage_bonus()
    returns = fetch_world_returns()
    dates = returns.index.strftime('%Y-%m').tolist()
    r_list = returns.tolist()

    results = []
    for industry, data in wage_bonus.items():
        years = sorted(data['wage'].keys())
        for start_year in map(int, years):
            portfolio = 0.0
            cashflows = []
            for date, r in zip(dates, r_list):
                year = int(date[:4])
                if year < start_year:
                    continue
                wage = data['wage'].get(str(year))
                bonus = data['bonus'].get(str(year))
                if wage is None or bonus is None:
                    break
                contribution = wage + bonus / 12.0
                portfolio += contribution
                cashflows.append(-contribution)
                portfolio *= (1 + r)
            cashflows.append(portfolio)
            results.append({'industry': industry, 'start_year': start_year, 'irr': irr(cashflows)})

    with open(PROCESSED_DIR / 'irr_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['industry', 'start_year', 'irr'])
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()
