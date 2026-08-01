# irr — 賃金・賞与を世界株へ積み立てた場合のIRR研究

日本の賃金・賞与統計と世界株ETF価格から、開始年別の積立キャッシュフローIRRを比較する研究コードです。

## 現在の状態：e-Stat次元を隔離中

過去の実装には、次の重大な問題がありました。

- e-StatのJSON取得に失敗すると、規則的に生成したサンプル賃金・賞与へ自動切替
- 市場データ取得に失敗すると、乱数で作った合成リターンへ自動切替
- 合成値を通常の`wage_bonus.yml`・市場系列として保存
- 一期間が月であるIRRを年率IRRとして表示
- 開始年ごとに投資期間が異なる結果を直接比較
- 同じ市場系列を共有する重複コホートへ独立二標本検定とt信頼区間を適用
- e-Statの複数次元を`@time`だけで辞書化し、同一年の行を上書きする可能性

現在は`config/estat_queries.yaml`を`verified: false`として隔離しています。企業規模1000人以上、産業、男女・年齢等の残存次元、単位、年次集計をe-Statメタ情報で確認するまで、本番分析は実行しません。

GitHub Pagesには過去の数値ではなく、隔離理由を表示します。

## 監査後の計算契約

- API失敗時のサンプル・合成データ代替は禁止
- e-Statクエリ、取得時刻、応答ハッシュを保存
- 1年に複数観測が残った場合は未指定次元として停止
- 市場価格の実際の取得開始日・終了日・観測数を保存
- 全コホートを同じ`horizon_months`で比較
- 月次IRRと複利年率IRRを別フィールドで保存
- 複数符号反転があるキャッシュフローはIRRの一意性を保証できないため拒否
- 重複コホートにはp値・独立標本信頼区間を生成しない
- 推論状態を`descriptive_only_dependent_overlapping_cohorts`として保存

## IRRの定義

月次キャッシュフローに対する一期間IRR `r_m`を解きます。

```text
Σ CF_t / (1 + r_m)^t = 0
```

年率換算:

```text
r_annual = (1 + r_m)^12 - 1
```

月次IRRを12倍する単純換算は行いません。

## 実行

現在はクエリ検証が終わるまで停止します。

```bash
python -m unittest discover -s tests -v
python src/fetch_and_compute_irr.py --horizon-months 60
```

分析を有効化する前に、`config/estat_queries.yaml`の各フィルタをe-Statメタ情報と照合し、未検証プレースホルダーを削除した上で`verified: true`へ変更します。

## 出力

検証済み分析が実行された場合:

```text
data/processed/irr_results.csv
  - monthly_irr
  - annualized_irr
  - horizon_months
  - first_month / last_month
  - total_contributions / terminal_value

data/processed/summary.json
  - 記述統計のみ
data/processed/provenance.json
  - e-Statクエリ・取得日時・応答ハッシュ
  - 市場系列の取得条件・ハッシュ
  - synthetic_fallback_used: false
  - inferential_statistics_generated: false
reports/irr_analysis_report.html
```

## 統計上の境界

開始年別コホートは、多くの月次市場リターンを共有します。そのため通常のWelch t検定、Mann–Whitney U検定、独立標本を前提とするt信頼区間は適用しません。

将来推論を行う場合は、少なくとも次のいずれかを別途設計する必要があります。

- 非重複期間だけを使う比較
- 時系列依存を保持するブロック・ブートストラップ
- 市場経路・産業・開始年の階層構造を表すモデル
- 事前登録した比較と多重検定補正

過去READMEの`p=0.012`、`Cohen's d=1.17`、資産差、完全性100%は再現済み結果として扱いません。

## テスト

- 月次IRRの複利年率換算
- 既知キャッシュフローからのIRR復元
- 複数符号反転の拒否
- 未検証e-Stat設定の停止
- 固定比較期間の保存
- 不完全期間の拒否
- 記述集計にp値を含めないこと

## 注意

- ACWI ETF、指数、円換算、配当再投資は同じものではありません
- 現コードの市場入力は`yfinance`取得系列で、公式指数データではありません
- 税、為替、手数料、失業、積立停止は未反映です
- 本プロジェクトは投資助言や将来資産保証ではありません

**README最終監査:** 2026-08-02
