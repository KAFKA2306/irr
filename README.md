# 賃金データ投資における内部収益率（IRR）計算

本リポジトリでは、日本の賃金および賞与収入をグローバル株式指数に投資した場合の
内部収益率（IRR）を計算します。データは公開APIから取得し、実行ごとに処理を行います。

## 構成
- `data/raw/`: ダウンロードした賃金/賞与データおよび株式リターンデータ（git管理対象外）
- `data/processed/`: 計算結果であるIRR値を格納するディレクトリ
- `src/fetch_and_compute_irr.py`: データ取得とIRR計算を実行するスクリプト
- `.github/workflows/irr.yml`: GitHub Actionsワークフローファイル（スクリプト実行用）

## ローカルでの使用方法
必要な依存関係をインストールし、スクリプトを実行してください：

```bash
pip install -r requirements.txt
python src/fetch_and_compute_irr.py
```

IRRの計算結果は`data/processed/irr_results.csv`ファイルに出力されます。

GitHub Actionsワークフローも同じスクリプトを実行し、計算結果を成果物としてアップロードします。
賃金データ取得用のe-Stat APIキーを`ESTAT_APP_ID`シークレットとして設定してください。
