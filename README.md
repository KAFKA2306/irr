# IRR for Wage Data Investments

This repository computes the internal rate of return (IRR) for investing
Japanese wage and bonus income into a global equity index. The data is fetched
from public APIs and processed on each run.

## Structure
- `data/raw/`: downloaded wage/bonus and stock return data (ignored from git)
- `data/processed/`: generated IRR results
- `src/fetch_and_compute_irr.py`: script fetching data and computing IRR
- `.github/workflows/irr.yml`: GitHub Actions workflow executing the script

## Local Usage
Install dependencies and run the script:

```bash
pip install -r requirements.txt
python src/fetch_and_compute_irr.py
```

IRR results will be written to `data/processed/irr_results.csv`.

The GitHub Actions workflow will run the same script and upload the results as
an artifact. Set the `ESTAT_APP_ID` secret with your e-Stat API key for wage
data retrieval.
