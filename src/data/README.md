# How to run the data pipeline
1. Run the download.py
2. Manually clean download the worldbank data.
3. Run harmonize.py
4. Run build_features.py
5. Run the tests at tests/sanity/test_build_features.py 
6. Run the scan_parquet.py to check the resulting tables for any possible issues/problemls.

```bash
# check speed on limited number of rows
python -m scan_parquet   --path /home/ubuntu/repos/kaggle-mitsui/data/processed/features_long.parquet   --out  /home/ubuntu/repos/kaggle-mitsui/data/interim/feature_scan_quick.csv   --batch-size 200000   --sample-size 100000   --include ""   --exclude date symbol src unit frequency   --exclude-prefix target_   --use-threads True   --limit-rows 2_000_000   --progress-every 500_000

# run the full scan
python -m scan_parquet   --path /home/ubuntu/repos/kaggle-mitsui/data/processed/features_long.parquet   --out  /home/ubuntu/repos/kaggle-mitsui/data/interim/feature_scan_full.csv   --batch-size 250000   --sample-size 200000   --exclude date symbol src unit frequency   --exclude-prefix target_   --u
se-threads True   --limit-rows 0   --progress-every 1_000_000
```

Check the resulting feature_scan.csv manually.