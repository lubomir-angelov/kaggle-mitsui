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

Exepcted results example:
```bash
Scanned 51 columns. Flagged 3:
  - month_start                    | flags=mostly_zero                              | std=0.178 | max=1 | miss=0.00%
  - month_end                      | flags=mostly_zero                              | std=0.178 | max=1 | miss=0.00%
  - z_21_xrank                     | flags=z_mean_shift;z_bad_std                   | std=0.299 | max=1 | miss=0.00%
```


month_start, month_end → “mostly_zero”: those are binary calendar flags. 

    Most days are not month boundaries, so zeros dominate.
    That’s expected and harmless. 
    We can even store them as bool/int8 to save memory.
    

z_21_xrank → “z_mean_shift; z_bad_std”: your “xrank” is a percent rank in [0,1] per day. 

    For a uniform variable on [0,1], the expected std is ~0.2887, not 1.0. 
    The scanner’s “z_*” heuristic assumes z-scores, so it flags the lower std/offset mean (~0.5). 
    Functionally fine.
