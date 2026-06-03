# Observation Log Layout

Raw observation logs stay in the `obslog_*` subdirectories.

Processed outputs from `../read_obslog.ipynb` or `process_obslog.py` are written to `processed/`:

- `co_<run>.csv`: filtered CO observation-log rows for each run.
- `date_visit_id_<run>.csv`: compact `date,visit_id` lists for download scripts.
- `date_visit_id_S25A.csv`: combined March, May, June, and September visit list.

The processing notebook is a thin wrapper around `process_obslog.py`. For batch use:

```bash
python obslog/process_obslog.py --config obslog/config_nov2025.json
```

Use `--config path/to/config.json` for a different set of future runs.
