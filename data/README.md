# data/

`prices.csv` — cached adjusted-close prices (one column per ticker) so the
pipeline runs without internet. It is written automatically on the first
successful Yahoo Finance download and committed on purpose: judges must be
able to reproduce results offline (organisers' "clean start" rule).

Delete this file to force a fresh download; if both the cache and the
download fail, the pipeline falls back to clearly-flagged synthetic data
(never present results based on synthetic data without saying so in the
write-up).
