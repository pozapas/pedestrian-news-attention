# Whose Deaths Dominate the News? Replication materials

Code and data for *Whose Deaths Dominate the News? Unequal Coverage of U.S. Pedestrian
Fatalities* (Amir Rafe and Subasish Das, Texas State University).

The study models how news attention is distributed across 12,874 news-derived
pedestrian fatal crash events in the United States from 2016 through 2025, linked to
18,843 articles. Every model covariate is an indicator for whether the first linked
article reported the attribute.

## Contents

| Path | What it holds |
|---|---|
| `data/events_first_article.csv` | Event panel used by every model. Attributes read from the first linked article, outcomes (`n_articles`, `n_outlets`, follow-up timing), state, year, review status |
| `data/events_event_record.csv` | The same events with attributes read from the consolidated event record, used only for the comparison column of the rate-ratio table |
| `data/event_article_bridge.csv` | Every event-to-article link with the article address, outlet, publication date, discovery source, retrieval run, retrieval date, and extraction model |
| `data/extraction_fields.md` | Fields of the stored extraction output |
| `data_build/` | The two event panels in Parquet form, as read by the scripts |
| `analysis/` | All analysis scripts, numbered in run order |
| `outputs/` | Model outputs (JSON), manuscript tables, and figures |
| `validation_app/` | Human validation: sampling design and stratum weights, the de-identified answers of the two coders with server-side timing, and the adjudication record |
| `webapp/` | Source of the blinded coding interface used for the validation |
| `figure_style/` | Shared plotting palette and style |

## Reproducing the results

Python 3.11 or later with `numpy`, `pandas`, `scipy`, `matplotlib`, `pyarrow`, and
`openpyxl`.

```
python analysis/p15_single_spec.py        # intensity model, both predictor codings (Table 3)
python analysis/p25_rebuild_models.py     # sensitivities, archetypes, recurrence, multiplicity
python analysis/p27_method_checks.py      # ZTP, dispersion envelope, recurrence at 2,000 replicates
python analysis/p28_figures.py            # Figures 3 and 4
python analysis/p26_table1_firstarticle.py
python analysis/p24_validation_table.py   # Table 4, human validation
```

Scripts `p01`, `p08`, `p10`, and `p20` read the full-text article corpus, which is not
redistributed. They expect it at the path in the `CRASHNEWS_DIR` environment variable.
All other scripts run from the files in this repository.

## What is not included

Article text is not redistributed because it remains under the copyright of its
publishers, and it can be retrieved from the addresses in `event_article_bridge.csv`.
Event summaries, article titles, and the free-text extraction fields are withheld
because they carry personal names. Coder identities and the credentials of the coding
interface are withheld.

## License

Code is released under the MIT License (`LICENSE`). Data files are released under the
Creative Commons Attribution 4.0 International License (CC BY 4.0).
