# Fields of the stored extraction output

Each linked article carries one structured record produced by the extraction step (model recorded per article in `event_article_bridge.csv`, column `llm_model`). The fields below are the keys of those records and the JSON types observed. Free-text fields (for example `crash_summary`, `source_evidence`, and `victim_name`) are not released because they reproduce article text or personal names.

| Field | Observed types |
|---|---|
| `alcohol_drugs_involved` | str |
| `causal_mechanism` | NoneType, str |
| `charges` | NoneType, list, str |
| `city` | NoneType, str |
| `cleaned_article_text` | str |
| `confidence` | float, str |
| `county` | NoneType, list, str |
| `crash_date` | NoneType, str |
| `crash_summary` | NoneType, str |
| `driver_action` | NoneType, list, str |
| `fatality_count` | NoneType, int |
| `hit_run` | NoneType, bool |
| `investigating_agency` | NoneType, str |
| `is_pedestrian_fatal_crash` | bool |
| `lighting_condition` | NoneType, str |
| `location_text` | NoneType, str |
| `pedestrian_action` | NoneType, str |
| `reject_reason` | NoneType, str |
| `road_name` | NoneType, str |
| `road_type` | NoneType, str |
| `source_evidence` | dict, list, str |
| `speed_limit` | NoneType, str |
| `state` | str |
| `uncertainty_flags` | list |
| `vehicle_type` | NoneType, list, str |
| `victim_age` | NoneType, float, int, str |
| `victim_gender` | NoneType, str |
| `victim_name` | NoneType, str |
| `weather_condition` | NoneType, str |
