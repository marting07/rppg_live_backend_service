# API Contract (Track B)

## Create Session

`POST /v1/liveness/sessions`

Request (JSON):

- `device_id` (optional)
- `app_version` (optional)
- `preferred_method` (optional)
- `expected_duration_seconds` (optional)

Response:

- `session_id`
- `stream_url`
- `access_token`
- `expires_at_unix`
- `capture_config` (`fps`, `patch_rows`, `patch_cols`, `min_duration_seconds`, `max_duration_seconds`, `transport_format`)

## Stream

`WS /v1/liveness/sessions/{id}/stream?token=...`

Client message (`sample_summary_chunk`):

- `type`
- `seq`
- `timestamp_ms`
- `patches`
  - `patch_id`
  - `mean_rgb`
  - `weight`
- `local_quality`
  - `face_present`
  - `brightness`
  - `motion_score`
  - `roi_coverage`

Client message (`end_stream`):

- `type=end_stream`

Server events:

- `ack`
- `quality_feedback`
- `provisional_result`
- `stable_result`
- `error`
- `complete`

## Stop Session

`POST /v1/liveness/sessions/{id}/stop`

Finalizes the session and returns the same payload shape as `GET .../result`.

## Get Result

`GET /v1/liveness/sessions/{id}/result`

Returns:

- `status`
- `decision`
- `liveness_score`
- `confidence`
- `selected_method`
- `method_scores`
- `quality_summary`
- `operational_metrics`
- `method_summary`
- `failure_reasons`

## Diagnostics

`GET /v1/liveness/sessions/{id}/diagnostics`

Returns live counters and the latest quality/result event snapshot for host-app diagnostics.
