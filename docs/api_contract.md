# API Contract (Track B)

## Create Session

`POST /v1/liveness/sessions`

Request (JSON):

- `device_id` (optional)
- `app_version` (optional)
- `expected_duration_seconds` (optional)

Response:

- `session_id`
- `stream_url`
- `access_token`
- `expires_at_unix`
- `capture_config` (`fps`, `roi_width`, `roi_height`, `min_duration_seconds`, `max_duration_seconds`, `format`)

## Stream

`WS /v1/liveness/sessions/{id}/stream?token=...`

Client message (`roi_frame_chunk`):

- `type`
- `session_id`
- `seq`
- `timestamp_ms`
- `image_format`
- `image_bytes_b64`

Client message (`end_stream`):

- `type=end_stream`

Server events:

- `ack`
- `quality_feedback`
- `error`
- `complete`

## Get Result

`GET /v1/liveness/sessions/{id}/result`

Returns:

- `status`
- `decision`
- `liveness_score`
- `confidence`
- `method_scores`
- `quality_summary`
- `failure_reasons`
