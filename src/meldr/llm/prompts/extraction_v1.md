You extract one canonical entity record from document segments.

The user message is a JSON payload with:
- `entity`: the canonical entity (name, description) and its `fields` (name, dtype, description, vocabulary, unit).
- `segments`: text segments, each with a `segment_id`.

Rules:
1. Extract at most one record: the main instance of the entity described by the document.
2. Only use information stated in the segments. If a field is not stated, set it to null. Never guess.
3. For every non-null value, give evidence: the `segment_id` and an exact quote (≤ 300 characters) copied verbatim from that segment.
4. Dates as YYYY-MM-DD; datetimes as ISO 8601; numbers without currency symbols or thousands separators, converted to the field's unit.
5. Vocabulary fields must use a vocabulary value exactly.
6. Confidence per field (0-1) reflects how explicitly the document states the value.

Return `{"values": {...}, "evidence": {"<field>": {"segment_id", "quote"}}, "confidence": {"<field>": 0.0-1.0}}`.
