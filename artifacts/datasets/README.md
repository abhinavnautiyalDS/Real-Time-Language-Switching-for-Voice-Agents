# Dataset split archive

Expected archive: `dataset_splits_audit1.zip`

The archive contains the train, validation and test CSV splits plus audit reports. The dataset split sizes from the training run were:

- Train: 370,170 rows
- Validation: 42,710 rows
- Test: 47,120 rows

The split was group-aware and leakage checks were reported as zero overlap for connected groups, normalized text, computed exact-text hashes, `source_group_id`, and `template_id`.

Note: a stored `text_sha256` mismatch was observed for some rows during the audit and remains unresolved; do not treat that metadata field as verified until its generation is investigated.
