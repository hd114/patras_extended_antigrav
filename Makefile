SHELL := /bin/bash

.PHONY: check validate snapshot snapshot-check handoff

check: validate
	@echo "OK: checks passed"

validate:
	python tools/validate_snapshot.py

snapshot:
	python tools/snapshot.py --repo-root . --snapshot docs/hub/CONTEXT_SNAPSHOT.md
	python tools/work_items_to_snapshot.py
	python tools/task_graph_to_snapshot.py
	python tools/repo_inventory_to_snapshot.py
	python tools/adr_to_snapshot.py

snapshot-check:
	python tools/validate_snapshot.py
	bash tools/check_snapshot_up_to_date.sh

handoff:
	bash tools/export_handoff_bundle.sh
