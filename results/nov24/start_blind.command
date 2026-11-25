#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
exec .venv/bin/python tools/rubric_annotation.py --gold-dir results/nov24/blind100 --annotations-dir results/nov24/annotations --annotator "${1:-reviewer}" --criteria factual_support concrete_product_info --annotation-set-id gold_balanced_100 --protocol-version factual-v0.2-plus-specificity-v0.1 --port 8876
