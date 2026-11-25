# Data, results and documentation license

The license in [LICENSE](LICENSE) (MIT) covers the source code: `src/`, `scripts/`, `tools/`, `tests/`, `configs/`, `prompts/` and `metadata/`.

The material the authors created — human annotations, comparison designs, judge outputs, metrics and reports in `data/` and `results/`, the documents in `docs/` (including the paper draft) — is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**: <https://creativecommons.org/licenses/by/4.0/>. Copyright (c) 2026 Zhou Yang and Chengkai Fang.

## Third-party material, not covered by these licenses

- **Product information.** Product titles, features, descriptions, categories and details (for example `product_items.jsonl`, `*_products.jsonl` and split files) come from the Office Products metadata of Amazon Reviews 2023 (Hou et al., 2024; <https://amazon-reviews-2023.github.io/>). They remain subject to that dataset's own terms and are included here only for research reproducibility.
- **Model-generated ads.** Ads written by third-party models (Qwen, DeepSeek, Claude, GPT, Gemma, GLM, LongCat, Mistral) are released under CC BY 4.0 to the extent the authors hold any rights in them; the providers' terms of use may also apply.

## Citation

If you use this material, please cite the dataset and this project:

```bibtex
@article{hou2024bridging,
  title   = {Bridging Language and Items for Retrieval and Recommendation},
  author  = {Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal = {arXiv preprint arXiv:2403.03952},
  year    = {2024}
}

@misc{adgeniusplus2026,
  title  = {{AdGenius+}: Supervised Fine-Tuning versus On-Policy Distillation for Small-Model Advertising Copy},
  author = {{Zhou Yang} and {Chengkai Fang}},
  year   = {2026},
  note   = {CS 289A course project, UC Berkeley},
  url    = {https://github.com/YANGZHOU66666/AdGenius-Plus}
}
