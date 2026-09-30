# Project Map

| BioMRAG component | Entry points |
|---|---|
| Unified multimodal retriever | `src/biomrag/models/vision_text_dual_encoder.py` |
| Four-GPU retriever training | `src/biomrag/retrieval/train_retriever.py`, `scripts/run_retriever_train.sh` |
| Inference-only retriever export | `src/biomrag/retrieval/export_inference_retriever.py` |
| Text passage segmentation | `src/biomrag/data/prepare_text_passages.py` |
| Knowledge-base assembly | `src/biomrag/data/assemble_kb.py` |
| Leakage and near-duplicate filtering | `src/biomrag/data/decontaminate_index.py` |
| BioMRAG IVF-PQ index/search | `src/biomrag/retrieval/build_faiss_index.py`, `search_faiss_index.py` |
| Candidate-subset sampling | `src/biomrag/proxy/sample_candidate_subsets.py` |
| Paired 7B response generation | `scripts/generate_proxy_responses.sh`, `merge_proxy_responses.sh` |
| GPT-4o teacher annotation/filtering | `src/biomrag/proxy/teacher_annotation.py`, `filter_teacher_annotations.py` |
| Released-subset selection | `src/biomrag/proxy/select_release_subsets.py`, `scripts/select_release_subsets.sh` |
| Proxy SFT construction | `src/biomrag/proxy/build_distillation_data.py`, `prepare_swift_sft_data.py` |
| Two-GPU proxy QLoRA and serving | `configs/proxy_qlora_swift.yaml`, `scripts/train_proxy_qlora_swift.sh`, `scripts/deploy_proxy_swift.sh` |
| Streaming subset deliberation | `src/biomrag/selection/streaming_selector.py`, `run_selection.py` |
| Final answer generation | `src/biomrag/generation/run_generation.py` |
| Semantic VQA evaluation | `src/biomrag/evaluation/vqa_judge.py` |
| BLEU/ROUGE/METEOR/CIDEr | `src/biomrag/evaluation/open_ended_metrics.py` |
| Latency aggregation | `src/biomrag/evaluation/latency_summary.py` |
