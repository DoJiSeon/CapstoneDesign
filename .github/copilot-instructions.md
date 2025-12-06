# Copilot Instructions for CapstoneDesign (Llama-AVSR)

## Project Overview
This is a **multimodal Audio-Visual Speech Recognition (AVSR) system** combining:
- **AV-HuBERT** encoder: Extracts audio/video features (pre-trained, frozen)
- **Llama LLM**: Decodes encoded features into text with LoRA fine-tuning
- **UADF Block** (optional): Uncertainty-aware dynamic fusion of audio/video modalities

The architecture unifies three modalities:
1. **Audio-only**: Whisper/WavLM → Llama
2. **Video-only**: AV-HuBERT → projector → Llama  
3. **Audio-visual**: Dual encoders → UADF fusion → projector → Llama

## Key Architecture Files & Data Flow

### Model Stack (`models/`)
- **`modeling_AVSRLLM.py`**: Main `AVSR_LLMs` module orchestrating encoders → projectors → UADF → LLM. Supports modality selection via `modality` parameter (audio/video/audiovisual/audiovisual_avhubert).
- **`lightning.py`**: `ModelModule_LLM` wraps model in PyTorch Lightning. Manages tokenizer, hidden size lookup (`get_llm_hidden_size()`), and special tokens (`<audio>`, `</audio>`, `<video>`, `</video>`).
- **`uadf_block.py`**: `UADFBlock` dynamically weights audio/video fusion using uncertainty (entropy). **Critical safety fix**: Casts to Float32 during entropy computation and interpolation to prevent NaN in BFloat16.
- **`Llama_LoRA.py`**: LoRA-enabled Llama for parameter-efficient fine-tuning.

### Data Pipeline (`datamodule/`)
- **`av_dataset.py`**: `AVDataset_LLM` loads multi-modal samples. Video is read via `torchvision.io.read_video()`, audio via `torchaudio.load()`. Returns `{"video": (T, C, H, W), "audio": (T, 1), "tokens": text}`.
- **`grid_dataset.py`**: GRID speech dataset loader for audio/video alignment testing.
- **`data_module.py`**: `DataModule_LLM` wraps dataset. **Key collation logic**: `collate_LLM()` pads sequences, handles train/val vs. inference modes differently, and manages special token insertion.
- **`transforms.py`**: Audio/video preprocessing (normalization, resizing).

### Training & Inference
- **`train.py`**: PyTorch Lightning trainer. Key args: `--modality`, `--use-lora-avhubert`, `--use-uadf`. Supports distributed training via DDPStrategy.
- **`inference_avsr.py`**: Single-sample inference (takes `name.wav` + `name.mp4`). Includes memory profiling (`print_memory_usage()`). Output: text transcription.
- **`eval_uadf_wer.py`**: Batch inference over folders to compute WER/CER. Template-based (copied from Gemini). Modify as needed for your metric/dataset setup.
- **`eval.py`**: Alternative batch inference script.

### Debugging & Utility Scripts
- **`check_audio.py`**: Inspect audio tensor dimensions (sample rate, channels, duration).
- **`check_video.py`**: Inspect video tensor dimensions (frames, channels, height, width).

### Configuration & Tests
- **`configs/llama_avsr_lrs3.yaml`**: Default inference config (beam size, max tokens).
- **`tests/test_uadf_block.py`**: Unit test for UADF with GRID dataset.

## Critical Developer Workflows

### Training
```bash
python train.py \
  --modality audiovisual_avhubert \
  --llm-model meta-llama/Llama-2-7b-hf \
  --audio-encoder-name whisper-base \
  --pretrain-avhubert-enc-audiovisual-path ./checkpoints/av_hubert.pt \
  --use-lora-avhubert \
  --unfrozen_modules lora_avhubert \
  --use-uadf
```

### Single-Sample Inference
```bash
python inference_avsr.py \
  --checkpoint ./checkpoints/best.ckpt \
  --modality audiovisual \
  --video-path sample.mp4 \
  --output-txt result.txt
```
**Input format**: `{name}.mp4` + `{name}.wav` (same prefix, different extensions)  
**Output**: Text transcription written to file.

### Batch Evaluation (WER/CER)
```bash
python eval_uadf_wer.py --checkpoint ./model.ckpt --data-dir ./test_samples/
```
**Note**: `eval_uadf_wer.py` is template-based (initially from Gemini). Modify script as needed for your metric/dataset requirements.

### Dimension Checking
```bash
python check_audio.py --audio-path sample.wav    # Check audio shape
python check_video.py --video-path sample.mp4    # Check video shape
```

### Testing
```bash
pytest tests/test_uadf_block.py -v
```

## Project-Specific Conventions & Patterns

### Modality Handling
- **Modality strings**: "audio", "video", "audiovisual", "audiovisual_avhubert" (distinct from plain "audiovisual" because av_hubert uses different preprocessing).
- All modality branches must be handled in `collate_LLM()`, dataset loaders, and model forward pass.
- Whisper/WavLM are used for audio; AV-HuBERT requires `logfbank` + stacking (see `av_dataset.py:115-120`).

### Special Token Management
- Always register modal tokens: `<audio>`, `</audio>`, `<video>`, `</video>` via `tokenizer.add_special_tokens()` before model initialization.
- Tokenizer must be added to model embedding via `resize_token_embeddings()` when adding new special tokens.
- Llama3 requires manual post-processor fix (see `lightning.py:70-80`) due to FastTokenizer not auto-appending EOS.

### Downsampling Ratios
- Audio/video encoders produce variable-length token sequences. Use downsampling to reduce sequence length:
  - `downsample_ratio_audio`: Typically 4x (e.g., 16kHz audio → 4k frames per second)
  - `downsample_ratio_video`: Typically 2x (e.g., 25 fps video)
- Applied in dataset loaders and UADF alignment via interpolation.

### UADF Fusion (When Used)
- UADF Block (`uadf_block.py`) is only active when `modality == "audiovisual"` and `use_uadf=True`.
- **Alignment requirement**: Audio and video sequences must be same length before fusion. UADF handles interpolation internally.
- **Fusion methods**: "uncertainty" (default, entropy-based) or "attention" (multi-head attention).
- **Safety pattern**: Always convert to Float32 before entropy/interpolation; restore original dtype after. Prevents NaN in BFloat16 mixed precision.

### LoRA & Unfrozen Modules
- AV-HuBERT encoder is **frozen by default**. LoRA adapters are optional via `--use-lora-avhubert`.
- Valid values for `--unfrozen_modules`: "lora_avhubert", "llm_head", "projector", or combinations.
- Check assertions in `lightning.py` (line ~65) when using LoRA.

### PyTorch Lightning Integration
- Model wrapped in `ModelModule_LLM` (inherits `LightningModule`).
- Custom scheduler: `WarmupCosineScheduler` from `utils/cosine.py`.
- Callbacks: `ModelCheckpoint`, `LearningRateMonitor`. Logging via `WandbLogger`.
- Train/val/test modes handled in collate via `is_trainval` flag.

### Memory & Precision
- **FP16 loading**: Whisper and other encoders loaded in FP16 on GPU to save memory (see `modeling_AVSRLLM.py:55`).
- **Inference memory profiling**: `inference_avsr.py` includes `print_memory_usage()` to track GPU/CPU RAM during inference.
- Batch size set to 1 for inference by design (autoregressive decoding).

## Integration Points & External Dependencies

### Pre-trained Models (Must Be Available)
- **AV-HuBERT**: Load via `fairseq.checkpoint_utils.load_model_ensemble_and_task()` (see `modeling_AVSRLLM.py:80`).
- **Llama**: From HuggingFace Hub (e.g., `meta-llama/Llama-2-7b-hf`).
- **Whisper/WavLM**: From HuggingFace.
- All paths specified via command-line args: `--pretrain-avhubert-enc-*` and `--llm-model`.

### fairseq Integration
- AV-HuBERT training code reused from `av_hubert/` subdirectory.
- Import pattern: `import fairseq` → `fairseq.checkpoint_utils.load_model_ensemble_and_task()`.
- Config files in `av_hubert/conf/` define AV-HuBERT architecture.

### Dataset Formats
- **LRS3** (Lip Reading Sentences 3): Expected structure: `{root_dir}/{dataset_name}/{rel_path}.mp4` + `.wav`.
- **GRID**: Simple format with aligned video/audio pairs.
- Label files: CSV format `dataset_name,rel_path,input_length,_,text`.

## Debugging & Testing Strategy

### Common Issues & Fixes
1. **NaN in UADF**: Float32 casting is essential; check `uadf_block.py:compute_uncertainty()` and `align_sequences()`.
2. **Token mismatch**: Verify special tokens registered before model initialization; compare tokenizer token count with model embedding size.
3. **Modality branch mismatch**: If a modality string is used but not handled, the dataset or collate will fail. Ensure all three places (dataset, collate, model) align.
4. **Sequence length misalignment**: UADF handles video padding, but if bypass UADF, ensure audio/video lengths match before fusion.

### Test Pattern
- Unit tests in `tests/`: Use `pytest` to validate individual components (UADF block, GRID dataset, etc.).
- Integration test: Run `inference_avsr.py` on a single sample to verify end-to-end pipeline.

## When Modifying the Codebase
1. **Adding new modality**: Update `modality` parameter in model, dataset, and collate function.
2. **Changing LLM model**: Update `llm_size` dict in `lightning.py` if new model hidden size is unknown.
3. **Adjusting special tokens**: Edit token list in `lightning.py:13-16`; re-run model initialization to resize embeddings.
4. **Extending UADF fusion methods**: Add new branch in `uadf_block.py:__init__()` and `forward()`.
