# Deep Learning Recreations

This repository collects three learning-focused deep learning projects:

- `Vision Transformer Recreation/` - a PyTorch Vision Transformer implementation with MNIST data download support and attention visualization output.
- `Flow Matching Experiment/` - a notebook-based flow matching experiment with generated particle/vector-field animations.
- `Chat GPT 2/` - a GPT-2 recreation with tokenizer utilities, model code, training scripts, Hugging Face weight loading, and supervised fine-tuning experiments.

Large generated artifacts are intentionally excluded from Git, including virtual environments, downloaded datasets, Python bytecode, and model checkpoints such as `gpt2_sft_final.pt`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Notes

- The GPT-2 project includes a more detailed README at `Chat GPT 2/README.md`.
- MNIST data for the Vision Transformer project can be regenerated with `Vision Transformer Recreation/download_visiontransformer_data.py`.
- The flow matching folder keeps the notebook and animation outputs so the experiment is viewable from GitHub.
