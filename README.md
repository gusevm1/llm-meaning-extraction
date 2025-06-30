# LLM Meaning Extraction & Compression Experiments

A research framework for testing extreme text compression capabilities of Large Language Models while preserving semantic meaning.

## Overview

This project compares the compression performance of **Gemini 2.5 Pro** and **GPT-4.1** on a standardized dataset of sentences, measuring both compression ratios and semantic preservation through multiple evaluation metrics.

## Key Features

- 🗜️ **Extreme Compression**: Target 15% of original length while preserving meaning
- 🧠 **Model Comparison**: Gemini 2.5 Pro vs GPT-4.1 with identical prompts
- 📊 **Comprehensive Evaluation**: BLEU scores, embedding similarity, keyword preservation
- 📈 **Rich Analysis**: Automated plotting and statistical analysis
- 🔄 **Blind Decompression**: Models reconstruct meaning without seeing originals

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd llm-meaning-extraction

# Install dependencies
pip install -r requirements.txt
```

### 2. API Setup

Copy the environment template and add your API keys:

```bash
cp env_template.txt .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key_here
# GOOGLE_API_KEY=your_google_key_here
```

### 3. Test Connections

Verify your API connections before running experiments:

```bash
# Test Gemini 2.5 Pro connection
python gcp_api.py

# Test GPT-4.1 connection  
python openai_api.py
```

### 4. Run Experiments

**Single experiment** (quick test):
```bash
# Gemini experiment
python extreme_compression_experiment_gemini.py

# OpenAI experiment
python extreme_compression_experiment_openai.py
```

**Full dataset experiments** (20 sentences):
Edit the `main()` function in each script to call `run_full_dataset_experiment()`

### 5. Analyze Results

Generate plots and statistical analysis:

```bash
python analyze_results.py
```

## Project Structure

```
llm-meaning-extraction/
├── 📋 Core Experiments
│   ├── extreme_compression_experiment_gemini.py   # Gemini 2.5 Pro experiments
│   ├── extreme_compression_experiment_openai.py   # GPT-4.1 experiments
│   └── semantic_evaluator.py                      # Evaluation metrics
│
├── 🔧 Utilities
│   ├── gcp_api.py                                 # Gemini connection test
│   ├── openai_api.py                              # OpenAI connection test
│   ├── dataset.py                                 # Standard sentence dataset
│   └── analyze_results.py                         # Results analysis & plotting
│
├── 📊 Latest Results
│   ├── compression_experiments_gemini_20250630_220351.csv
│   ├── compression_experiments_openai_20250630_225241.csv
│   ├── filtered_results_gemini_20250630_224808.csv
│   ├── filtered_results_openai_20250630_223421.csv
│   ├── compression_analysis_gemini.png
│   └── compression_analysis_openai.png
│
└── ⚙️ Configuration
    ├── requirements.txt                           # Python dependencies
    ├── setup.py                                   # Package setup
    ├── env_template.txt                           # Environment template
    └── .gitignore                                 # Git ignore patterns
```

## Methodology

### Compression Strategy

Both models use **identical progressive prompts**:

1. **Attempt 1**: Soft request for target compression
2. **Attempt 2-3**: Increased urgency, introduce unicode concepts  
3. **Attempt 4-5**: Critical directives, unicode as "semantic containers"
4. **Attempt 6-7**: Final attempts with maximum meaning density focus

### Evaluation Metrics

- **Compression Ratio**: Achieved vs target (15%)
- **BLEU Score**: N-gram overlap with original
- **Embedding Similarity**: Cosine similarity of sentence embeddings
- **Keyword Preservation**: Important word retention
- **LLM Similarity**: GPT-4 semantic comparison (0-10 scale)
- **Composite Score**: Weighted combination of all metrics

### Model Configuration

- **Gemini 2.5 Pro**: Dynamic reasoning, progressive temperature (0.1→0.7)
- **GPT-4.1**: Progressive temperature (0.1→0.7), identical prompts
- **Target**: ≤15% of original length
- **Max Attempts**: 7 per sentence

## Example Results

```
📊 EXPERIMENT RESULTS

📝 ORIGINAL (377 chars):
   The ancient Mediterranean trading routes, established by Phoenician merchants around 1200 BCE...

🗜️ COMPRESSED (39 chars):
   🏺⚖️1200BCE→🌊💰🏛️≈⭐🎭🔄

📤 DECOMPRESSED (289 chars):
   Ancient Mediterranean trade routes from 1200 BCE by Phoenicians enabled exchange of goods...

📈 COMPRESSION PERFORMANCE:
   Achieved Ratio: 0.103 (10.3% of original)
   Target: ≤15% ✅ SUCCESS
   Characters Saved: 338 (89.7%)
```

## ⚠️ Cost Warning

**Both APIs will charge your account for usage!**

- **OpenAI API**: Charges per token for GPT-4.1 (~$2.00 input / $8.00 output per 1M tokens)
- **Google API**: Charges per token for Gemini 2.5 Pro (~$1.25 input / $5.00 output per 1M tokens)
- **Development Cost**: The entire project development (including multiple validation runs with different models) cost approximately **1.00 CHF total**
- **Single Experiment**: ~0.10-0.20 CHF per full dataset run (20 sentences)
- **Connection Tests**: Nearly free (~0.01 CHF per test)

**Recommendation**: Start with connection tests and single experiments before running full dataset experiments.

## API Requirements

- **OpenAI API**: GPT-4.1 access required
- **Google API**: Gemini 2.5 Pro access required  
- **Rate Limits**: Experiments include appropriate delays
- **Budget Planning**: Monitor your API usage through respective dashboards

## Research Applications

- 🔬 **Cognitive Science**: How do LLMs compress semantic information?
- 📡 **Communications**: Ultra-dense information encoding
- 🧠 **AI Research**: Comparing reasoning vs pattern-matching approaches
- 📊 **Linguistics**: Semantic density and meaning preservation

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this research framework, please cite:

```bibtex
@software{llm_meaning_extraction,
  title={LLM Meaning Extraction \& Compression Experiments},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/llm-meaning-extraction}
}
```
