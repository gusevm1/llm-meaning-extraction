# LLM Meaning Extraction & Compression Experiments

A research framework for testing extreme text compression capabilities of Large Language Models while preserving semantic meaning. This project accompanies the term paper "Speaking in Code: Probing the Efficiency–Expressivity Trade-off in Large Language Models through a Compression–Decompression Game" for Philosophy of Language & Computation at ETH Zürich.

## Overview

This project compares the compression performance of **Gemini 2.5 Pro** and **GPT-4o** on a standardized dataset of sentences, measuring both compression ratios and semantic preservation through multiple evaluation metrics.

## Key Features

- 🗜️ **Extreme Compression**: Target ≤15% of original length while preserving meaning
- 🧠 **Model Comparison**: Gemini 2.5 Pro vs GPT-4o with identical prompts
- 📊 **Comprehensive Evaluation**: BLEU scores, embedding similarity, keyword preservation, LLM-based semantic scoring
- 📈 **Rich Analysis**: Automated plotting and statistical analysis with human survey validation
- 🔄 **Blind Decompression**: Models reconstruct meaning without seeing originals

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/gusevm1/llm-meaning-extraction.git
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

# Test GPT-4o connection  
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
# Performance comparison analysis
python performance_comparison_analysis.py

# Individual sentence visualizations
python visualize_individual_sentences_plotly.py

# Human survey analysis (requires survey data)
python human_survey_analysis.py
```

## Project Structure

```
llm-meaning-extraction/
├── 📋 Core Experiments
│   ├── extreme_compression_experiment_gemini.py   # Gemini 2.5 Pro experiments
│   ├── extreme_compression_experiment_openai.py   # GPT-4o experiments
│   └── semantic_evaluator.py                      # Evaluation metrics
│
├── 📊 Analysis & Visualization
│   ├── performance_comparison_analysis.py         # Model comparison analysis
│   ├── visualize_individual_sentences_plotly.py   # Individual sentence plots
│   ├── human_survey_analysis.py                   # Human evaluation analysis
│   └── dataset.py                                 # Standard sentence dataset
│
├── 🔧 Utilities
│   ├── gcp_api.py                                 # Gemini connection test
│   └── openai_api.py                              # OpenAI connection test
│
├── 📈 Results Directories
│   ├── performance_comparison/                    # Model comparison outputs
│   ├── individual_sentences_plotly/               # Per-sentence visualizations
│   ├── human_survey_analysis/                     # Human evaluation plots
│   └── *.csv                                      # Experimental data files
│
└── ⚙️ Configuration
    ├── requirements.txt                           # Python dependencies
    ├── setup.py                                   # Package setup
    ├── env_template.txt                           # Environment template
    └── .gitignore                                 # Git ignore patterns
```

## Methodology

### Compression Strategy

Both models use **identical progressive prompts** with escalating urgency:

1. **Attempt 1-2**: Baseline compression requests with Unicode encouragement
2. **Attempt 3**: Creative push with mathematical notation and symbols  
3. **Attempt 4-5**: Critical directives with Unicode as "semantic containers"
4. **Attempt 6-7**: Final attempts emphasizing maximum meaning density

Temperature gradually increases (0.1→0.7) to balance stability with creative exploration.

### Evaluation Metrics

- **Compression Ratio**: Achieved vs target (≤15%)
- **BLEU Score**: N-gram overlap with original sentence
- **Embedding Similarity**: Cosine similarity using all-MiniLM-L6-v2
- **Keyword Preservation**: Retention of non-stop content words
- **LLM Similarity**: GPT-4o semantic equivalence rating (0-1 scale)
- **Composite Score**: Weighted combination (30% Embedding + 30% LLM + 20% BLEU + 20% Keywords)

### Model Configuration

- **Gemini 2.5 Pro**: Reasoning-oriented architecture with adaptive thinking
- **GPT-4o (2024-11-20)**: Omni model with 128k context window
- **Target**: ≤15% of original character count
- **Max Attempts**: 7 per sentence with early stopping on success

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
   Semantic Score: 0.654 (Good quality)
```

## Key Findings

- **Universal Success**: Both models achieved 100% success rate in reaching ≤15% compression target
- **Reasoning Advantage**: Gemini 2.5 Pro demonstrated superior semantic preservation (0.576 vs 0.474 composite score)
- **Quality Distribution**: Gemini achieved 55% "Good" ratings with 0% "Poor", vs GPT-4o's 15% "Good" and 30% "Poor"
- **Human Alignment**: Moderate correlation (r = 0.516) between automated scores and human similarity ratings
- **Evaluation Insights**: LLM-based assessment (r = 0.644) aligns better with human judgment than embedding similarity (r = 0.176)

## ⚠️ Cost Warning

**Both APIs will charge your account for usage!**

- **OpenAI API**: GPT-4o costs ~$2.50 input / $10.00 output per 1M tokens
- **Google API**: Gemini 2.5 Pro costs ~$1.25 input / $5.00 output per 1M tokens
- **Full Project Cost**: Complete experimental runs cost approximately **25 CHF total**
- **Single Experiment**: ~0.50-1.00 CHF per full dataset run (20 sentences)
- **Connection Tests**: Nearly free (~0.01 CHF per test)

**Recommendation**: Start with connection tests and single experiments before running full dataset experiments.

## API Requirements

- **OpenAI API**: GPT-4o access required
- **Google API**: Gemini 2.5 Pro access required  
- **Rate Limits**: Experiments include appropriate delays
- **Budget Planning**: Monitor your API usage through respective dashboards

## Research Applications

- 🔬 **Philosophy of Language**: Form vs. meaning debate in AI systems
- 📡 **Information Theory**: Semantic density and compression limits
- 🧠 **Cognitive Science**: Reasoning vs. pattern matching in language models
- 📊 **Evaluation Methodology**: Human-AI alignment in semantic assessment

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is provided for academic and research purposes. Please cite appropriately if using this work.

## Citation

If you use this research framework, please cite:

```bibtex
@software{gusev2025_llm_compression,
  title={LLM Meaning Extraction \& Compression Experiments: Speaking in Code},
  author={Maxim Gusev},
  year={2025},
  url={https://github.com/gusevm1/llm-meaning-extraction},
  note={Term paper for Philosophy of Language \& Computation, ETH Zürich}
}
```

## Contact

For questions about this research or the accompanying term paper:
- **Author**: Maxim Gusev
- **Email**: gusevm@ethz.ch
- **Institution**: ETH Zürich, Philosophy of Language & Computation (Spring 2025)
