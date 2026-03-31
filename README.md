# LLM Brain Analysis

A framework for using Large Language Models to analyze brain region functions across different species.

## Overview

This tool provides two primary analysis workflows:

1. **Functions Analysis** (`top-functions`): Identifies the top 5 functions associated with brain regions and creates similarity matrices using embeddings
2. **Probabilities Analysis** (`query-functions`): Calculates the probability of specific functions being associated with brain regions

All cloud-based LLM queries are routed through [OpenRouter](https://openrouter.ai/), giving you access to hundreds of models (OpenAI, Anthropic, Google, Meta, Mistral, etc.) with a single API key. The framework also supports:
- **BrainGPT**: A local neuroscience-specialised model (Llama-2 + LoRA adapter)
- **Dummy**: A mock model for testing without API usage

## Installation

### Prerequisites
- Python 3.12+
- Conda (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Hana-Ali/neuroLLM.git
   cd neuroLLM
   ```

2. Run the setup script (creates conda environment, installs dependencies, generates config templates):
   ```bash
   bash setup_environment.sh
   conda activate llm_neuro
   ```

   Or manually:
   ```bash
   conda create -n llm_neuro python=3.12
   conda activate llm_neuro
   conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn
   pip install openai requests python-dotenv
   # Only needed if using BrainGPT:
   pip install peft transformers torch
   # Only needed if using local embeddings (--embedding-provider local):
   pip install sentence-transformers
   ```

3. Set up API keys by creating a `.env` file in the project root:
   ```
   OPENROUTER_API_KEY=your-openrouter-key-here
   OPENAI_API_KEY=your-openai-key-here           # Only needed for top-functions (embeddings)
   HF_TOKEN=your-huggingface-token-here          # Only needed for BrainGPT
   ```

   | Key | Required? | Purpose | Where to get it |
   |-----|-----------|---------|-----------------|
   | `OPENROUTER_API_KEY` | Yes (for cloud models) | Routes all LLM queries | [openrouter.ai/keys](https://openrouter.ai/keys) |
   | `OPENAI_API_KEY` | Only for `top-functions` with `--embedding-provider openai` (default) | Generates text embeddings via `text-embedding-3-large` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
   | `HF_TOKEN` | Only for BrainGPT | Downloads Llama-2 base model from Hugging Face | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

   > **BrainGPT note:** The base model (`meta-llama/Llama-2-7b-chat-hf`) is gated. You must first request access at [huggingface.co/meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and wait for approval before your `HF_TOKEN` will work.

### Atlas Files

Create atlas files containing brain region names:

```
atlases/
├── human/
│   └── DesikanKilliany68.csv
├── macaque/
│   └── RM_NMT82_.csv
└── mouse/
    └── Allen72.csv
```

Each CSV should contain region names in the first column.

## Usage

```bash
python main.py {list-models|top-functions|query-functions|test} [OPTIONS]
```

### Commands

#### `list-models` -- List Available Models
Lists chat models available on OpenRouter with pricing, then exits.

```bash
python main.py list-models                  # all chat models (default)
python main.py list-models --filter free    # zero-cost models only
python main.py list-models --filter paid    # cheapest 3 paid models per provider
```

| Option | Description | Default |
|--------|-------------|---------|
| `--filter` | `all` shows every chat model, `free` shows zero-cost models, `paid` shows the 3 cheapest paid models per provider sorted by combined prompt + completion cost | `all` |

> **Note:** Free models on OpenRouter may return a `404` error if your account's privacy/data policy settings are too restrictive. If this happens, adjust your settings at [openrouter.ai/settings/privacy](https://openrouter.ai/settings/privacy) or use a paid model instead.

#### `top-functions` -- Embedding of Top Functions
Identifies top 5 functions for brain regions and creates similarity matrices:

```bash
python main.py top-functions --atlas-name DesikanKilliany68
```

#### `query-functions` -- Probabilistic Functional Association
Calculates probabilities of specific functions being associated with regions:

```bash
python main.py query-functions --atlas-name DesikanKilliany68 --functions "spatial cognition,memory,attention"
```

#### `test` -- Test Workflow
Runs a quick test of both analysis types using the dummy model:

```bash
python main.py test --atlas-name DesikanKilliany68
```

### Options

#### Shared options (`top-functions`, `query-functions`, `test`)

| Option | Description | Default |
|--------|-------------|---------|
| `--atlas-name` | Atlas to use (must exist in `atlases/{species}/`) | **Required** |
| `--species` | Target species: `human`, `macaque`, `mouse` | `human` |
| `--models` | Comma-separated OpenRouter model IDs, `braingpt`, or `dummy` | `dummy` |
| `--regions` | Comma-separated brain regions | All regions in atlas |
| `--separate-hemispheres` | Analyze left and right hemispheres separately | `False` |
| `--prompt-template-name` | Custom prompt template name | `default` |
| `--workers` | Number of parallel workers | `4` |
| `--skip-visualization` | Skip creating visualizations | `False` |
| `--skip-raw-saving` | Clean up raw data files after processing | `False` |
| `--max-tokens` | Maximum number of tokens to generate per LLM response | `256` |

#### `top-functions`-specific options

| Option | Description | Default |
|--------|-------------|---------|
| `--embedding-provider` | `openai` uses `text-embedding-3-large` (requires `OPENAI_API_KEY`). `local` runs `BAAI/bge-large-en-v1.5` on your machine (no API key needed). | `openai` |

#### `query-functions`-specific options

| Option | Description |
|--------|-------------|
| `--functions` | Comma-separated function names to query (e.g. `"spatial cognition,memory"`) |
| `--function-group` | Use a predefined function group from `functions.json` instead of listing functions |

> If neither `--functions` nor `--function-group` is given, the default function set from `functions.json` is used.

### Choosing Models

Models are specified by their **OpenRouter model ID** (e.g. `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`). To browse available models and their pricing:

```bash
python main.py list-models                # all chat models
python main.py list-models --filter paid  # cheapest 3 per provider — good starting point
```

You can pass one or more model IDs:

```bash
# Single model
--models "openai/gpt-4o-mini"

# Multiple models
--models "openai/gpt-4o-mini,anthropic/claude-3.5-sonnet,google/gemini-2.0-flash-001"

# Local BrainGPT model (requires HF_TOKEN)
--models "braingpt"

# Mix cloud and local
--models "openai/gpt-4o-mini,braingpt"

# Dummy model for testing (default)
--models "dummy"
```

## Examples

### Basic Usage

```bash
# Analyze functions for human brain using dummy model
python main.py top-functions --atlas-name DesikanKilliany68

# Analyze probabilities for specific functions
python main.py query-functions --atlas-name DesikanKilliany68 --functions "spatial cognition,memory,attention"

# Run test workflow
python main.py test --atlas-name DesikanKilliany68
```

### Advanced Usage

```bash
# Use a cloud model with hemisphere separation
python main.py top-functions --atlas-name DesikanKilliany68 \
  --models "openai/gpt-4o-mini" --separate-hemispheres --workers 8

# Analyze specific regions only
python main.py top-functions --atlas-name DesikanKilliany68 \
  --regions "hippocampus,amygdala,prefrontal cortex"

# Use function groups for probabilities
python main.py query-functions --atlas-name DesikanKilliany68 --function-group memory

# Compare multiple models
python main.py query-functions --atlas-name DesikanKilliany68 \
  --models "openai/gpt-4o-mini,anthropic/claude-3.5-sonnet" \
  --functions "memory,attention,language"
```

## Function Groups

Manage sets of related functions in `functions.json`:

```json
{
  "functions": [
    "cognitive control",
    "emotion",
    "language",
    "memory",
    "vision"
  ],
  "groups": {
    "memory": [
      "spatial cognition",
      "rationality",
      "creativity"
    ],
    "awareness": [
      "metacognition",
      "consciousness"
    ]
  }
}
```

Use groups with `--function-group memory` instead of listing individual functions.

## Prompt Templates

Customize LLM prompts by creating template files:

- `prompts/functions/custom_template.txt` -- For function analysis
- `prompts/probabilities/custom_template.txt` -- For probability analysis

Templates support variables:
- `{species}` -- Target species
- `{region}` -- Brain region name
- `{hemisphere_part}` -- Hemisphere phrase ("in the left hemisphere of the" or "in the")
- `{function}` -- Function name (probabilities only)

Use with `--prompt-template-name custom_template`

## Output Structure

Results are organized in the `results/` directory:

```
results/
├── prompts/           # Generated prompts used
├── raw/               # Raw LLM responses
├── embeddings/        # Vector embeddings
├── aggregated/        # Processed results
│   ├── functions/     # Function lists and similarity matrices
│   └── probabilities/ # Probability distributions
└── visualizations/    # Plots and heatmaps
    ├── similarities/  # Similarity matrix plots
    └── probabilities/ # Probability heatmaps
```

Each subdirectory is further organized as: `{species}/{atlas_name}/{model_name}/{prompt_template_name}/{hemisphere_setting}/`

## Troubleshooting

- Check `llm_prompting.log` for detailed execution logs
- Ensure API keys are properly set in `.env` file
- Verify atlas files exist in correct directory structure
- Use `--models dummy` for testing without API usage
- Use `list-models` to verify your OpenRouter API key works and see available models
- **BrainGPT not loading?** Make sure you've been granted access to `meta-llama/Llama-2-7b-chat-hf` on Hugging Face and that your `HF_TOKEN` has the correct permissions
- Check that function names in `--functions` match those in literature
