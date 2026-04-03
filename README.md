# LLM Brain Analysis

A framework for using Large Language Models to analyze brain region functions across different species. 

This repository provides code to illustrate the central approach in Luppi et al., "Cognitive cartography of mammalian brains using meta-analysis of AI experts." bioRxiv (2025) ([preprint]([https://doi.org/10.1038/s41551-024-01242-2](https://doi.org/10.64898/2025.12.01.691701))).
Authors: Andrea I. Luppi, Hana Ali, Zhen-Qi Liu, Filip Milisav, Alessandro Gozzi, Danilo Bzdok & Bratislav Misic.



## Overview

This tool provides three primary analysis workflows:

1. **Functions Analysis** (`top-functions`): Identifies the top 5 functions associated with brain regions, embeds them, and creates similarity matrices
2. **Probabilities Analysis** (`query-functions`): Calculates the probability of specific functions being associated with brain regions
3. **Rankings Analysis** (`rank-pairs`): Ranks pairs of brain regions by relevance to specific functions

Additional features:
- **Justification** (`--justify`): Ask the LLM to explain its reasoning alongside its answer (stored in parallel files, does not affect numeric processing)
- **Retesting** (`--retest X`): Query the LLM multiple times and average the results for reliability analysis (mean for probabilities, mode for rankings, semantic consensus + mean embedding for functions)

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
python main.py {list-models|top-functions|query-functions|rank-pairs|test} [OPTIONS]
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
python main.py top-functions --atlas-name DesikanKilliany68 --species human
```

#### `query-functions` -- Probabilistic Functional Association
Calculates probabilities of specific functions being associated with regions:

```bash
python main.py query-functions --atlas-name DesikanKilliany68 --species human \
  --functions "spatial cognition,memory,attention"
```

#### `rank-pairs` -- Pairwise Region Ranking
Ranks which of two brain regions is more relevant to a specific function:

```bash
# Rank specific pairs for specific functions
python main.py rank-pairs --atlas-name DesikanKilliany68 --species human \
  --functions "memory,attention" --pairs "cuneus:precuneus,hippocampus:amygdala"

# Generate all unique pairs from atlas (warning if >100 pairs)
python main.py rank-pairs --atlas-name DesikanKilliany68 --species human \
  --functions "memory"
```

#### `test` -- Test Workflow
Runs a quick test of all three analysis types using the dummy model:

```bash
python main.py test --atlas-name DesikanKilliany68 --species human
```

### Options

#### Shared options (`top-functions`, `query-functions`, `rank-pairs`, `test`)

| Option | Description | Default |
|--------|-------------|---------|
| `--species` | Target species: `human`, `macaque`, `mouse` | **Required** |
| `--atlas-name` | Atlas to use (must exist in `atlases/{species}/`) | Required when `--regions` is not provided |
| `--models` | Comma-separated OpenRouter model IDs, `braingpt`, or `dummy` | `dummy` |
| `--regions` | Comma-separated brain regions | All regions in atlas |
| `--separate-hemispheres` | Analyze left and right hemispheres separately | `False` |
| `--prompt-template-name` | Custom prompt template name | `default` |
| `--workers` | Number of parallel workers | `4` |
| `--skip-visualization` | Skip creating visualizations | `False` |
| `--skip-raw-saving` | Clean up raw data files after processing | `False` |
| `--max-tokens` | Maximum tokens per LLM response | `512` with `--justify`, `256` otherwise |
| `--justify` | Ask the LLM to provide a justification alongside its answer | `False` |
| `--retest` | Number of times to repeat each query and average results | `1` |
| `--temperature` | Temperature for model querying (higher = more variability across trials) | `0.0` |

#### `top-functions`-specific options

| Option | Description | Default |
|--------|-------------|---------|
| `--embedding-provider` | `openai` uses `text-embedding-3-large` (requires `OPENAI_API_KEY`). `local` runs `BAAI/bge-large-en-v1.5` on your machine (no API key needed). | `openai` |
| `--consensus-threshold` | Cosine similarity threshold for semantic clustering when computing consensus functions across retests | `0.80` |

#### `query-functions`-specific options

| Option | Description |
|--------|-------------|
| `--functions` | Comma-separated function names to query (e.g. `"spatial cognition,memory"`) |
| `--function-group` | Use a predefined function group from `functions.json` instead of listing functions |

> If neither `--functions` nor `--function-group` is given, the default function set from `functions.json` is used.

#### `rank-pairs`-specific options

| Option | Description |
|--------|-------------|
| `--functions` | Comma-separated function names to rank pairs for |
| `--function-group` | Use a predefined function group from `functions.json` |
| `--pairs` | Region pairs as `"region1:region2,region3:region4"`. If omitted, generates all unique pairs from the atlas |

> **Note:** Omitting `--pairs` generates all C(n,2) unique pairs from the atlas. For atlases with many regions this can produce a large number of queries. A warning is logged when pair count exceeds 100.

### Choosing Models

Models are specified by their **OpenRouter model ID** (e.g. `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`). To browse available models and their pricing:

```bash
python main.py list-models                # all chat models
python main.py list-models --filter paid  # cheapest 3 per provider -- good starting point
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

## Incremental Execution & Skipping

The framework automatically skips work that has already been completed, enabling incremental runs:

- **Trial-level skipping**: Each trial checks if its result file already exists. If a previous run completed trial 0-2 of 5, a re-run will only execute trials 3-4.
- **Final result skipping**: After all trials are complete, the framework checks if the final aggregated result exists before recomputing.
- **Full skip**: If all trials and the final result exist for a given (region, model, hemisphere, trial) combination, the entire task is skipped.
- **Justification files**: When `--justify` is used, skipping also verifies that corresponding justification files exist alongside numeric results.

This means you can safely re-run the same command after interruptions or failures, and only the missing work will be executed.

## Examples

### Basic Usage

```bash
# Analyze functions for human brain using dummy model
python main.py top-functions --atlas-name DesikanKilliany68 --species human

# Analyze probabilities for specific functions
python main.py query-functions --atlas-name DesikanKilliany68 --species human \
  --functions "spatial cognition,memory,attention"

# Run test workflow
python main.py test --atlas-name DesikanKilliany68 --species human
```

### Advanced Usage

```bash
# Use a cloud model with hemisphere separation
python main.py top-functions --atlas-name DesikanKilliany68 --species human \
  --models "openai/gpt-4o-mini" --separate-hemispheres --workers 8

# Analyze specific regions only
python main.py top-functions --atlas-name DesikanKilliany68 --species human \
  --regions "hippocampus,amygdala,prefrontal cortex"

# Use function groups for probabilities
python main.py query-functions --atlas-name DesikanKilliany68 --species human \
  --function-group memory

# Compare multiple models
python main.py query-functions --atlas-name DesikanKilliany68 --species human \
  --models "openai/gpt-4o-mini,anthropic/claude-3.5-sonnet" \
  --functions "memory,attention,language"

# Rank specific brain region pairs
python main.py rank-pairs --atlas-name DesikanKilliany68 --species human \
  --models "openai/gpt-4o-mini" \
  --functions "memory,attention" --pairs "cuneus:precuneus,hippocampus:amygdala"

# Rank all pairs from atlas for a function group
python main.py rank-pairs --atlas-name DesikanKilliany68 --species human \
  --models "openai/gpt-4o-mini" --function-group memory
```

### Justification

Add `--justify` to any analysis command to ask the LLM to explain its reasoning. Justifications are stored in parallel files alongside numeric results and do not affect processing (e.g., embeddings ignore justification text).

```bash
# Functions with justification
python main.py top-functions --atlas-name DesikanKilliany68 --species human \
  --models "openai/gpt-4o-mini" --justify

# Probabilities with justification
python main.py query-functions --atlas-name DesikanKilliany68 --species human \
  --models "openai/gpt-4o-mini" --functions "memory" --justify

# Rankings with justification
python main.py rank-pairs --atlas-name DesikanKilliany68 --species human \
  --models "openai/gpt-4o-mini" --functions "memory" \
  --pairs "cuneus:precuneus" --justify
```

### Retesting

Use `--retest X` to query the LLM multiple times and average the results. This measures reliability and produces richer output:

- **Probabilities**: Averaged via mean. Per-trial results, std, min, max are saved.
- **Rankings**: Averaged via mode (most common 1/2). Agreement ratio is tracked.
- **Functions**: Semantic consensus via agglomerative clustering on per-function embeddings (functions with cosine similarity above `--consensus-threshold` are grouped). Consensus list ranked by trial coverage then cluster size. Mean embedding computed across all trials.

```bash
# Retest probabilities 5 times
python main.py query-functions --atlas-name DesikanKilliany68 --species human \
  --models "openai/gpt-4o-mini" --functions "memory" --retest 5

# Retest with higher temperature (more variability across trials)
python main.py query-functions --atlas-name DesikanKilliany68 --species human \
  --models "openai/gpt-4o-mini" --functions "memory" \
  --retest 5 --temperature 0.9

# Combine all features
python main.py rank-pairs --atlas-name DesikanKilliany68 --species human \
  --models "openai/gpt-4o-mini" --functions "memory" \
  --pairs "cuneus:precuneus" --justify --retest 3
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
- `prompts/rankings/custom_template.txt` -- For ranking analysis

Templates support variables:
- `{species}` -- Target species
- `{region}` -- Brain region name
- `{hemisphere_part}` -- Hemisphere phrase ("in the left hemisphere of the" or "in the")
- `{function}` -- Function name (probabilities and rankings only)

Use with `--prompt-template-name custom_template`

## Output Structure

Results are organized in the `results/` directory. Each subdirectory is further organized as:
`{species}/{atlas_name}/{model_name}/{prompt_template_name}/{hemisphere_setting}/`

where `hemisphere_setting` is either `no_separation` or `separation/left` / `separation/right`.

```
results/
├── raw/                                # Trial-by-trial LLM responses
│   ├── top-functions/
│   │   └── {species}/{atlas}/{model}/{template}/{hemisphere}/
│   │       ├── cleaned/               # Parsed function lists
│   │       │   ├── trial_0/{region}.json
│   │       │   └── final/{region}.json
│   │       ├── embeddings/            # Per-region embedding CSVs
│   │       │   ├── trial_0/{region}.csv
│   │       │   └── final/
│   │       ├── justifications/        # (if --justify)
│   │       │   ├── trial_0/{region}.json
│   │       │   └── final/
│   │       └── retest_summary/        # (if --retest > 1)
│   │           └── {region}.json
│   ├── query-functions/
│   │   └── .../{trial_X}/{region}.json
│   └── rankings/
│       └── .../{trial_X}/{r1}_{r2}.json
│
├── aggregated/                         # Merged & post-processed results
│   ├── top-functions/
│   │   └── .../
│   │       ├── all_responses.json      # All functions per region
│   │       ├── all_embeddings.csv      # Regions x embedding dimensions
│   │       └── per_function_embeddings.csv
│   ├── probabilities/
│   │   └── .../
│   │       └── probability_distribution.csv  # Regions x functions
│   └── rankings/
│       └── .../
│           └── {r1}_vs_{r2}/results.csv
│
├── visualizations/                     # Plots and heatmaps
│   ├── similarities/
│   │   └── .../
│   │       ├── similarity_matrix.csv
│   │       └── similarity_matrix.png
│   ├── probabilities/
│   │   └── .../
│   │       └── heatmap.png
│   └── rankings/
│       └── .../
│
└── prompts/                            # Generated prompts (for reproducibility)
    └── {type}/{species}/{atlas}/{template}/{hemisphere}/
        ├── prompt_YYYYMMDD_HHMMSS.txt
        └── prompt_YYYYMMDD_HHMMSS_metadata.json
```

## Project Structure

```
clean_llm_prompting/
├── main.py                     # Entry point & command dispatch
├── argument_parser.py          # CLI argument parsing
├── functions.json              # Predefined functions & groups
├── .env                        # API keys
├── setup_environment.sh        # Environment setup script
│
├── utils/
│   ├── brain_analyser.py       # Main analysis orchestrator
│   ├── api_clients.py          # LLM client management (OpenRouter, BrainGPT, dummy)
│   ├── prompts.py              # Prompt generation & template loading
│   │
│   ├── core/                   # Core analysis logic
│   │   ├── task_processor.py   # Generic task loop (trials, skipping, retry)
│   │   ├── function_task.py    # top-functions workflow
│   │   ├── probability_task.py # query-functions workflow
│   │   ├── ranking_task.py     # rank-pairs workflow
│   │   ├── function_registry.py # Load functions & groups from functions.json
│   │   ├── response_cleaning.py # Extract values from LLM responses
│   │   ├── retest_averaging.py # Consensus, mean, mode aggregation
│   │   ├── aggregation.py      # Post-processing (JSON -> CSV)
│   │   └── visualisation.py    # Generate heatmaps & similarity matrices
│   │
│   ├── misc/                   # Utilities
│   │   ├── logging_setup.py    # Custom colored logging
│   │   ├── variables.py        # Default templates & config
│   │   ├── atlas.py            # Load & validate brain regions
│   │   ├── model_listing.py    # List OpenRouter models
│   │   └── query_saves.py      # Thread-safe JSON saving
│   │
│   └── paths/                  # Path constructors
│       ├── base.py             # Base path logic & defaults
│       ├── query.py            # Raw query result paths
│       ├── embeddings.py       # Embedding paths
│       ├── aggregation.py      # Aggregated result paths
│       ├── visualisation.py    # Visualization output paths
│       ├── prompts.py          # Prompt archive paths
│       └── atlas.py            # Atlas file paths
│
├── prompts/                    # Prompt templates
│   ├── functions/default.txt
│   ├── probabilities/default.txt
│   └── rankings/default.txt
│
├── atlases/                    # Brain region definitions
│   ├── human/
│   ├── macaque/
│   └── mouse/
│
└── results/                    # Output directory (auto-created)
```

## Troubleshooting

- Check `llm_prompting.log` for detailed execution logs
- Ensure API keys are properly set in `.env` file
- Verify atlas files exist in correct directory structure
- Use `--models dummy` for testing without API usage
- Use `list-models` to verify your OpenRouter API key works and see available models
- **BrainGPT not loading?** Make sure you've been granted access to `meta-llama/Llama-2-7b-chat-hf` on Hugging Face and that your `HF_TOKEN` has the correct permissions
- Check that function names in `--functions` match those in literature
- **Interrupted run?** Re-run the same command -- completed trials are automatically skipped
