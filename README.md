# LLM Brain Analysis

A framework for using Large Language Models to analyze brain region functions across different species.

## Overview

This tool provides two primary analysis workflows:

1. **Functions Analysis**: Identifies the top 5 functions associated with brain regions and creates similarity matrices using embeddings
2. **Probabilities Analysis**: Calculates the probability of specific functions being associated with brain regions

The framework supports multiple LLM models across different providers:
- **OpenAI**: GPT-4o-mini
- **Anthropic**: Claude 3.7 Sonnet
- **Google**: Gemini 2.0 Flash  
- **TogetherAI**: Qwen2.5, Mistral, Llama 3.3, DeepSeek
- **Testing**: Dummy model for development

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

2. Create and activate conda environment:
   ```bash
   conda create -n neuroLLM python=3.12
   conda activate neuroLLM
   ```

3. Install dependencies:
   ```bash
   conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn
   pip install openai anthropic google-genai together python-dotenv
   ```

4. Set up API keys by creating a `.env` file:
   ```
   OPENAI_API_KEY=your-openai-key
   CLAUDE_API_KEY=your-claude-key
   GEMINI_API_KEY=your-gemini-key
   TOGETHERAI_API_KEY=your-togetherai-key
   ```

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

Each CSV should contain region names in the first column

## Usage

The tool provides a command-line interface with three main commands:

```bash
python main.py {functions|probabilities|test} [OPTIONS]
```

### Commands

#### Functions Analysis
Identifies top 5 functions for brain regions and creates similarity matrices:

```bash
python main.py functions [OPTIONS]
```

#### Probabilities Analysis  
Calculates probabilities of specific functions being associated with regions:

```bash
python main.py probabilities [OPTIONS]
```

#### Test Workflow
Runs a quick test of both analysis types:

```bash
python main.py test [OPTIONS]
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--species` | Target species: `human`, `macaque`, `mouse` | `human` |
| `--atlas-name` | Atlas to use (must exist in atlases/{species}/) | Required |
| `--models` | Model selection: `paid`, `all`, `dummy`, or comma-separated names | `dummy` |
| `--regions` | Comma-separated region names, or leave empty for all regions | All regions |
| `--separate-hemispheres` | Analyze left/right hemispheres separately | `False` |
| `--prompt-template-name` | Custom prompt template name | `default` |
| `--workers` | Number of parallel workers | `4` |
| `--skip-visualization` | Skip creating visualizations | `False` |
| `--skip-raw-saving` | Clean up raw data files after processing | `False` |

### Probabilities-Specific Options

| Option | Description |
|--------|-------------|
| `--functions` | Comma-separated function names to analyze |
| `--function-group` | Use predefined function group from functions.json |

## Examples

### Basic Usage

```bash
# Analyze functions for human brain using dummy model
python main.py functions --atlas-name DesikanKilliany68

# Analyze probabilities for specific functions
python main.py probabilities --atlas-name DesikanKilliany68 --functions "spatial cognition,memory,attention"

# Run test workflow
python main.py test --atlas-name DesikanKilliany68
```

### Advanced Usage

```bash
# Use paid models with hemisphere separation
python main.py functions --atlas-name DesikanKilliany68 --models paid --separate-hemispheres --workers 8

# Analyze specific regions only
python main.py functions --atlas-name DesikanKilliany68 --regions "hippocampus,amygdala,prefrontal cortex"

# Use function groups for probabilities
python main.py probabilities --atlas-name DesikanKilliany68 --function-group memory

# Use specific models
python main.py functions --atlas-name DesikanKilliany68 --models "openai,claude"
```

## Model Categories

Models are organized by access type:

- **`paid`**: OpenAI GPT-4o-mini, Claude 3.7 Sonnet, Gemini 2.0 Flash, Qwen2.5, Mistral
- **`dummy`**: Test model (no API usage)
- **`all`**: All models including dummy

You can also specify individual models: `"openai,claude,gemini"`

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

- `prompts/functions/custom_template.txt` - For function analysis
- `prompts/probabilities/custom_template.txt` - For probability analysis

Templates support variables:
- `{species}` - Target species
- `{region}` - Brain region name  
- `{hemisphere_part}` - Hemisphere phrase ("in the left hemisphere of the" or "in the")
- `{function}` - Function name (probabilities only)

Use with `--prompt-template-name custom_template`

## Output Structure

Results are organized in the `results/` directory:

```
results/
├── prompts/           # Generated prompts used
├── raw/              # Raw LLM responses  
├── embeddings/       # Vector embeddings
├── aggregated/       # Processed results
│   ├── functions/    # Function lists and similarity matrices
│   └── probabilities/ # Probability distributions
└── visualizations/   # Plots and heatmaps
    ├── similarities/ # Similarity matrix plots
    └── probabilities/ # Probability heatmaps
```

Followed by `<species>/<atlas_name>/<model_name>/<prompt_template_name>/<separate_hemispheres>` subdirectories

## Troubleshooting

- Check `llm_prompting.log` for detailed execution logs
- Ensure API keys are properly set in `.env` file
- Verify atlas files exist in correct directory structure
- Use `--models dummy` for testing without API usage
- Check that function names in `--functions` match those in literature