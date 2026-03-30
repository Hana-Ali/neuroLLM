import os
import json
from glob import glob
from typing import List

from utils.misc.logging_setup import logger
from utils.misc.variables import DEFAULT_TEMPLATES
from utils.paths.prompts import PromptPathConstructor


def create_default_templates():
    """Create default template files if they don't exist"""

    # Define two prompting types
    tasks = DEFAULT_TEMPLATES.keys()

    for task in tasks:

        # Ensure prompt directory exists
        default_path = PromptPathConstructor.construct_template_path(
            prompt_type=task, template_name="default"
        )
        os.makedirs(os.path.dirname(default_path), exist_ok=True)

        # Create default template file if missing
        if not os.path.exists(default_path):
            with open(default_path, "w") as f:
                f.write(DEFAULT_TEMPLATES[task])


def get_available_templates(prompt_type: str) -> List[str]:
    """
    Get list of available template names

    Args:
        * prompt_type: "function" or "probability"

    Returns:
        * List of template names (without .txt)
    """
    prompt_dir = PromptPathConstructor.get_prompt_dir(prompt_type=prompt_type)
    pattern = f"{prompt_dir}/*.txt"
    files = glob(pattern)
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def load_custom_template(prompt_type: str, template_name: str) -> str:
    """
    Load a template from file or return default

    Args:
        * prompt_type: Type of analysis ("functions" or "probabilities")
        * template_name: Name of the template file (without .txt)

    Returns:
        * Template string
    """
    create_default_templates()  # Ensure defaults exist

    # Construct template path and ensure it exists
    template_path = PromptPathConstructor.construct_template_path(
        prompt_type=prompt_type, template_name=template_name
    )
    if not os.path.exists(template_path):
        logger.error(
            f"Template {template_name} for {prompt_type} not found",
            exc_info=True,
        )
        raise

    # Load and return template content
    with open(template_path, "r") as f:
        return f.read()


def generate_prompt(
    prompt_type: str,
    region_name: str,
    species: str,
    atlas_name: str,
    hemisphere: str = None,
    function: str = None,
    template_name: str = "default",
    save_to_results: bool = False,
) -> str:
    """
    Generate analysis prompt for functions or probabilities

    Args:
        * prompt_type: "function" or "probability"
        * species: Target species
        * region_name: Brain region name
        * hemisphere: Hemisphere ("left"/"right"/None)
        * function: Function name (required for probability prompts)
        * template_name: Template name to use
        * atlas_name: Atlas name (for saving)
        * save_to_results: Whether to save prompt

    Returns:
        * Generated prompt string
    """
    # Load template
    template = load_custom_template(
        prompt_type=prompt_type, template_name=template_name
    )

    # Prepare variables for formatting
    format_vars = {
        "species": species,
        "region": region_name,
    }

    # Handle hemisphere part - includes the "in the **X** of the" phrase
    format_vars["hemisphere_part"] = (
        f"in the **{hemisphere} hemisphere** of the"
        if hemisphere
        else "in the"
    )

    # Add function for probability prompts
    if prompt_type == "probabilities":
        assert function, "Function must be provided for probability prompts"
        format_vars["function"] = function

    # Format template with all replacements at once
    prompt = template.format(**format_vars)

    # Save if requested
    if save_to_results:
        save_generated_prompt(
            prompt=prompt,
            prompt_type=prompt_type,
            species=species,
            region=region_name,
            hemisphere=hemisphere,
            atlas_name=atlas_name,
            template_name=template_name,
            function=function,
        )

    return prompt


def save_generated_prompt(
    prompt: str,
    prompt_type: str,
    species: str,
    region: str,
    hemisphere: str,
    atlas_name: str,
    template_name: str,
    function: str = None,
):
    """
    Save generated prompt to results

    Args:
        * prompt: Generated prompt string
        * prompt_type: "function" or "probability"
        * species: Species name
        * region: Brain region name
        * hemisphere: Hemisphere ("left"/"right"/None)
        * atlas_name: Atlas name
        * template_name: Template name used
        * function: Function name (for probability prompts)
    """
    # Construct the results prompt path
    results_prompt_path = PromptPathConstructor.construct_results_prompt_path(
        prompt_type=prompt_type,
        species=species,
        atlas_name=atlas_name,
        hemisphere=hemisphere if hemisphere else "no_separation",
        template_name=template_name,
    )

    # Ensure directory exists
    results_dirname = os.path.dirname(results_prompt_path)
    os.makedirs(results_dirname, exist_ok=True)

    # Check if we already have a prompt saved (avoid duplicates)
    if glob(f"{results_dirname}/prompt_*.txt"):
        # logger.processing(
        #     f"Prompt already exists in {results_dirname}, skipping save"
        # )
        return

    # Save new prompt
    with open(results_prompt_path, "w") as f:
        f.write(prompt)

    # Save metadata
    results_timestamp = results_prompt_path.split("prompt_")[-1].split(".txt")[
        0
    ]
    metadata = {
        "template": template_name,
        "species": species,
        "region": region,
        "hemisphere": hemisphere,
        "atlas": atlas_name,
        "function": function,
        "timestamp": results_timestamp,
    }

    # Save metadata alongside prompt
    results_without_ext = os.path.splitext(results_prompt_path)[0]
    with open(f"{results_without_ext}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
