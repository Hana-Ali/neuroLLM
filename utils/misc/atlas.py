import os
import sys
import pandas as pd
from typing import List

from utils.misc.logging_setup import logger
from utils.paths.atlas import AtlasPathConstructor


def load_clean_regions(species: str, atlas_path: str) -> List[str]:
    """
    Load and clean region names from an atlas file

    Args:
        * species: Species name (human, macaque, mouse)
        * atlas_path: Path to the atlas CSV file

    Returns:
        List of cleaned region names
    """
    try:
        df = pd.read_csv(atlas_path, delimiter=",", header=None)
        df[0] = df[0].str.replace(r"^[^_]+_[^_]+_", "", regex=True)
        df[0] = df[0].str.replace(r"'", "", regex=True)
        df[0] = df[0].str.lower()

        # "bankssts" is a Desikan-Killiany atlas abbreviation;
        # expand for LLM comprehension
        if species.lower() == "human":
            df[0] = df[0].str.replace(
                "bankssts", "banks of the superior temporal sulcus"
            )

        regions = df[0].to_list()
        return regions
    except Exception as e:
        logger.error_status(
            f"Error loading regions from {atlas_path}: {str(e)}", exc_info=True
        )
        raise


def load_regions_for_species(
    species: str,
    atlas_name: str,
) -> List[str]:
    """
    Load regions for a species from an atlas file

    Args:
        * species: Species name
        * atlas_name: Atlas name (filename without extension)

    Returns:
        * List of regions
    """
    # Construct the atlas path
    atlas_path = AtlasPathConstructor.construct_atlas_path(
        species=species, atlas_name=atlas_name
    )

    if not os.path.exists(atlas_path):
        logger.error_status(
            f"Atlas file not found: {atlas_path}", exc_info=True
        )
        raise

    # Load all regions from the atlas
    return load_clean_regions(species=species, atlas_path=atlas_path)


def validate_analysis_inputs(args):
    """
    Validate that --species and sufficient region/atlas information is
    provided for the given command. Species is always required since atlas
    names are not unique across species.

    rank-pairs requires --species and one of:
        * --atlas-name  (all region combinations used)
        * --regions     (explicit regions, all combinations used)
        * --pairs       (explicit pairs)

    All other commands require --species and one of:
        * --atlas-name
        * --regions

    Args:
        * args: Parsed command-line arguments

    Raises:
        * SystemExit if validation fails
    """
    if not args.species:
        logger.error_status("--species is required")
        sys.exit(1)

    if args.command == "rank-pairs":
        if not (args.atlas_name or args.regions or args.pairs):
            logger.error_status(
                "For rank-pairs: provide --atlas-name, --regions, or --pairs"
            )
            sys.exit(1)
    else:
        if not args.atlas_name and not args.regions:
            logger.error_status(
                "Either --atlas-name or --regions must be provided"
            )
            sys.exit(1)
