import os
import pandas as pd
from typing import List, Dict

from utils.misc.logging_setup import logger
from utils.paths.atlas import AtlasPathConstructor


def get_species_atlas(
    atlas_dir: str, atlas_name: str = None
) -> Dict[str, List[str]]:
    """
    Load atlas data for all available species

    Args:
        atlas_dir: Directory containing species subdirectories
        atlas_name: Optional atlas name

    Returns:
        Dictionary mapping species names to lists of region names
    """
    species_atlas = {}
    try:
        # Get all subdirectories in atlas_dir
        # (these are the species directories)
        species_dirs = [
            d
            for d in os.listdir(atlas_dir)
            if os.path.isdir(os.path.join(atlas_dir, d))
        ]

        for species in species_dirs:
            species_dir_path = os.path.join(atlas_dir, species)

            # If atlas_name specified, look for that specific atlas file
            if atlas_name:
                atlas_path = os.path.join(
                    species_dir_path, f"{atlas_name}.csv"
                )
                if os.path.exists(atlas_path):
                    species_atlas[species] = load_clean_regions(
                        species, atlas_path
                    )
                    logger.processing(
                        f"Loaded {len(species_atlas[species])} "
                        f"regions for {species} using {atlas_name} atlas"
                    )
            # Otherwise, load all available atlas files for this species
            else:
                for file in os.listdir(species_dir_path):
                    if file.endswith(".csv"):
                        atlas_name = file.split(".")[0]
                        atlas_path = os.path.join(species_dir_path, file)
                        species_atlas[species] = load_clean_regions(
                            species, atlas_path
                        )
                        logger.processing(
                            f"Loaded {len(species_atlas[species])} "
                            f"regions for {species} from {atlas_name} atlas"
                        )
    except Exception as e:
        logger.error_status(
            f"Error loading species atlases: {str(e)}", exc_info=True
        )
        raise

    return species_atlas


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

        # Special handling for human atlas
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
    Load regions for a species, optionally filtering to a specific region

    Args:
        * species: Species name
        * atlas_name: Optional atlas subfolder to use

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
