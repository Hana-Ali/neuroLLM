from utils.paths.base import BasePathConstructor, DEFAULT_PATHS


class AtlasPathConstructor(BasePathConstructor):
    """Handles atlas and species-related path construction"""

    @staticmethod
    def construct_atlas_path(species: str, atlas_name: str):
        """
        Construct path to atlas file

        Args:
            * species: Species used for the analysis
            * atlas_name: Name of the atlas

        Returns:
            * Path to the corresponding species atlas
        """
        return f"{DEFAULT_PATHS['atlas']}/{species}/{atlas_name}.csv"
