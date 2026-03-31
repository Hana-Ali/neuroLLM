import sys
import random
from types import SimpleNamespace
from argument_parser import parse_args

import numpy as np
import torch

from dotenv import load_dotenv

from utils.brain_analyser import BrainAnalyser
from utils.api_clients import APIClientManager
from utils.misc.model_listing import list_available_models
from utils.core.function_processing import load_function_group, load_functions
from utils.paths.base import DEFAULT_PATHS

from utils.misc.logging_setup import logger


def main():
    """Main LLM brain analysis script"""
    # Set global seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load environment variables from .env file
    load_dotenv(DEFAULT_PATHS["env_file"])

    # Parse command-line arguments
    args = parse_args()

    # Handle list-models command early exit
    if args.command == "list-models":
        list_available_models(filter=args.filter)
        sys.exit(0)

    # Determine models to use based on command
    models_input = "dummy" if args.command == "test" else args.models

    # Initialize API clients
    logger.info("Initializing API clients...")
    try:
        client_manager = APIClientManager(
            models=models_input,
            embedding_provider=(
                args.embedding_provider
                if args.command == "top-functions"
                else None
            ),
            max_tokens=args.max_tokens,
        )
        model_names = client_manager.model_names
        logger.info(f"Using models: {', '.join(model_names)}")
    except Exception as e:
        logger.error_status(
            f"Failed to initialize API clients: {e}", exc_info=True
        )
        sys.exit(1)

    # Create base config as SimpleNamespace
    config = SimpleNamespace(
        species=args.species,
        regions=args.regions.split(",") if args.regions else None,
        models=model_names,
        functions=None,  # To be set later for probabilities
        workers=1 if args.command == "test" else args.workers,
        skip_visualization=args.skip_visualization,
        skip_raw_saving=args.skip_raw_saving,
        atlas_name=args.atlas_name,
        separate_hemispheres=args.separate_hemispheres,
        prompt_template_name=args.prompt_template_name,
        client_manager=client_manager,
    )

    # Run the appropriate command
    try:
        if args.command == "top-functions":
            logger.info(f"Running functions analysis for {args.species}...")
            analyser = BrainAnalyser(config=config)
            analyser.analyze_functions()

        elif args.command == "query-functions":
            # Get functions for probability analysis
            if args.function_group:
                functions = load_function_group(group_name=args.function_group)
                if not functions:
                    logger.error_status(
                        f"Function group '{args.function_group}' not found."
                    )
                    sys.exit(1)
                logger.info(
                    f"Using functions from group '{args.function_group}': "
                    f"{', '.join(functions)}"
                )
            elif args.functions:
                functions = [f.strip() for f in args.functions.split(",")]
                logger.info(
                    f"Using specified functions: {', '.join(functions)}"
                )
            else:
                functions, _ = load_functions()  # Default function set
                logger.info(f"Using default functions: {', '.join(functions)}")

            config.functions = functions

            logger.info(
                f"Running probabilities analysis for {args.species}..."
            )
            analyser = BrainAnalyser(config=config)
            analyser.analyze_probabilities()

        elif args.command == "test":
            logger.info("Running test workflow...")
            analyser1 = BrainAnalyser(config=config)

            logger.info("Testing functions analysis...")
            analyser1.analyze_functions()
            logger.success("Functions test completed")

            # Test probability analysis with dummy model
            logger.info("Testing probabilities analysis...")
            config.functions = ["spatial cognition", "consciousness"]

            analyser2 = BrainAnalyser(config=config)
            analyser2.analyze_probabilities()
            logger.success("Probabilities test completed")

        else:
            logger.error_status(
                f"Unknown command: {args.command}, please use from available "
                "commands: list-models, top-functions, query-functions, test"
            )
            sys.exit(1)

        logger.success("Analysis completed successfully!")

    except KeyboardInterrupt:
        logger.error("\nAnalysis interrupted by user", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error_status(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
