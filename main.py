"""
Simplified main script for brain region analysis

Works with the new BrainAnalyser class system
"""

import sys
from types import SimpleNamespace
from argument_parser import parse_args

from utils.brain_analyser import BrainAnalyser
from utils.api_clients import APIClientManager
from utils.core.function_processing import load_function_group, load_functions

from utils.misc.logging_setup import logger


def main():
    """Main LLM brain analysis script"""
    args = parse_args()

    # Determine models to use based on command
    models_input = "dummy" if args.command == "test" else args.models

    # Initialize API clients
    logger.info("Initializing API clients...")
    client_manager = APIClientManager(models=models_input)

    try:
        model_names, _ = client_manager.init_clients()
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
        if args.command == "functions":
            logger.info(f"Running functions analysis for {args.species}...")
            analyser = BrainAnalyser(config=config)
            analyser.analyze_functions()

        elif args.command == "probabilities":
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
                "commands: functions, probabilities, test"
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
