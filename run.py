# The single entry point for the swimming POC.
# As each pipeline stage is built (see CLAUDE.md section 17), it gets wired in here.

import argparse

from src.profiling import load_config

STAGES_READY = []  # names of pipeline stages that are wired up so far


def main():
    parser = argparse.ArgumentParser(
        description="Swimming performance tracking POC. See CLAUDE.md for the build order."
    )
    parser.add_argument("--video", help="Path to a video in data/raw/")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    load_config(args.config)  # fails loudly if config.yaml is missing or malformed

    if not STAGES_READY:
        print("No pipeline stages are wired up yet — see CLAUDE.md section 17 for the build order.")
        return


if __name__ == "__main__":
    main()
