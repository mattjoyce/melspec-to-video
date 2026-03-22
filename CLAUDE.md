# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- **Virtual Environment**: `~/Environments/melspec-to-video/.venv/`
- **Setup Environment**: `~/Environments/melspec-to-video/.venv/bin/pip install -r requirements.txt`
- **Run Program**: `~/Environments/melspec-to-video/.venv/bin/python melspec_to_video.py --config config5.yaml [COMMAND] [OPTIONS]`
- **Main Commands**:
  - `profile`: Generate audio profile (`--input`, `--path`, `--project_file`, `--sr`, `--start`, `--duration`)
  - `spectrograms`: Create spectrogram images (`--path`, `--project_file`, `--sr`, `--start`, `--duration`, `--overwrite`)
  - `mp4`: Create video from spectrograms (`--path`, `--project_file`, `--output`, `--start`, `--duration`, `--cpu`, `--include_audio`)
  - `info`: Display configuration information

## Code Style Guidelines
- **Imports**: Group standard lib, third-party, then local imports
- **Formatting**: PEP 8 guidelines, 4-space indentation
- **Types**: Use type hints from `typing` module for function parameters and returns
- **Naming**: Snake_case for functions/variables, CamelCase for classes
- **Error Handling**: Use proper exception handling with specific exception types
- **Documentation**: Docstrings for all functions following Google-style format
- **Config**: Use YAML for configuration files, accessed via the `Params` classes

Always think and make a plan before coding.  Share the plan and ask first.
Check the todo.txt for priority
