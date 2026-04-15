#!/usr/bin/env python3
import argparse
from pathlib import Path

import generate


INT_KEYS = {
    "count",
    "height",
    "max_sequence_length",
    "return_index",
    "seed",
    "steps",
    "width",
}
FLOAT_KEYS = {"guidance"}
STRING_KEYS = {"name", "negative", "output"}
BOOL_KEYS = {"randomize_seeds"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Nucleus Image FP8 runner")
    generate.add_generation_args(parser, require_prompt=False)
    return parser.parse_args()


def print_help() -> None:
    print("Commands:")
    print("  /help                     show this help")
    print("  /show                     show current generation settings")
    print("  /set KEY VALUE            update a setting, e.g. /set steps 20")
    print("  /random on|off            enable or disable random seeds")
    print("  /negative TEXT            set default negative prompt")
    print("  /negative off             clear the negative prompt")
    print("  /quit                     exit")
    print("Anything else is treated as a prompt and generated immediately.")


def print_settings(args: argparse.Namespace) -> None:
    print("Current settings:")
    print(f"  size={args.width}x{args.height}")
    print(f"  steps={args.steps} guidance={args.guidance}")
    print(f"  count={args.count} seed={args.seed} randomize_seeds={args.randomize_seeds}")
    print(f"  negative={args.negative!r}")
    print(f"  name={args.name!r} output={args.output!r}")
    print(f"  text_encoder_device={args.text_encoder_device} vae_device={args.vae_device}")


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def parse_value(key: str, value: str):
    if key in INT_KEYS:
        return int(value)
    if key in FLOAT_KEYS:
        return float(value)
    if key in BOOL_KEYS:
        return parse_bool(value)
    if key in STRING_KEYS:
        normalized = value.strip()
        if normalized.lower() in {"", "none", "off", "clear"}:
            return None
        return value
    raise KeyError(key)


def apply_set(args: argparse.Namespace, command: str) -> None:
    parts = command.split(maxsplit=2)
    if len(parts) < 3:
        print("Usage: /set KEY VALUE")
        return

    key, value = parts[1], parts[2]
    if not hasattr(args, key):
        print(f"Unknown setting: {key}")
        return

    try:
        parsed = parse_value(key, value)
    except KeyError:
        print(f"Setting is not editable here: {key}")
        return
    except ValueError as exc:
        print(str(exc))
        return

    setattr(args, key, parsed)
    print(f"Set {key}={getattr(args, key)!r}")


def apply_negative(args: argparse.Namespace, command: str) -> None:
    parts = command.split(maxsplit=1)
    if len(parts) == 1:
        print(f"negative={args.negative!r}")
        return

    value = parts[1].strip()
    if value.lower() in {"off", "none", "clear"}:
        args.negative = None
    else:
        args.negative = value
    print(f"negative={args.negative!r}")


def handle_command(args: argparse.Namespace, command: str) -> bool:
    if command in {"/quit", "/exit"}:
        return False
    if command == "/help":
        print_help()
        return True
    if command == "/show":
        print_settings(args)
        return True
    if command.startswith("/set "):
        apply_set(args, command)
        return True
    if command.startswith("/random "):
        parts = command.split(maxsplit=1)
        try:
            args.randomize_seeds = parse_bool(parts[1])
        except ValueError as exc:
            print(str(exc))
            return True
        print(f"randomize_seeds={args.randomize_seeds}")
        return True
    if command.startswith("/negative"):
        apply_negative(args, command)
        return True

    print("Unknown command. Use /help.")
    return True


def main() -> None:
    args = parse_args()
    pipe, execution_device = generate.load_pipeline(args)

    print()
    print("Interactive mode ready.")
    print("Type a prompt and press Enter to generate.")
    print("Use /help for commands.")
    print()

    while True:
        try:
            line = input("prompt> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue

        if not line:
            continue

        if line.startswith("/"):
            should_continue = handle_command(args, line)
            if not should_continue:
                break
            continue

        run_args = generate.clone_args(args, prompt=line)
        print()
        try:
            output_paths = generate.generate_images(pipe, execution_device, run_args)
        except KeyboardInterrupt:
            print()
            print("Generation interrupted.")
            continue

        if len(output_paths) == 1:
            print(f"Last output: {output_paths[0]}")
        else:
            print(f"Last output set: {Path(output_paths[0]).parent}")
        print()


if __name__ == "__main__":
    main()
