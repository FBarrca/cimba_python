"""Verify that the Python wrapper links to the native Cimba library."""

import cimba


def main() -> None:
    print(f"Hello world, I am Cimba {cimba.native_version()}.")


if __name__ == "__main__":
    main()

