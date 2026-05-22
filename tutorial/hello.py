import cimba


def message() -> str:
    return f"Hello world, I am Cimba {cimba.native_version()}"


def main() -> None:
    print(message())


if __name__ == "__main__":
    main()
