def while_fac(a: int = 4) -> None:
    """Calculate factorial by while loop."""
    array_range = list(range(1, a + 1))
    idx = 1
    result = 1
    while idx < a:
        result *= array_range[idx]
        idx += 1
    print(result)


def main() -> None:
    """Run main function."""
    while_fac()


if __name__ == "__main__":
    main()
