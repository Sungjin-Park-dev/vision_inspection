#!/usr/bin/env python3
"""
CLI Utilities for Consistent Output Formatting

Provides consistent formatting functions for command-line output across
all vision inspection pipeline scripts.
"""

from typing import Optional


def print_section_header(
    title: str,
    width: int = 70,
    char: str = '=',
    newline_before: bool = True
) -> None:
    """
    Print a formatted section header

    Args:
        title: Section title text
        width: Total width of the header (default: 70)
        char: Character to use for borders (default: '=')
        newline_before: Add newline before header (default: True)

    Example:
        >>> print_section_header("INITIALIZATION")

        ======================================================================
        INITIALIZATION
        ======================================================================
    """
    if newline_before:
        print()
    print(char * width)
    print(title)
    print(char * width)


def print_subsection(
    title: str,
    width: int = 70,
    char: str = '-',
    indent: int = 0
) -> None:
    """
    Print a formatted subsection header

    Args:
        title: Subsection title text
        width: Total width of the header (default: 70)
        char: Character to use for border (default: '-')
        indent: Number of spaces to indent (default: 0)

    Example:
        >>> print_subsection("Loading Data", indent=2)
          ----------------------------------------------------------------------
          Loading Data
          ----------------------------------------------------------------------
    """
    indent_str = ' ' * indent
    print(f"{indent_str}{char * width}")
    print(f"{indent_str}{title}")
    print(f"{indent_str}{char * width}")


def print_key_value(
    key: str,
    value: any,
    indent: int = 2,
    width: int = 35
) -> None:
    """
    Print a key-value pair with consistent formatting

    Args:
        key: Key name
        value: Value to print
        indent: Number of spaces to indent (default: 2)
        width: Width for key column (default: 35)

    Example:
        >>> print_key_value("Total waypoints", 150)
          Total waypoints:                   150
    """
    indent_str = ' ' * indent
    print(f"{indent_str}{key:<{width}}: {value}")


def print_success(message: str, indent: int = 0) -> None:
    """
    Print a success message with checkmark

    Args:
        message: Success message
        indent: Number of spaces to indent (default: 0)

    Example:
        >>> print_success("Trajectory saved")
        ✓ Trajectory saved
    """
    indent_str = ' ' * indent
    print(f"{indent_str}✓ {message}")


def print_warning(message: str, indent: int = 0) -> None:
    """
    Print a warning message

    Args:
        message: Warning message
        indent: Number of spaces to indent (default: 0)

    Example:
        >>> print_warning("Mesh coordinates may be incorrect")
        ⚠ WARNING: Mesh coordinates may be incorrect
    """
    indent_str = ' ' * indent
    print(f"{indent_str}⚠ WARNING: {message}")


def print_error(message: str, indent: int = 0) -> None:
    """
    Print an error message

    Args:
        message: Error message
        indent: Number of spaces to indent (default: 0)

    Example:
        >>> print_error("File not found")
        ✗ ERROR: File not found
    """
    indent_str = ' ' * indent
    print(f"{indent_str}✗ ERROR: {message}")


def print_progress(
    current: int,
    total: int,
    description: str = "",
    indent: int = 2
) -> None:
    """
    Print progress information

    Args:
        current: Current iteration number
        total: Total number of iterations
        description: Optional description
        indent: Number of spaces to indent (default: 2)

    Example:
        >>> print_progress(50, 100, "Processing waypoints")
          Processing waypoints: 50/100 (50.0%)
    """
    indent_str = ' ' * indent
    percentage = (current / total * 100) if total > 0 else 0
    if description:
        print(f"{indent_str}{description}: {current}/{total} ({percentage:.1f}%)")
    else:
        print(f"{indent_str}Progress: {current}/{total} ({percentage:.1f}%)")


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string

    Example:
        >>> format_time(125.5)
        '2m 5.5s'
        >>> format_time(3.14)
        '3.14s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def print_timing_summary(
    timings: dict,
    indent: int = 2,
    width: int = 40
) -> None:
    """
    Print a summary of timing information

    Args:
        timings: Dictionary of {operation_name: time_in_seconds}
        indent: Number of spaces to indent (default: 2)
        width: Width for operation name column (default: 40)

    Example:
        >>> print_timing_summary({
        ...     "Loading data": 1.5,
        ...     "Processing": 45.2,
        ...     "Saving": 0.8
        ... })
          Loading data:                           1.50s
          Processing:                            45.20s
          Saving:                                 0.80s
    """
    indent_str = ' ' * indent
    for operation, time_sec in timings.items():
        formatted_time = format_time(time_sec)
        print(f"{indent_str}{operation:<{width}}: {formatted_time:>10}")


def print_statistics(
    stats: dict,
    indent: int = 2,
    width: int = 40
) -> None:
    """
    Print statistics in a formatted table

    Args:
        stats: Dictionary of {stat_name: value}
        indent: Number of spaces to indent (default: 2)
        width: Width for stat name column (default: 40)

    Example:
        >>> print_statistics({
        ...     "Total waypoints": 150,
        ...     "Collision-free": 142,
        ...     "Success rate": "94.7%"
        ... })
          Total waypoints:                        150
          Collision-free:                         142
          Success rate:                         94.7%
    """
    indent_str = ' ' * indent
    for name, value in stats.items():
        print(f"{indent_str}{name:<{width}}: {value:>10}")
