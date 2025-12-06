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
