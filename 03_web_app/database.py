"""
Database operations for the Medicinal Leaf Classifier web application.

This module handles all SQLite database interactions including connection
management, user registration, and authentication.

Imports from: config
Imported by: app
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DATABASE_PATH


def get_db_connection() -> Optional[sqlite3.Connection]:
    """
    Create and return a SQLite database connection.

    Automatically creates the 'users' table if it does not exist.

    Returns:
        sqlite3.Connection or None if connection fails.

    Raises:
        No exceptions are raised; errors are printed to stdout.
    """
    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                datetime TEXT NOT NULL
            )
        """)

        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection failed: {e}")
        return None


def create_user(username: str, email: str, password: str) -> bool:
    """
    Insert a new user into the database.

    Args:
        username: Unique username for the new user.
        email: Unique email address for the new user.
        password: Password for the new user.

    Returns:
        True if user was created successfully, False otherwise.

    Raises:
        No exceptions are raised; errors are printed to stdout.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return False

        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password, datetime) VALUES (?, ?, ?, ?)",
            (username, email, password, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Error creating user: {e}")
        return False


def validate_user(username: str, password: str) -> tuple[bool, str]:
    """
    Validate a user's credentials against the database.

    Args:
        username: Username to validate.
        password: Password to validate.

    Returns:
        A tuple of (is_valid, username).

    Raises:
        No exceptions are raised; errors are printed to stdout.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return False, username

        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and user["password"] == password:
            return True, username
        return False, username
    except sqlite3.Error as e:
        print(f"Database error during validation: {e}")
        return False, username


def get_user_id(username: str) -> Optional[int]:
    """
    Look up a user's ID by username.

    Args:
        username: Username to look up.

    Returns:
        The user's integer ID, or None if not found.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        return user["id"] if user else None
    except sqlite3.Error as e:
        print(f"Database error looking up user: {e}")
        return None


# ── Module self-test
# Run: python -m database
# Expected output: prints database path and connection status
if __name__ == "__main__":
    print(f"Database path: {DATABASE_PATH}")
    conn = get_db_connection()
    if conn:
        print("Connection successful")
        conn.close()
    else:
        print("Connection failed")
