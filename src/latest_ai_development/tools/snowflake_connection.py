from __future__ import annotations

import os
import re
from typing import Any

import snowflake.connector
from dotenv import load_dotenv

from latest_ai_development.secrets_manager import SecretsManager

load_dotenv()

_secrets_manager: SecretsManager | None = None

_SECRET_VALUE_KEYS = {
    "PASSWORD",
    "PAT",
    "PAT_SECRET",
    "TOKEN",
    "ACCESS_TOKEN",
    "PRIVATE_KEY",
    "CONNECTION_STRING",
}


def _normalize_key(key: str) -> str:
    return str(key or "").strip().upper()


def _env_name(env_prefix: str, key: str) -> str:
    return f"{env_prefix}_{_normalize_key(key)}"


def _read_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _get_secrets_manager() -> SecretsManager:
    global _secrets_manager

    if _secrets_manager is None:
        _secrets_manager = SecretsManager()

    return _secrets_manager


def _resolve_value(
    env_prefix: str,
    key: str,
    defaults: dict[str, Any] | None = None,
    *,
    required: bool = True,
) -> str | None:
    normalized_key = _normalize_key(key)
    defaults = defaults or {}

    env_value = _read_env(_env_name(env_prefix, normalized_key))
    if env_value:
        return env_value

    default_value = defaults.get(normalized_key.lower())
    if default_value is None:
        default_value = defaults.get(normalized_key)

    if default_value is not None and str(default_value).strip():
        return str(default_value).strip()

    if required:
        raise RuntimeError(
            f"{_env_name(env_prefix, normalized_key)} is required for Snowflake connection. "
            "Set the tool-specific environment variable or provide it in the generated "
            "tool defaults."
        )

    return None


def _resolve_pat(env_prefix: str) -> str:
    pat_secret_env = _env_name(env_prefix, "PAT_SECRET")
    pat_secret = _read_env(pat_secret_env)

    if not pat_secret:
        raise RuntimeError(
            f"{pat_secret_env} is required for Snowflake connection. "
            "Set this tool-specific environment variable to the SecretsManager secret "
            "name for the Snowflake PAT."
        )

    pat = _get_secrets_manager().get_secret(pat_secret)

    if not pat or not str(pat).strip():
        raise RuntimeError(
            f"Snowflake PAT secret '{pat_secret}' could not be resolved. "
            f"Check {pat_secret_env}."
        )

    return str(pat).strip()


def _quote_identifier(identifier: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", identifier):
        return identifier
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _validate_no_secret_defaults(defaults: dict[str, Any] | None) -> None:
    for key, value in (defaults or {}).items():
        normalized_key = _normalize_key(key)
        if normalized_key in _SECRET_VALUE_KEYS and value:
            raise RuntimeError(
                f"Generated Snowflake defaults must not contain secret value key '{key}'. "
                "Set the tool-specific PAT_SECRET environment variable instead."
            )


class SnowflakeConnection:
    _connections: dict[str, snowflake.connector.SnowflakeConnection] = {}

    @classmethod
    def get_connection(
        cls,
        env_prefix: str,
        defaults: dict[str, Any] | None = None,
    ):
        if not env_prefix or not str(env_prefix).strip():
            raise RuntimeError("env_prefix is required for Snowflake connection.")

        env_prefix = str(env_prefix).strip().upper()
        _validate_no_secret_defaults(defaults)

        try:
            account = _resolve_value(env_prefix, "ACCOUNT", defaults)
            user = _resolve_value(env_prefix, "USER", defaults)
            password = _resolve_pat(env_prefix)
            host = _resolve_value(env_prefix, "HOST", defaults, required=False)
            warehouse = _resolve_value(env_prefix, "WAREHOUSE", defaults)
            database = _resolve_value(env_prefix, "DATABASE", defaults, required=False)
            schema = _resolve_value(env_prefix, "SCHEMA", defaults, required=False)

            connection = cls._connections.get(env_prefix)
            if connection is None or connection.is_closed():
                connect_args = {
                    "account": account,
                    "user": user,
                    "password": password,
                    "warehouse": warehouse,
                }

                if database:
                    connect_args["database"] = database
                if schema:
                    connect_args["schema"] = schema
                if host:
                    connect_args["host"] = host
                connection = snowflake.connector.connect(**connect_args)
                cls._connections[env_prefix] = connection

            with connection.cursor() as cursor:
                cursor.execute(f"USE WAREHOUSE {_quote_identifier(warehouse)}")

            return connection

        except snowflake.connector.errors.DatabaseError as exc:
            message = str(exc)
            if (
                "PAT" in message
                or "token" in message.lower()
                or "expired" in message.lower()
            ):
                raise RuntimeError(
                    "Snowflake PAT has expired or is invalid. "
                    f"Update the secret referenced by {_env_name(env_prefix, 'PAT_SECRET')}."
                ) from exc
            raise

    @classmethod
    def close(cls, env_prefix: str | None = None):
        if env_prefix:
            normalized_prefix = str(env_prefix).strip().upper()
            connection = cls._connections.pop(normalized_prefix, None)
            if connection and not connection.is_closed():
                connection.close()
            return

        for connection in cls._connections.values():
            if connection and not connection.is_closed():
                connection.close()
        cls._connections.clear()
