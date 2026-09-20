"""Shared Bedrock runtime client factory.

Drop-in from the reference snippet. Kept as its own module so any script in
this repo can share a single, consistent client configuration.
"""

import os
import subprocess
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "us.anthropic.claude-opus-5"
DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
MAX_TOKENS = 16000


def get_region() -> str:
    """Pick the AWS region from env vars, falling back to the default."""
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_REGION
    )


def get_model_id() -> str:
    """Return the Claude model id to use for Bedrock Converse calls."""
    return os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID


def get_embedding_model_id() -> str:
    """Return the Titan/embedding model id used to vectorise text."""
    return os.environ.get("BEDROCK_EMBEDDING_MODEL_ID") or DEFAULT_EMBEDDING_MODEL_ID


def make_client():
    """Build a fresh Bedrock runtime boto3 client for the resolved region."""
    return boto3.client("bedrock-runtime", region_name=get_region())


def refresh_credentials() -> bool:
    """Attempt SSO login to refresh expired credentials. Returns True if successful."""
    profile = os.environ.get("AWS_PROFILE") or os.environ.get("AWS_DEFAULT_PROFILE")
    cmd = ["aws", "sso", "login"]
    if profile:
        cmd += ["--profile", profile]
    print(f"[Auth] Credentials expired. Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def is_expired_token_error(ex: Exception) -> bool:
    """True when the exception looks like AWS SSO/token expiry."""
    if isinstance(ex, ClientError):
        return ex.response["Error"]["Code"] in ("ExpiredTokenException", "ExpiredToken")
    return isinstance(ex, NoCredentialsError) or \
        type(ex).__name__ in ("SSOTokenLoadError", "TokenRetrievalError", "UnauthorizedSSOTokenError")


def invoke_with_refresh(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), refreshing credentials once if token is expired."""
    try:
        return fn(*args, **kwargs)
    except Exception as ex:
        if not is_expired_token_error(ex):
            raise
        if not refresh_credentials():
            print("[Auth] Could not refresh credentials. Please re-authenticate manually.")
            sys.exit(1)
        return fn(*args, **kwargs)
