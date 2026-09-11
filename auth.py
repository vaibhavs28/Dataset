import urllib.parse
import requests
import logging
from pathlib import Path
import config

logger = logging.getLogger("auth")


def get_login_url() -> str:
    """Generates the Upstox OAuth login URL."""
    if not config.UPSTOX_API_KEY:
        raise ValueError("UPSTOX_API_KEY is missing. Please set it in .env or config.py.")
    
    params = {
        "response_type": "code",
        "client_id": config.UPSTOX_API_KEY,
        "redirect_uri": config.UPSTOX_REDIRECT_URI
    }
    return f"{config.UPSTOX_AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code_for_token(auth_code: str) -> str:
    """
    Exchanges the authorization code for an Upstox access token
    and saves it to the local .env file.
    """
    if not config.UPSTOX_API_KEY or not config.UPSTOX_API_SECRET:
        raise ValueError("UPSTOX_API_KEY or UPSTOX_API_SECRET is not configured.")

    payload = {
        "code": auth_code,
        "client_id": config.UPSTOX_API_KEY,
        "client_secret": config.UPSTOX_API_SECRET,
        "redirect_uri": config.UPSTOX_REDIRECT_URI,
        "grant_type": "authorization_code"
    }

    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(config.UPSTOX_TOKEN_URL, data=payload, headers=headers, timeout=20)
    response_data = response.json()

    if response.status_code == 200 and "access_token" in response_data:
        access_token = response_data["access_token"]
        save_access_token(access_token)
        return access_token
    else:
        error_msg = response_data.get("errors", [{}])[0].get("message", response.text)
        raise RuntimeError(f"Failed to obtain token from Upstox: {error_msg}")


def save_access_token(token: str):
    """Saves the access token into the .env file."""
    env_file = config.ENV_PATH
    lines = []
    token_updated = False

    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.startswith("UPSTOX_ACCESS_TOKEN="):
                    lines.append(f"UPSTOX_ACCESS_TOKEN={token.strip()}\n")
                    token_updated = True
                else:
                    lines.append(line)

    if not token_updated:
        lines.append(f"UPSTOX_ACCESS_TOKEN={token.strip()}\n")

    with open(env_file, "w") as f:
        f.writelines(lines)

    config.UPSTOX_ACCESS_TOKEN = token.strip()
    logger.info("Upstox Access Token saved successfully.")


def save_credentials(api_key: str, api_secret: str, redirect_uri: str):
    """Saves API Key and Secret into .env."""
    env_file = config.ENV_PATH
    existing_vars = {}
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing_vars[k.strip()] = v.strip()

    existing_vars["UPSTOX_API_KEY"] = api_key.strip()
    existing_vars["UPSTOX_API_SECRET"] = api_secret.strip()
    existing_vars["UPSTOX_REDIRECT_URI"] = redirect_uri.strip()

    with open(env_file, "w") as f:
        for k, v in existing_vars.items():
            f.write(f"{k}={v}\n")

    config.UPSTOX_API_KEY = api_key.strip()
    config.UPSTOX_API_SECRET = api_secret.strip()
    config.UPSTOX_REDIRECT_URI = redirect_uri.strip()
