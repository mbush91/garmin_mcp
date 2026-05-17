import base64
import contextlib
import hashlib
import secrets
import time
from urllib.parse import urlencode

import httpx
import uvicorn
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Mount, Route


ACCESS_TOKEN_TTL_SECONDS = 3600
AUTH_CODE_TTL_SECONDS = 300


authorization_states = {}
authorization_codes = {}
access_tokens = {}
registered_clients = {}


class GarminOAuthTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        token_data = access_tokens.get(token)
        if not token_data:
            return None
        if token_data["expires_at"] < int(time.time()):
            access_tokens.pop(token, None)
            return None
        return AccessToken(
            token=token,
            client_id=token_data["client_id"],
            scopes=token_data["scopes"],
            expires_at=token_data["expires_at"],
            resource=token_data["resource"],
        )


def _base64url_sha256(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _json_error(error: str, description: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        {"error": error, "error_description": description},
        status_code=status_code,
    )


def _parse_csv(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


async def _request_data(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return await request.json()
    form = await request.form()
    if isinstance(form, FormData):
        return dict(form)
    return {}


def _resource_metadata(base_url: str) -> dict:
    return {
        "resource": f"{base_url}/mcp",
        "authorization_servers": [base_url],
        "scopes_supported": ["mcp"],
        "bearer_methods_supported": ["header"],
    }


def _authorization_server_metadata(base_url: str) -> dict:
    return {
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/authorize",
        "token_endpoint": f"{base_url}/token",
        "registration_endpoint": f"{base_url}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": ["mcp"],
    }


def _require_setting(settings: dict, key: str) -> str:
    value = settings.get(key, "").strip()
    if not value:
        raise RuntimeError(f"{key} is required when MCP_OAUTH_ENABLED=true")
    return value


def create_oauth_asgi_app(mcp, settings: dict) -> Starlette:
    public_base_url = _require_setting(settings, "PUBLIC_BASE_URL").rstrip("/")
    google_client_id = _require_setting(settings, "GOOGLE_CLIENT_ID")
    google_client_secret = _require_setting(settings, "GOOGLE_CLIENT_SECRET")
    allowed_emails = _parse_csv(_require_setting(settings, "GARMIN_MCP_ALLOWED_EMAILS"))
    google_callback_url = f"{public_base_url}/callback"

    async def protected_resource_metadata(request: Request) -> JSONResponse:
        return JSONResponse(_resource_metadata(public_base_url))

    async def authorization_server_metadata(request: Request) -> JSONResponse:
        return JSONResponse(_authorization_server_metadata(public_base_url))

    async def register(request: Request) -> JSONResponse:
        data = await _request_data(request)
        client_id = secrets.token_urlsafe(24)
        client = {
            "client_id": client_id,
            "client_id_issued_at": int(time.time()),
            "redirect_uris": data.get("redirect_uris", []),
            "client_name": data.get("client_name", "ChatGPT"),
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "scope": data.get("scope", "mcp"),
        }
        registered_clients[client_id] = client
        return JSONResponse(client, status_code=201)

    async def authorize(request: Request):
        params = request.query_params
        if params.get("response_type") != "code":
            return _json_error("unsupported_response_type", "Only authorization code flow is supported")
        client_id = params.get("client_id")
        redirect_uri = params.get("redirect_uri")
        code_challenge = params.get("code_challenge")
        code_challenge_method = params.get("code_challenge_method")
        if not client_id or not redirect_uri or not code_challenge:
            return _json_error("invalid_request", "client_id, redirect_uri, and code_challenge are required")
        if code_challenge_method != "S256":
            return _json_error("invalid_request", "Only S256 PKCE is supported")
        if client_id in registered_clients:
            redirect_uris = registered_clients[client_id].get("redirect_uris") or []
            if redirect_uris and redirect_uri not in redirect_uris:
                return _json_error("invalid_request", "redirect_uri is not registered for this client")
        oauth_state = secrets.token_urlsafe(32)
        authorization_states[oauth_state] = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": params.get("state"),
            "code_challenge": code_challenge,
            "resource": params.get("resource") or f"{public_base_url}/mcp",
            "scopes": [scope for scope in params.get("scope", "mcp").split() if scope] or ["mcp"],
            "expires_at": int(time.time()) + AUTH_CODE_TTL_SECONDS,
        }
        google_params = {
            "client_id": google_client_id,
            "redirect_uri": google_callback_url,
            "response_type": "code",
            "scope": "openid email profile",
            "state": oauth_state,
            "prompt": "select_account",
        }
        return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(google_params)}")

    async def callback(request: Request):
        params = request.query_params
        oauth_state = params.get("state")
        google_code = params.get("code")
        state_data = authorization_states.pop(oauth_state, None) if oauth_state else None
        if not google_code or not state_data:
            return _json_error("invalid_request", "Invalid or expired OAuth state")
        if state_data["expires_at"] < int(time.time()):
            return _json_error("invalid_request", "Expired OAuth state")
        async with httpx.AsyncClient(timeout=15) as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": google_client_id,
                    "client_secret": google_client_secret,
                    "code": google_code,
                    "grant_type": "authorization_code",
                    "redirect_uri": google_callback_url,
                },
            )
            if token_response.status_code >= 400:
                return _json_error("access_denied", "Google token exchange failed", 401)
            token_data = token_response.json()
            id_token = token_data.get("id_token")
            user_response = await client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": id_token},
            )
            if user_response.status_code >= 400:
                return _json_error("access_denied", "Google identity verification failed", 401)
            user_data = user_response.json()
        email = user_data.get("email", "").lower()
        if email not in allowed_emails:
            return _json_error("access_denied", "Email is not allowed", 403)
        auth_code = secrets.token_urlsafe(32)
        authorization_codes[auth_code] = {
            **state_data,
            "email": email,
            "expires_at": int(time.time()) + AUTH_CODE_TTL_SECONDS,
        }
        redirect_params = {"code": auth_code}
        if state_data.get("state"):
            redirect_params["state"] = state_data["state"]
        separator = "&" if "?" in state_data["redirect_uri"] else "?"
        return RedirectResponse(f"{state_data['redirect_uri']}{separator}{urlencode(redirect_params)}")

    async def token(request: Request) -> JSONResponse:
        data = await _request_data(request)
        if data.get("grant_type") != "authorization_code":
            return _json_error("unsupported_grant_type", "Only authorization_code is supported")
        auth_code = data.get("code")
        code_data = authorization_codes.pop(auth_code, None) if auth_code else None
        if not code_data:
            return _json_error("invalid_grant", "Invalid authorization code")
        if code_data["expires_at"] < int(time.time()):
            return _json_error("invalid_grant", "Expired authorization code")
        if data.get("client_id") != code_data["client_id"]:
            return _json_error("invalid_grant", "client_id does not match authorization code")
        if data.get("redirect_uri") != code_data["redirect_uri"]:
            return _json_error("invalid_grant", "redirect_uri does not match authorization code")
        code_verifier = data.get("code_verifier")
        if not code_verifier or _base64url_sha256(code_verifier) != code_data["code_challenge"]:
            return _json_error("invalid_grant", "PKCE verification failed")
        access_token = secrets.token_urlsafe(48)
        expires_at = int(time.time()) + ACCESS_TOKEN_TTL_SECONDS
        access_tokens[access_token] = {
            "client_id": code_data["client_id"],
            "email": code_data["email"],
            "scopes": code_data["scopes"],
            "resource": code_data["resource"],
            "expires_at": expires_at,
        }
        return JSONResponse(
            {
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": ACCESS_TOKEN_TTL_SECONDS,
                "scope": " ".join(code_data["scopes"]),
            }
        )

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        async with mcp.session_manager.run():
            yield

    return Starlette(
        routes=[
            Route("/.well-known/oauth-protected-resource", protected_resource_metadata, methods=["GET"]),
            Route("/.well-known/oauth-protected-resource/mcp", protected_resource_metadata, methods=["GET"]),
            Route("/.well-known/oauth-authorization-server", authorization_server_metadata, methods=["GET"]),
            Route("/register", register, methods=["POST"]),
            Route("/authorize", authorize, methods=["GET"]),
            Route("/callback", callback, methods=["GET"]),
            Route("/token", token, methods=["POST"]),
            Mount("/", app=mcp.streamable_http_app()),
        ],
        lifespan=lifespan,
    )


def oauth_auth_settings(public_base_url: str) -> AuthSettings:
    base_url = public_base_url.rstrip("/")
    return AuthSettings(
        issuer_url=AnyHttpUrl(base_url),
        resource_server_url=AnyHttpUrl(f"{base_url}/mcp"),
        required_scopes=["mcp"],
    )


def run_oauth_http_server(mcp, host: str, port: int, settings: dict) -> None:
    asgi_app = create_oauth_asgi_app(mcp, settings)
    uvicorn.run(asgi_app, host=host, port=port)
