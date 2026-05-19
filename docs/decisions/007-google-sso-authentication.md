# ADR 007: Google SSO via OAuth 2.0 Authorization Code Flow

**Date:** 2026-05-18
**Status:** Accepted

## Context

The app needed authentication before exposing the `/analyze` endpoint publicly. Three broad approaches were on the table: hand-roll username/password auth, use a third-party auth service (Auth0, Clerk), or delegate entirely to an existing identity provider via OAuth/OIDC.

The user base is a small internal team, all of whom already have Google accounts. There is no requirement to support non-Google sign-in.

## Decision

Implement Google SSO using the OAuth 2.0 Authorization Code flow with PKCE-equivalent CSRF protection (state parameter). Sessions are stored in an HttpOnly, SameSite=Lax JWT cookie issued by the app, not in a server-side session store.

The implementation is in `src/qsf/api/auth.py` and uses:
- **Authlib** for the OAuth 2.0 client (handles authorization URL construction, token exchange)
- **python-jose** for JWT signing and verification
- **Google's `/userinfo` endpoint** to retrieve email, display name, profile picture, and `sub` after token exchange

The flow:
1. `GET /auth/login` — generates a random `state` token, stores it in a short-lived HttpOnly cookie, redirects to Google's authorization URL
2. Google redirects back to `GET /auth/callback` with an authorization code and the `state` value
3. The callback validates the `state` against the cookie (CSRF protection), exchanges the code for an access token, fetches userinfo, and issues a signed JWT session cookie (`qsent_session`)
4. All subsequent requests authenticate by verifying the JWT cookie in `get_current_user()`

Session cookies are `httponly=True` (no JS access), `samesite="lax"` (CSRF mitigation), and `secure=True` in production (set via `SECURE_COOKIES=true`). Sessions expire after 8 hours.

The Google-issued `sub` field is stored in the JWT alongside email, so it is available for PII-free logging and tracing (see ADR 006).

A `TEST_MODE=true` bypass route is registered only when that env var is set, and issues a session with a fixed fake user. It is never reachable in production.

## Alternatives Considered

**Hand-rolled username/password auth**

Rejected. Building password auth correctly requires: secure password hashing (bcrypt/argon2), email verification flows, password reset flows, brute-force rate limiting, breach detection, and ongoing maintenance as attack patterns evolve. Each of these is a meaningful surface area for error. Teams that hand-roll this regularly get it wrong in ways that are only discovered after a breach. The effort is not proportional to the benefit when the user base already has Google accounts.

**Auth0 / Clerk / similar managed auth**

Rejected for now. These services are excellent but add a vendor dependency, a paid tier at scale, and a separate admin UI. For a small internal tool, the overhead is not justified. The current implementation handles everything Auth0 would handle for this use case (SSO, session management, user identity) with less moving parts.

**Server-side sessions (database-backed)**

Rejected. Storing session state in a database requires a sessions table, expiry cleanup jobs, and a database query on every authenticated request. A signed JWT cookie is stateless — the server verifies the signature and expiry without any I/O. The tradeoff is that individual sessions cannot be revoked before expiry, which is acceptable for an 8-hour session window on an internal tool.

**OAuth implicit flow**

Rejected. The implicit flow returns tokens directly in the URL fragment, which exposes them to browser history and referrer headers. It is deprecated in OAuth 2.1. The Authorization Code flow keeps tokens out of the URL entirely.

## Consequences

**Security properties:**
- Credential management (passwords, MFA, breach detection) is fully delegated to Google
- CSRF protection via the `state` parameter round-trip
- XSS cannot steal the session cookie (`httponly=True`)
- Avatar URLs are validated against a `googleusercontent.com` allowlist before being stored in the JWT, preventing open redirects or content injection via a malicious picture URL

**Operational notes:**
- Requires a Google Cloud project with an OAuth 2.0 client configured. The authorized redirect URI must be set to `{BASE_URL}/auth/callback`
- Any Google account can sign in by default — no allowlist. If the team is comfortable with this, no further config is needed. To restrict to a specific domain, add a domain check in the callback against `userinfo["hd"]` (Google's hosted domain claim)
- Sessions cannot be forcibly revoked before the 8-hour expiry. If a session needs to be invalidated immediately (e.g. a stolen laptop), rotate `SECRET_KEY` in the environment — this invalidates all active sessions simultaneously
- The `TEST_MODE` bypass must never be enabled in a production environment. It issues a valid session for a hardcoded fake user with no authentication check
