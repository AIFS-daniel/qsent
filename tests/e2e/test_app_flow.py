"""
E2E tests for the main app flow after authentication.

Uses the /auth/test-login bypass endpoint (only available when TEST_MODE=true).

Requires server running: TEST_MODE=true uvicorn qsf.api.main:app --reload
"""
import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(autouse=True)
def authenticate(page: Page, base_url: str):
    """Log in via the test bypass before each test."""
    page.goto(f"{base_url}/auth/test-login")
    # Should redirect to / after setting the session cookie
    page.wait_for_url(base_url + "/", timeout=5000)


def test_authenticated_user_sees_app(page: Page, base_url: str):
    page.goto(base_url)
    expect(page.locator("h1")).to_contain_text("QSent")
    expect(page.locator("#analyzeBtn")).to_be_visible()


def test_analyze_renders_chart(page: Page, base_url: str):
    page.goto(base_url)
    page.fill("#ticker", "IONQ")
    page.click("#analyzeBtn")
    # Chart section appears after successful response
    expect(page.locator("#chartSection")).to_be_visible(timeout=30000)


def test_logout_redirects_to_login(page: Page, base_url: str):
    page.goto(f"{base_url}/auth/logout")
    page.wait_for_url(f"{base_url}/login.html", timeout=5000)
    expect(page).to_have_url(f"{base_url}/login.html")
