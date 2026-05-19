"""
E2E tests for the main app flow after authentication.

Uses the /auth/test-login bypass endpoint (only available when TEST_MODE=true).
API calls are intercepted with page.route() so no real external services are hit.

Requires server running: TEST_MODE=true uvicorn qsf.api.main:app --reload
"""
import json

import pytest
from playwright.sync_api import Page, expect

MOCK_ANALYZE_RESPONSE = {
    "ticker": "IONQ",
    "last_updated": "2026-01-10T00:00:00",
    "sentiment_score": 0.5,
    "data_points": 10,
    "breakdown": {"news_sentiment": 0.6, "social_sentiment": 0.4, "trend": "stable"},
    "daily_data": [
        {
            "date": "2026-01-10",
            "close": 48.0,
            "volume": 1000000,
            "ror": None,
            "news_sentiment": 0.5,
            "social_sentiment": 0.4,
        }
    ],
}


@pytest.fixture(autouse=True)
def authenticate(page: Page, base_url: str):
    """Log in via the test bypass before each test."""
    page.goto(f"{base_url}/auth/test-login")
    page.wait_for_url(base_url + "/", timeout=5000)


def test_authenticated_user_sees_app(page: Page, base_url: str):
    page.goto(base_url)
    expect(page.locator("h1")).to_contain_text("QSent")
    expect(page.locator("#analyzeBtn")).to_be_visible()


def test_analyze_renders_chart(page: Page, base_url: str):
    page.route(
        "**/analyze",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(MOCK_ANALYZE_RESPONSE),
        ),
    )
    page.goto(base_url)
    page.fill("#ticker", "IONQ")
    page.click("#analyzeBtn")
    expect(page.locator("#chartSection")).to_be_visible(timeout=5000)


def test_logout_redirects_to_login(page: Page, base_url: str):
    page.goto(f"{base_url}/auth/logout")
    page.wait_for_url(f"{base_url}/login.html", timeout=5000)
    expect(page).to_have_url(f"{base_url}/login.html")


def test_profile_button_is_visible(page: Page, base_url: str):
    page.goto(base_url)
    expect(page.locator("#profileBtn")).to_be_visible()


def test_profile_dropdown_opens_on_click(page: Page, base_url: str):
    page.goto(base_url)
    expect(page.locator("#profileDropdown")).not_to_be_visible()
    page.click("#profileBtn")
    expect(page.locator("#profileDropdown")).to_be_visible()


def test_profile_dropdown_shows_user_info(page: Page, base_url: str):
    page.goto(base_url)
    page.click("#profileBtn")
    expect(page.locator("#pdName")).to_have_text("Test User")
    expect(page.locator("#pdEmail")).to_have_text("test@example.com")


def test_profile_dropdown_closes_on_outside_click(page: Page, base_url: str):
    page.goto(base_url)
    page.click("#profileBtn")
    expect(page.locator("#profileDropdown")).to_be_visible()
    page.click("h1")
    expect(page.locator("#profileDropdown")).not_to_be_visible()


def test_signout_link_navigates_to_logout(page: Page, base_url: str):
    page.goto(base_url)
    page.click("#profileBtn")
    page.click(".pd-signout")
    page.wait_for_url(f"{base_url}/login.html", timeout=5000)
    expect(page).to_have_url(f"{base_url}/login.html")
