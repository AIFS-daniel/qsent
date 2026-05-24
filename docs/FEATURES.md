# QSent Features

> Auto-generated from the integration and e2e test suite. Each section reflects a product feature area; each item is a verified behavior.

## App Experience

- Authenticated user sees app
- Analyze renders chart
- Logout redirects to login
- Profile button is visible
- Profile dropdown opens on click
- Profile dropdown shows user info
- Profile dropdown closes on outside click
- Signout link navigates to logout

## Login

- Login page shows google button
- Unauthenticated visit redirects to login
- Google button points to auth login

## Sentiment Analysis

- Health
- Analyze valid ticker
- Analyze returns 404 on error
- Analyze missing ticker
- Analyze ticker is uppercased

## Authentication

- Analyze requires auth
- News comparison requires auth
- Health no auth required
- Stream no auth required
- Analyze succeeds with valid cookie
- Avatar requires auth
- Avatar returns 404 when no picture
- Avatar returns 404 when picture key missing
- Avatar returns 404 on failed fetch
- Avatar returns 200 with image bytes
- Avatar defaults to jpeg when no content type
- Avatar follows redirects

## News Comparison

- Valid request returns 200 with correct shape
- Tickers are uppercased
- Whitespace tickers filtered out
- All empty tickers returns 422
- Missing tickers field returns 422
- Provider error returns 200 with error in payload
- Stream returns text event stream
- Stream emits all expected event types
- Stream ticker start payload
- Stream article result has required fields
- Stream done event is last
- Stream empty tickers returns 422
- Stream tickers are uppercased

## Analysis Pipeline

- Happy path produces result
- Invalid ticker sets error
- No text data sets error
- Ticker is available throughout pipeline
