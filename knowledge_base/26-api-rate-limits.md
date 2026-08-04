# API rate limits

The Nimbus API allows 120 requests per minute per API key on the Team plan and 600 per minute on the Business plan. If you exceed the limit you get an HTTP 429 response with a Retry-After header telling you how many seconds to wait. Batch reads where possible and cache results that do not change often. Rate limits are per key, so spreading load across keys does not raise your total allowance.
