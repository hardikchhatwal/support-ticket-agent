# API rate limits and 429 errors
Category: technical

If your API requests return HTTP 429, you are exceeding your plan's rate limit.

1. Check the `Retry-After` header and wait that many seconds before retrying.
2. Add exponential backoff to your client so retries space out automatically.
3. Batch requests where possible instead of sending many small calls.
4. Review your current limits under Settings > API > Usage.

Free plans allow 60 requests/minute; paid plans allow 600 requests/minute. Sustained 429s despite backoff usually mean you need a higher tier. Requests for a custom limit beyond the published tiers must be reviewed by our team.
