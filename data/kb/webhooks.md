# Setting up and troubleshooting webhooks
Category: technical

Webhooks send an HTTP POST to your server when events happen in your workspace.

1. Open Settings > API > Webhooks and click "Add endpoint".
2. Enter an HTTPS URL and choose the events to subscribe to.
3. Copy the signing secret and verify the `X-Signature` header on each request.
4. Click "Send test event" to confirm delivery.

Your endpoint must respond with a 2xx status within 10 seconds. Failed deliveries are retried with exponential backoff for up to 24 hours; see each attempt under Webhooks > Delivery log. Endpoints that fail continuously for 3 days are disabled automatically and can be re-enabled from the same page. Missing events usually mean the endpoint timed out or returned a non-2xx response; check the delivery log first.
