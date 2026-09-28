# Creating and rotating API keys
Category: technical

API keys authenticate requests to our REST API.

1. Sign in and open Settings > API > Keys.
2. Click "Create key", give it a name, and choose read-only or read-write scope.
3. Copy the key immediately; it is shown only once.
4. Send it in the `Authorization: Bearer <key>` header on every request.

To rotate a key, create a new one, update your integration, then click "Revoke" on the old key. Revoked keys stop working within one minute. HTTP 401 responses mean the key is missing, mistyped or revoked; HTTP 403 means the key's scope does not allow that action. Keys belong to the user who created them and are revoked automatically if that user is removed from the workspace.
