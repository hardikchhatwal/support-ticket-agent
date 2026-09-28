# Single sign-on (SSO) with SAML
Category: account

SSO lets your team sign in with your company identity provider (Okta, Azure AD / Entra ID, Google Workspace, OneLogin). It is available on the Business and Enterprise plans.

1. Sign in as a workspace admin and open Settings > Security > Single sign-on.
2. Copy the ACS URL and Entity ID into a new SAML app in your identity provider.
3. Paste your provider's metadata URL back into the SSO settings.
4. Click "Test connection", then "Enforce SSO" once the test passes.

After SSO is enforced, members sign in through your identity provider and password sign-in is disabled for them. Admins keep a password-based break-glass login. If a test fails with a certificate or signature error, re-download the metadata from your identity provider. SCIM user provisioning and custom SAML attribute mapping require setup help from our account team.
