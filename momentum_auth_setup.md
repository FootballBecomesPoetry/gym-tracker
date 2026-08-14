# Momentum — turning on Google sign-in

This is stage 1 of multi-user: **who you are**, not yet **whose data you see**.
After this, everyone still sees the same log. Stage 2 adds `user_id` columns and
stage 3 makes queries filter on them.

You can install `patch_auth.py` and leave it there. With no `[auth]` section the
app runs exactly as it does today and the sidebar says "Single-user mode".

---

## 1. Install the dependency

```powershell
pip install "streamlit>=1.42" "Authlib>=1.3.2"
```

Add both to `requirements.txt`:

```
streamlit>=1.42
Authlib>=1.3.2
tzdata
```

`Authlib` is what Streamlit uses under the hood for OIDC. Without it, `st.login()`
raises at runtime rather than at import, so it fails at the worst moment.

---

## 2. Create Google credentials

1. Go to **console.cloud.google.com** and create a project (or reuse one).
2. **APIs & Services → OAuth consent screen**
   - User type: **External**
   - App name: Momentum, plus your email in the two contact fields
   - Scopes: the defaults are fine — you only need email and profile
   - **Test users:** add your own Google address and your friends'. While the app
     is in "Testing" only these accounts can sign in, which is a useful second
     lock alongside the allowlist below.
3. **APIs & Services → Credentials → Create credentials → OAuth client ID**
   - Type: **Web application**
   - Authorised redirect URIs — add both:
     ```
     http://localhost:8501/oauth2callback
     https://YOUR-APP.streamlit.app/oauth2callback
     ```
     Replace the second with your real Streamlit Cloud URL. The path must be
     exactly `/oauth2callback`, and Google matches it character for character —
     a trailing slash will fail.
4. Copy the **Client ID** and **Client secret**.

---

## 3. Generate a cookie secret

This signs the session cookie. It should be long and random — not a word.

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 4. Add it to secrets.toml

Append to `.streamlit/secrets.toml`, keeping your existing sections intact:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "paste-the-random-hex-here"
allowed_emails = ["you@gmail.com", "mate@gmail.com"]

[auth.google]
client_id = "....apps.googleusercontent.com"
client_secret = "...."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Watch the section order — TOML assigns every key to the header above it. `[auth]`
must come after your `[connections.postgres]` block is finished, or `url` ends up
in the wrong place and the database connection breaks. That's the same trap that
cost you an hour on the Gemini key.

`allowed_emails` is optional. Omit it and anyone who can sign in with Google gets
in; include it and only those addresses do. For "me and a few friends", keep it.

**For the deployed app**, paste the same block into the Streamlit Cloud secrets
box, but change `redirect_uri` to your `https://YOUR-APP.streamlit.app/oauth2callback`.
Local secrets.toml is not uploaded.

---

## 5. Restart and test

```powershell
streamlit run gymap.py
```

- **Sign in with Google button** → good.
- Sign in with your own account **first**. It takes `user_id` 1, which is the id
  all your existing data gets assigned to in stage 2.
- Sidebar should show your name and a Sign out button.
- Test the allowlist by signing in with an address you didn't list — you should
  get "That account isn't on the allowlist."

---

## If it doesn't work

**`redirect_uri_mismatch`** — the URI in Google Console doesn't exactly match the
one in secrets.toml. Check protocol, port, path, and trailing slash.

**Still says "Single-user mode"** — the `[auth]` section isn't being read. Check
the file is saved to disk (`type .streamlit\secrets.toml`) and that Streamlit was
fully restarted, not just reloaded in the browser. Secrets load at startup.

**"has no built-in login"** — Streamlit is older than 1.42. `pip install --upgrade streamlit`.

**403 access_denied** — your Google account isn't in the consent screen's test
users list.

---

## What comes next

**Stage 2 — schema migration.** Adds `user_id` to all fourteen tables and
backfills every existing row to user 1. Run once. **Back up first**, since this
alters primary keys and is not something you want to retry blind.

**Stage 3 — query rewrite.** Every query gains a user filter. Only correct once
stage 2 has actually run, which is why they're separate.

Until stage 3 lands, anyone you let in sees your data. Keep the allowlist to
yourself until then.
