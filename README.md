# Predicting March Madness (Public App)

This repository contains the **Streamlit app code** for the Predicting March Madness project.

- Live app (add after deploy): `<STREAMLIT_APP_URL>`

## Why this repo has no data

The datasets and model artifacts are kept in a **private GitHub repo** to avoid making them publicly downloadable.
This public app repo fetches required files **server-side** at runtime using a read-only GitHub token stored in Streamlit Secrets.

## Deployment (Streamlit Community Cloud)

1) Push this repo to GitHub as **public**.
2) Create a new app on Streamlit Community Cloud pointing at this repo.
3) In the Streamlit app settings, add Secrets:

```toml
[private_data]
repo = "CodyButtles/Predicting-March-Madness"
ref = "main"
token = "<YOUR_FINE_GRAINED_READ_ONLY_TOKEN>"
```

Token guidance:
- Use a **fine-grained** GitHub token
- Limit access to only the private repo `CodyButtles/Predicting-March-Madness`
- Grant **read-only** access to repository contents

## Notes

- The app is pinned to a single year (2024) for public deployment.
- If you run this repo locally without the private data repo + token, pages that require data will fail with missing-file errors.
