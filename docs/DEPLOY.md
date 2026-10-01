# Deploying to Cloud Run (runbook)

You run these steps yourself. Nothing here has been run for you: the deploy uses
your Google Cloud project, your billing, and your API keys.

Replace the `<PLACEHOLDERS>`. Run everything from the repo root.

## 1. Prerequisites

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Billing must be enabled on the project. Cloud Build builds the image from the
`Dockerfile`, so you don't need Docker installed locally.

## 2. Secrets (never bake them into the image)

The image contains no keys (`.env` is excluded by `.dockerignore`). Provide
`OPENAI_API_KEY` and the `LANGFUSE_*` values at deploy time.

**Option A: Secret Manager (recommended).** Values stay out of your shell history
and the Cloud Run config:

```bash
gcloud services enable secretmanager.googleapis.com
printf '%s' '<OPENAI_KEY>'   | gcloud secrets create openai-api-key      --data-file=-
printf '%s' '<LF_PUBLIC>'    | gcloud secrets create langfuse-public-key --data-file=-
printf '%s' '<LF_SECRET>'    | gcloud secrets create langfuse-secret-key --data-file=-
# Let Cloud Run's runtime service account read them:
#   gcloud projects add-iam-policy-binding <PROJECT_ID> \
#     --member serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com \
#     --role roles/secretmanager.secretAccessor
```

**Option B: plain env vars (quicker, less safe).** The values end up in your shell
history and are visible in the Cloud Run service config:

```
--set-env-vars OPENAI_API_KEY=<OPENAI_KEY>,LANGFUSE_PUBLIC_KEY=<LF_PUBLIC>,LANGFUSE_SECRET_KEY=<LF_SECRET>,LANGFUSE_HOST=https://cloud.langfuse.com
```

Use `https://us.cloud.langfuse.com` as `LANGFUSE_HOST` if your Langfuse project is in the US region.

## 3. Deploy

With plain env vars (Option B):

```bash
gcloud run deploy support-agent \
  --source . \
  --region <REGION> \
  --allow-unauthenticated \
  --memory 2Gi --cpu 1 \
  --min-instances 0 \
  --timeout 120 \
  --set-env-vars OPENAI_API_KEY=<OPENAI_KEY>,LANGFUSE_PUBLIC_KEY=<LF_PUBLIC>,LANGFUSE_SECRET_KEY=<LF_SECRET>,LANGFUSE_HOST=https://cloud.langfuse.com
```

With Secret Manager (Option A), replace the last line with:

```bash
  --set-secrets OPENAI_API_KEY=openai-api-key:latest,LANGFUSE_PUBLIC_KEY=langfuse-public-key:latest,LANGFUSE_SECRET_KEY=langfuse-secret-key:latest \
  --set-env-vars LANGFUSE_HOST=https://cloud.langfuse.com
```

**`--allow-unauthenticated` makes the service public.** Anyone with the URL can
call it, and each request costs OpenAI tokens. The app has an in-memory per-IP
rate limit (30 requests/min, set with `RATE_LIMIT_PER_MIN`), but it resets on
restart and is per instance. Set a billing budget alert on the project, and a
spend limit on the OpenAI key.

The first build is slow (several minutes): the image is large because it bundles
torch and the embedding model.

## 4. Verify

```bash
curl <SERVICE_URL>/healthz        # expect {"status":"ok"}
```

Then open `<SERVICE_URL>/` in a browser and ask a question. `<SERVICE_URL>` is
printed at the end of `gcloud run deploy`.

## Known limitations (demo-grade by design)

- **Cold starts.** `--min-instances 0` scales to zero when idle, so the first
  request after a quiet period is slow while the container starts and loads the
  embedding model. Use `--min-instances 1` to avoid this; it costs money while idle.
- **Conversation memory is not durable.** SQLite lives on the container's
  ephemeral disk. Memory is lost on cold starts and is not shared between
  instances, so a follow-up routed to a different instance won't see earlier
  turns. Fine for a demo; real scale needs a managed store (for example
  Postgres/Cloud SQL with LangGraph's Postgres checkpointer).
- **Escalations are not durable either.** `escalate_to_human` appends to a log
  file on the same ephemeral disk, and Slack notification only happens if
  `SLACK_WEBHOOK_URL` is set. Set it if you want escalations to reach a person.
