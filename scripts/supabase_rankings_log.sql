-- Supabase / Postgres schema for nightly (and local) rankings run logs.
-- Run in Supabase SQL Editor after the journal table (or independently).
--
-- GitHub Actions secrets (recommended):
--   SUPABASE_URL          = https://YOUR_PROJECT.supabase.co
--   SUPABASE_SERVICE_KEY  = service_role key  (or SUPABASE_KEY)
--
-- Optional Streamlit secrets to *read* recent runs in the Screener UI:
--
--   [rankings_log]
--   enabled = true
--   supabase_url = "https://YOUR_PROJECT.supabase.co"
--   supabase_key = "YOUR_SERVICE_ROLE_KEY"
--   table = "rankings_run_log"

create table if not exists public.rankings_run_log (
  id bigserial primary key,
  logged_at timestamptz not null default now(),
  generated_at text,
  status text not null default 'success',
  runner text,
  run_id text,
  run_url text,
  workflow text,
  watchlist text,
  engine text,
  n_requested integer,
  n_scored integer,
  n_failed integer,
  batch_size integer,
  elapsed_s double precision,
  max_symbols integer,
  error_message text,
  top_symbols text,
  repo text,
  git_sha text,
  meta jsonb
);

create index if not exists rankings_run_log_logged_at
  on public.rankings_run_log (logged_at desc);

create index if not exists rankings_run_log_status
  on public.rankings_run_log (status, logged_at desc);

create index if not exists rankings_run_log_run_id
  on public.rankings_run_log (run_id);

-- Service role bypasses RLS. No public policies by default.
alter table public.rankings_run_log enable row level security;
