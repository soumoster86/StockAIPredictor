-- Supabase / Postgres schema for cloud-persistent signal journal.
-- Run in Supabase SQL Editor, then put credentials in Streamlit secrets.
--
-- Required secrets.toml:
--
--   [journal]
--   backend = "supabase"
--   supabase_url = "https://YOUR_PROJECT.supabase.co"
--   supabase_key = "YOUR_SERVICE_ROLE_OR_ANON_KEY"
--   table = "signal_journal"

create table if not exists public.signal_journal (
  id bigserial primary key,
  username text not null,
  signal_date text not null,
  symbol text not null,
  name text,
  model_type text not null default '',
  signal text,
  probability double precision,
  rating text,
  entry double precision,
  stop double precision,
  target double precision,
  reward_risk double precision,
  risk_score double precision,
  logged_at text,
  created_at timestamptz not null default now()
);

-- One log per user / day / symbol / model
create unique index if not exists signal_journal_dedupe
  on public.signal_journal (username, signal_date, symbol, model_type);

create index if not exists signal_journal_user_date
  on public.signal_journal (username, signal_date desc);

-- If using the anon key from the browser/app, enable RLS and add policies.
-- For a private Streamlit app with the service_role key, you can skip RLS.
alter table public.signal_journal enable row level security;

-- Service role bypasses RLS. Optional: allow authenticated users to manage
-- only their rows if you later switch to user JWTs.
drop policy if exists "service full access" on public.signal_journal;
-- No public policies by default — use service_role key in Streamlit secrets.
