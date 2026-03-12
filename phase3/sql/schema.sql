-- ============================================================================
-- Maritime Intelligence Agent - Database Schema
-- ============================================================================
-- Version: 3.0
-- Description: Core data model for maritime news ingestion, analysis, and
--              risk tracking. Optimized for time-series queries and
--              analytical reporting.
--
-- Tables:
--   - news_alerts: Raw ingested articles with NLP analysis results
--   - processing_log: Audit trail of every workflow execution
--   - risk_thresholds: Configurable alerting rules
--
-- Usage:
--   psql -U maritime_user -d maritime_intel -f 01_schema.sql
-- ============================================================================

-- Ensure we're in the correct database
\c maritime_intel

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search


-- ============================================================================
-- Table: news_alerts
-- ============================================================================
-- Stores every maritime news article scraped from RSS feeds, along with
-- the AI-generated risk assessment from the FastAPI service.
--
-- Design notes:
--   - `alert_id` is a UUID for globally unique identification
--   - `source_url` is indexed for deduplication checks
--   - `risk_score` is constrained to [1, 10] to match the AI service output
--   - `entities` stores the spaCy NER output as JSONB for queryability
--   - GIN indexes on JSONB columns enable fast filtering by entity type
-- ============================================================================

CREATE TABLE IF NOT EXISTS news_alerts (
    -- Primary key
    alert_id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Source metadata
    source_name         VARCHAR(255) NOT NULL,              -- e.g. "gCaptain", "Maritime Executive"
    source_url          TEXT NOT NULL UNIQUE,               -- RSS item link (deduplication key)
    published_at        TIMESTAMP WITH TIME ZONE,           -- Article publication time
    scraped_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),  -- When we ingested it
    
    -- Article content
    headline            TEXT NOT NULL,
    full_text           TEXT,                               -- Article body (if available)
    summary             TEXT,                               -- RSS description/excerpt
    keywords            TEXT[],                             -- Extracted/provided keywords
    
    -- NLP analysis results (from FastAPI /analyze endpoint)
    risk_score          INTEGER CHECK (risk_score BETWEEN 1 AND 10),
    sentiment_label     VARCHAR(20),                        -- "positive" | "neutral" | "negative"
    sentiment_score     NUMERIC(6, 5),                      -- Confidence [0.0, 1.0]
    sentiment_all_scores JSONB,                             -- {"positive": 0.05, "neutral": 0.15, "negative": 0.80}
    
    -- Named entities (from spaCy)
    entities            JSONB,                              -- Full entity list with labels
    locations           JSONB,                              -- Location entities only (GPE, LOC, FAC)
    
    -- Risk score breakdown (for explainability)
    score_breakdown     JSONB,                              -- Intermediate scoring components
    processing_time_ms  NUMERIC(10, 2),                     -- AI inference latency
    
    -- Status tracking
    processing_status   VARCHAR(50) DEFAULT 'pending',      -- "pending" | "analyzed" | "failed" | "archived"
    error_message       TEXT,                               -- Populated if processing_status = "failed"
    flagged_for_review  BOOLEAN DEFAULT FALSE,              -- Manual review flag
    reviewed_by         VARCHAR(255),                       -- Username of reviewer
    reviewed_at         TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX idx_news_alerts_scraped_at ON news_alerts (scraped_at DESC);
CREATE INDEX idx_news_alerts_published_at ON news_alerts (published_at DESC NULLS LAST);
CREATE INDEX idx_news_alerts_risk_score ON news_alerts (risk_score DESC) WHERE risk_score IS NOT NULL;
CREATE INDEX idx_news_alerts_source_name ON news_alerts (source_name);
CREATE INDEX idx_news_alerts_processing_status ON news_alerts (processing_status);
CREATE INDEX idx_news_alerts_sentiment_label ON news_alerts (sentiment_label);
CREATE INDEX idx_news_alerts_flagged ON news_alerts (flagged_for_review) WHERE flagged_for_review = TRUE;

-- GIN indexes for JSONB columns (enables fast entity/location filtering)
CREATE INDEX idx_news_alerts_entities_gin ON news_alerts USING GIN (entities);
CREATE INDEX idx_news_alerts_locations_gin ON news_alerts USING GIN (locations);

-- Full-text search index on headline + full_text
CREATE INDEX idx_news_alerts_fts ON news_alerts USING GIN (to_tsvector('english', headline || ' ' || COALESCE(full_text, '')));

-- Trigram index for fuzzy headline matching (deduplication assist)
CREATE INDEX idx_news_alerts_headline_trgm ON news_alerts USING GIN (headline gin_trgm_ops);

-- Comment on table
COMMENT ON TABLE news_alerts IS 'Maritime news articles with AI risk analysis';
COMMENT ON COLUMN news_alerts.source_url IS 'Unique URL used for deduplication';
COMMENT ON COLUMN news_alerts.risk_score IS 'AI-generated risk score [1-10], where 10 = critical';
COMMENT ON COLUMN news_alerts.entities IS 'spaCy NER output: [{"text": "Shanghai", "label": "GPE", ...}, ...]';
COMMENT ON COLUMN news_alerts.locations IS 'Location entities only (GPE, LOC, FAC)';
COMMENT ON COLUMN news_alerts.processing_status IS 'Workflow state: pending | analyzed | failed | archived';


-- ============================================================================
-- Table: processing_log
-- ============================================================================
-- Audit trail of every n8n workflow execution, capturing success/failure
-- metrics and error diagnostics.
-- ============================================================================

CREATE TABLE IF NOT EXISTS processing_log (
    log_id              SERIAL PRIMARY KEY,
    workflow_name       VARCHAR(255) NOT NULL,              -- e.g. "Maritime RSS Scraper"
    execution_id        VARCHAR(255),                       -- n8n execution ID (if available)
    started_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMP WITH TIME ZONE,
    duration_ms         INTEGER,
    
    -- Metrics
    items_scraped       INTEGER DEFAULT 0,
    items_filtered      INTEGER DEFAULT 0,
    items_analyzed      INTEGER DEFAULT 0,
    items_stored        INTEGER DEFAULT 0,
    items_failed        INTEGER DEFAULT 0,
    
    -- Status
    status              VARCHAR(50) NOT NULL,               -- "success" | "partial" | "failed"
    error_details       JSONB,                              -- Structured error info
    
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_processing_log_started_at ON processing_log (started_at DESC);
CREATE INDEX idx_processing_log_workflow_name ON processing_log (workflow_name);
CREATE INDEX idx_processing_log_status ON processing_log (status);

COMMENT ON TABLE processing_log IS 'Audit log of n8n workflow executions';


-- ============================================================================
-- Table: risk_thresholds
-- ============================================================================
-- Configurable alerting rules. n8n workflows can query this table to
-- determine whether a given risk_score should trigger a notification.
-- ============================================================================

CREATE TABLE IF NOT EXISTS risk_thresholds (
    threshold_id        SERIAL PRIMARY KEY,
    threshold_name      VARCHAR(255) NOT NULL UNIQUE,       -- e.g. "Critical Alert"
    min_risk_score      INTEGER NOT NULL CHECK (min_risk_score BETWEEN 1 AND 10),
    notification_channel VARCHAR(100),                      -- "email" | "slack" | "sms" | "webhook"
    notification_target  TEXT,                              -- Email address, Slack webhook URL, etc.
    enabled             BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Insert default thresholds
INSERT INTO risk_thresholds (threshold_name, min_risk_score, notification_channel, notification_target, enabled)
VALUES
    ('Low Risk Monitor',      1, 'none',   NULL,                           FALSE),
    ('Medium Risk Review',    4, 'none',   NULL,                           FALSE),
    ('High Risk Alert',       7, 'email',  'analyst@maritime-intel.local', TRUE),
    ('Critical Escalation',   9, 'slack',  'https://hooks.slack.com/...',  TRUE)
ON CONFLICT (threshold_name) DO NOTHING;

COMMENT ON TABLE risk_thresholds IS 'Configurable alerting rules based on risk_score';


-- ============================================================================
-- Function: update_updated_at_column()
-- ============================================================================
-- Trigger function to automatically update the `updated_at` timestamp
-- whenever a row is modified.
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to news_alerts
DROP TRIGGER IF EXISTS trigger_update_news_alerts_updated_at ON news_alerts;
CREATE TRIGGER trigger_update_news_alerts_updated_at
    BEFORE UPDATE ON news_alerts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Attach trigger to risk_thresholds
DROP TRIGGER IF EXISTS trigger_update_risk_thresholds_updated_at ON risk_thresholds;
CREATE TRIGGER trigger_update_risk_thresholds_updated_at
    BEFORE UPDATE ON risk_thresholds
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================================================
-- Materialized View: daily_risk_summary
-- ============================================================================
-- Pre-aggregated daily statistics for dashboard queries.
-- Refresh periodically via n8n or a cron job.
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS daily_risk_summary AS
SELECT
    DATE(scraped_at) AS date,
    source_name,
    COUNT(*) AS total_articles,
    COUNT(*) FILTER (WHERE risk_score >= 7) AS high_risk_count,
    COUNT(*) FILTER (WHERE risk_score >= 9) AS critical_risk_count,
    AVG(risk_score) AS avg_risk_score,
    MAX(risk_score) AS max_risk_score,
    COUNT(*) FILTER (WHERE sentiment_label = 'negative') AS negative_sentiment_count,
    COUNT(*) FILTER (WHERE sentiment_label = 'positive') AS positive_sentiment_count,
    AVG(processing_time_ms) AS avg_processing_time_ms
FROM
    news_alerts
WHERE
    processing_status = 'analyzed'
GROUP BY
    DATE(scraped_at), source_name
ORDER BY
    date DESC, source_name;

-- Index for fast date-range queries
CREATE UNIQUE INDEX idx_daily_risk_summary_date_source ON daily_risk_summary (date, source_name);

COMMENT ON MATERIALIZED VIEW daily_risk_summary IS 'Pre-aggregated daily statistics (refresh periodically)';


-- ============================================================================
-- Useful Query Functions
-- ============================================================================

-- Function: Get high-risk articles from the last N hours
CREATE OR REPLACE FUNCTION get_recent_high_risk_alerts(
    hours_back INTEGER DEFAULT 24,
    min_risk INTEGER DEFAULT 7
)
RETURNS TABLE (
    alert_id UUID,
    headline TEXT,
    source_name VARCHAR,
    source_url TEXT,
    risk_score INTEGER,
    sentiment_label VARCHAR,
    locations JSONB,
    scraped_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        na.alert_id,
        na.headline,
        na.source_name,
        na.source_url,
        na.risk_score,
        na.sentiment_label,
        na.locations,
        na.scraped_at
    FROM
        news_alerts na
    WHERE
        na.scraped_at >= NOW() - (hours_back || ' hours')::INTERVAL
        AND na.risk_score >= min_risk
        AND na.processing_status = 'analyzed'
    ORDER BY
        na.risk_score DESC, na.scraped_at DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_recent_high_risk_alerts IS 'Retrieve high-risk alerts from the last N hours';


-- ============================================================================
-- Grant Permissions
-- ============================================================================
-- Ensure the maritime_user has full access to all tables

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO maritime_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO maritime_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO maritime_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO maritime_user;

-- Also grant access to future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO maritime_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO maritime_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO maritime_user;


-- ============================================================================
-- Verification Queries
-- ============================================================================

-- Uncomment these to verify the schema after running the script:

-- \dt                                    -- List all tables
-- \d news_alerts                         -- Describe news_alerts table
-- \di                                    -- List all indexes
-- SELECT * FROM risk_thresholds;         -- Show default thresholds
-- SELECT * FROM daily_risk_summary;      -- Check materialized view

-- Test the function:
-- SELECT * FROM get_recent_high_risk_alerts(24, 7);


-- ============================================================================
-- Schema Version Tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version         VARCHAR(20) PRIMARY KEY,
    applied_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    description     TEXT
);

INSERT INTO schema_version (version, description)
VALUES ('3.0.0', 'Phase 3: Complete intelligence schema with audit logging and risk thresholds')
ON CONFLICT (version) DO NOTHING;

-- End of schema script