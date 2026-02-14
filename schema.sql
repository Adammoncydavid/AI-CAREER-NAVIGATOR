-- schema.sql

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Profiles
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    state TEXT,
    location_type TEXT,
    age INTEGER,
    education_level TEXT,
    preferences TEXT, -- JSON string for language, speed, etc.
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Skills
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL,
    skill_name TEXT NOT NULL,
    proficiency_score INTEGER CHECK(proficiency_score BETWEEN 0 AND 100),
    assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (profile_id) REFERENCES profiles (id) ON DELETE CASCADE
);

-- Career Predictions
CREATE TABLE IF NOT EXISTS career_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER, -- Can be NULL for guest users
    target_role TEXT,
    feasibility_score INTEGER,
    input_snapshot TEXT, -- JSON string of all inputs
    prediction_result TEXT, -- JSON string of the full result
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (profile_id) REFERENCES profiles (id) ON DELETE SET NULL
);

-- Ethics Audits (Bias Detection)
CREATE TABLE IF NOT EXISTS ethics_audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    trust_score REAL,
    bias_flags TEXT, -- JSON string of flags
    model_version TEXT,
    audited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES career_predictions (id) ON DELETE CASCADE
);

-- AI Generated Projects
CREATE TABLE IF NOT EXISTS generated_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    github_template_url TEXT,
    instructions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (profile_id) REFERENCES profiles (id) ON DELETE CASCADE
);
