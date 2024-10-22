-- Users Table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,         -- Auto-incrementing unique ID starting from 1
    username VARCHAR(50) NOT NULL UNIQUE,  -- Unique username
    password VARCHAR(50) NOT NULL,         -- Password (store hashed in real-world cases)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP  -- Timestamp for user creation
);

-- Conversations Table
CREATE TABLE conversations (
    conversation_id SERIAL PRIMARY KEY,  -- Auto-incrementing unique ID starting from 1
    conversation_name VARCHAR(50),       -- Optional name or title for the conversation
    user_id INT REFERENCES users(user_id),  -- Foreign key linking to Users table
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP  -- Timestamp for when the conversation starts
);

-- Messages Table
CREATE TABLE messages (
    message_id SERIAL PRIMARY KEY,                          -- Auto-incrementing message ID starting from 1
    conversation_id INT REFERENCES conversations(conversation_id),  -- Foreign key linking to Conversations
    sender VARCHAR(50) NOT NULL,                            -- Identifies who sent the message (user or AI)
    message_text TEXT,                                      -- Actual message text
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP  -- Timestamp for when the message was sent
);

-- Files Table
CREATE TABLE files (
    file_id SERIAL PRIMARY KEY,                             -- Auto-incrementing unique ID starting from 1
    file_name VARCHAR(50) NOT NULL,                         -- Name of the uploaded file
    size VARCHAR(20) NOT NULL,                              -- Size of the file
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,  -- Timestamp for file creation
    conversation_id INT REFERENCES conversations(conversation_id)  -- Foreign key linking to Conversations
);
