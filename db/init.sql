CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id        bigserial PRIMARY KEY,
    content   text NOT NULL,
    source    text NOT NULL,
    page      integer,
    embedding vector(768) NOT NULL
);