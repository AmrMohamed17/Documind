from app.core.embedder import embed_document, embed_query
from app.core.db import get_pool


def add_documents(chunks: list) -> int:
    # 1. Embed everything first — these are network calls to Gemini,
    #    so we do them BEFORE borrowing a database connection.
    rows = [
        (
            chunk.page_content,
            chunk.metadata.get("source", "unknown"),
            chunk.metadata.get("page"),
            embed_document(chunk.page_content),
        )
        for chunk in chunks
    ]

    # 2. Borrow a connection just long enough to write the rows.
    #    Leaving the block commits automatically.
    with get_pool().connection() as conn:
        for content, source, page, embedding in rows:
            conn.execute(
                """
                INSERT INTO chunks (content, source, page, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (content, source, page, embedding),
            )
    return len(rows)


def similarity_search(query: str, k: int = 4) -> list[dict]:
    # Embed the question first (network call), THEN borrow a connection.
    query_embedding = embed_query(query)

    with get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT content, source, page, embedding <=> %s::vector AS distance
            FROM chunks
            ORDER BY distance
            LIMIT %s
            """,
            (query_embedding, k),
        ).fetchall()

    return [
        {"content": r[0], "source": r[1], "page": r[2], "distance": r[3]}
        for r in rows
    ]


def list_sources() -> list[str]:
    """Return the distinct source files currently stored."""
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT source FROM chunks ORDER BY source"
        ).fetchall()
    return [r[0] for r in rows]