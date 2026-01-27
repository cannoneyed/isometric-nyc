"""
Database-backed generation queue.

Provides a robust, persistent queue for generation and render requests
that survives server restarts and doesn't rely on client-side state.
"""

import json
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class QueueItemType(str, Enum):
  GENERATE = "generate"
  RENDER = "render"


class QueueItemStatus(str, Enum):
  PENDING = "pending"
  PROCESSING = "processing"
  COMPLETE = "complete"
  ERROR = "error"


@dataclass
class QueueItem:
  """Represents a single item in the generation queue."""

  id: int
  item_type: QueueItemType
  quadrants: list[tuple[int, int]]
  model_id: str | None
  status: QueueItemStatus
  created_at: float
  started_at: float | None
  completed_at: float | None
  error_message: str | None
  result_message: str | None
  context_quadrants: list[tuple[int, int]] | None = None
  prompt: str | None = None
  negative_prompt: str | None = None

  @classmethod
  def from_row(cls, row: tuple) -> "QueueItem":
    """Create a QueueItem from a database row."""
    # Handle schema evolution: 10 base columns + optional context_quadrants + optional prompt + optional negative_prompt
    context = None
    prompt = None
    negative_prompt = None
    if len(row) > 10 and row[10]:
      context = json.loads(row[10])
    if len(row) > 11 and row[11]:
      prompt = row[11]
    if len(row) > 12 and row[12]:
      negative_prompt = row[12]

    return cls(
      id=row[0],
      item_type=QueueItemType(row[1]),
      quadrants=json.loads(row[2]),
      model_id=row[3],
      status=QueueItemStatus(row[4]),
      created_at=row[5],
      started_at=row[6],
      completed_at=row[7],
      error_message=row[8],
      result_message=row[9],
      context_quadrants=context,
      prompt=prompt,
      negative_prompt=negative_prompt,
    )

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary for JSON serialization."""
    result = {
      "id": self.id,
      "type": self.item_type.value,
      "quadrants": self.quadrants,
      "model_id": self.model_id,
      "status": self.status.value,
      "created_at": self.created_at,
      "started_at": self.started_at,
      "completed_at": self.completed_at,
      "error_message": self.error_message,
      "result_message": self.result_message,
    }
    if self.context_quadrants:
      result["context_quadrants"] = self.context_quadrants
    if self.prompt:
      result["prompt"] = self.prompt
    if self.negative_prompt:
      result["negative_prompt"] = self.negative_prompt
    return result


def init_queue_table(conn: sqlite3.Connection) -> None:
  """Initialize the generation_queue table if it doesn't exist."""
  cursor = conn.cursor()
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS generation_queue (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      item_type TEXT NOT NULL,
      quadrants TEXT NOT NULL,
      model_id TEXT,
      status TEXT NOT NULL DEFAULT 'pending',
      created_at REAL NOT NULL,
      started_at REAL,
      completed_at REAL,
      error_message TEXT,
      result_message TEXT,
      context_quadrants TEXT,
      prompt TEXT,
      negative_prompt TEXT
    )
  """)
  # Create index on status for efficient queue queries
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_queue_status ON generation_queue(status)
  """)

  # Migration: Add columns if they don't exist (for existing dbs)
  cursor.execute("PRAGMA table_info(generation_queue)")
  columns = [row[1] for row in cursor.fetchall()]
  if "context_quadrants" not in columns:
    cursor.execute("ALTER TABLE generation_queue ADD COLUMN context_quadrants TEXT")
  if "prompt" not in columns:
    cursor.execute("ALTER TABLE generation_queue ADD COLUMN prompt TEXT")
  if "negative_prompt" not in columns:
    cursor.execute("ALTER TABLE generation_queue ADD COLUMN negative_prompt TEXT")

  conn.commit()


def add_to_queue(
  conn: sqlite3.Connection,
  item_type: QueueItemType,
  quadrants: list[tuple[int, int]],
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
  negative_prompt: str | None = None,
) -> QueueItem:
  """
  Add a new item to the generation queue.

  Args:
    conn: Database connection
    item_type: Type of operation (generate or render)
    quadrants: List of (x, y) quadrant coordinates to generate
    model_id: Optional model ID for generation
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as
      context. These quadrants provide surrounding pixel art context for the
      generation. If a context quadrant has a generation, that will be used;
      otherwise the render will be used.
    prompt: Optional additional prompt text for generation
    negative_prompt: Optional negative prompt text for generation

  Returns:
    The created QueueItem
  """
  cursor = conn.cursor()
  created_at = time.time()

  context_json = json.dumps(context_quadrants) if context_quadrants else None

  cursor.execute(
    """
    INSERT INTO generation_queue
      (item_type, quadrants, model_id, status, created_at, context_quadrants, prompt, negative_prompt)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
      item_type.value,
      json.dumps(quadrants),
      model_id,
      QueueItemStatus.PENDING.value,
      created_at,
      context_json,
      prompt,
      negative_prompt,
    ),
  )
  conn.commit()

  item_id = cursor.lastrowid
  return QueueItem(
    id=item_id,
    item_type=item_type,
    quadrants=quadrants,
    model_id=model_id,
    status=QueueItemStatus.PENDING,
    created_at=created_at,
    started_at=None,
    completed_at=None,
    error_message=None,
    result_message=None,
    context_quadrants=context_quadrants,
    prompt=prompt,
    negative_prompt=negative_prompt,
  )


def get_next_pending_item(conn: sqlite3.Connection) -> QueueItem | None:
  """
  Get the next pending item from the queue.

  Returns the oldest pending item, or None if queue is empty.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, item_type, quadrants, model_id, status,
           created_at, started_at, completed_at, error_message, result_message,
           context_quadrants, prompt, negative_prompt
    FROM generation_queue
    WHERE status = ?
    ORDER BY created_at ASC
    LIMIT 1
    """,
    (QueueItemStatus.PENDING.value,),
  )
  row = cursor.fetchone()
  return QueueItem.from_row(row) if row else None


def get_next_pending_item_for_available_model(
  conn: sqlite3.Connection, busy_models: set[str | None]
) -> QueueItem | None:
  """
  Get the next pending item for a model that isn't currently busy.

  This enables parallel processing of different models' queues.

  Args:
    conn: Database connection
    busy_models: Set of model_ids that are currently processing
                 (None represents the default/no model)

  Returns None if no available items.
  """
  cursor = conn.cursor()

  # Get all pending items ordered by creation time
  cursor.execute(
    """
    SELECT id, item_type, quadrants, model_id, status,
           created_at, started_at, completed_at, error_message, result_message,
           context_quadrants, prompt, negative_prompt
    FROM generation_queue
    WHERE status = ?
    ORDER BY created_at ASC
    """,
    (QueueItemStatus.PENDING.value,),
  )

  for row in cursor.fetchall():
    item = QueueItem.from_row(row)
    # Check if this model is available (not busy)
    if item.model_id not in busy_models:
      return item

  return None


def get_processing_item(conn: sqlite3.Connection) -> QueueItem | None:
  """
  Get the currently processing item, if any.

  Returns the item currently being processed, or None.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, item_type, quadrants, model_id, status,
           created_at, started_at, completed_at, error_message, result_message,
           context_quadrants, prompt, negative_prompt
    FROM generation_queue
    WHERE status = ?
    ORDER BY started_at DESC
    LIMIT 1
    """,
    (QueueItemStatus.PROCESSING.value,),
  )
  row = cursor.fetchone()
  return QueueItem.from_row(row) if row else None


def mark_item_processing(conn: sqlite3.Connection, item_id: int) -> None:
  """Mark a queue item as processing."""
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE generation_queue
    SET status = ?, started_at = ?
    WHERE id = ?
    """,
    (QueueItemStatus.PROCESSING.value, time.time(), item_id),
  )
  conn.commit()


def mark_item_complete(
  conn: sqlite3.Connection, item_id: int, result_message: str | None = None
) -> None:
  """Mark a queue item as complete."""
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE generation_queue
    SET status = ?, completed_at = ?, result_message = ?
    WHERE id = ?
    """,
    (QueueItemStatus.COMPLETE.value, time.time(), result_message, item_id),
  )
  conn.commit()


def mark_item_error(conn: sqlite3.Connection, item_id: int, error_message: str) -> None:
  """Mark a queue item as errored."""
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE generation_queue
    SET status = ?, completed_at = ?, error_message = ?
    WHERE id = ?
    """,
    (QueueItemStatus.ERROR.value, time.time(), error_message, item_id),
  )
  conn.commit()


def get_pending_queue(conn: sqlite3.Connection) -> list[QueueItem]:
  """Get all pending items in the queue, ordered by creation time."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, item_type, quadrants, model_id, status,
           created_at, started_at, completed_at, error_message, result_message,
           context_quadrants, prompt, negative_prompt
    FROM generation_queue
    WHERE status = ?
    ORDER BY created_at ASC
    """,
    (QueueItemStatus.PENDING.value,),
  )
  return [QueueItem.from_row(row) for row in cursor.fetchall()]


def get_queue_position(conn: sqlite3.Connection, item_id: int) -> int:
  """
  Get the position of an item in the queue.

  Returns 0 if the item is currently processing,
  1 if it's first in the pending queue, etc.
  Returns -1 if the item is not found or already complete.
  """
  cursor = conn.cursor()

  # Check if it's processing
  cursor.execute(
    "SELECT 1 FROM generation_queue WHERE id = ? AND status = ?",
    (item_id, QueueItemStatus.PROCESSING.value),
  )
  if cursor.fetchone():
    return 0

  # Check if it's pending and get position
  cursor.execute(
    """
    SELECT COUNT(*) + 1
    FROM generation_queue
    WHERE status = ? AND created_at < (
      SELECT created_at FROM generation_queue WHERE id = ?
    )
    """,
    (QueueItemStatus.PENDING.value, item_id),
  )
  row = cursor.fetchone()
  if row and row[0] > 0:
    return row[0]

  return -1


def get_queue_status(conn: sqlite3.Connection) -> dict[str, Any]:
  """
  Get a summary of the current queue status.

  Returns a dictionary with:
    - is_processing: bool
    - current_item: dict | None
    - pending_count: int
    - pending_items: list of dicts
  """
  processing = get_processing_item(conn)
  pending = get_pending_queue(conn)

  return {
    "is_processing": processing is not None,
    "current_item": processing.to_dict() if processing else None,
    "pending_count": len(pending),
    "pending_items": [item.to_dict() for item in pending],
  }


def get_all_processing_items(conn: sqlite3.Connection) -> list[QueueItem]:
  """Get all items currently being processed (one per model in parallel mode)."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, item_type, quadrants, model_id, status,
           created_at, started_at, completed_at, error_message, result_message,
           context_quadrants, prompt, negative_prompt
    FROM generation_queue
    WHERE status = ?
    ORDER BY started_at ASC
    """,
    (QueueItemStatus.PROCESSING.value,),
  )
  return [QueueItem.from_row(row) for row in cursor.fetchall()]


def get_queue_status_by_model(conn: sqlite3.Connection) -> dict[str, Any]:
  """
  Get queue status grouped by model.

  Returns a dictionary with:
    - by_model: dict mapping model_id -> {
        is_processing: bool,
        current_item: dict | None,
        pending_count: int,
        pending_items: list of dicts,
        position: int (1-based, 0 if processing)
      }
    - total_pending: int
    - processing_models: list of model_ids currently processing
    - all_processing_quadrants: list of all quadrants currently being processed
  """
  processing_items = get_all_processing_items(conn)
  pending = get_pending_queue(conn)

  # Group pending items by model_id
  by_model: dict[str, dict[str, Any]] = {}

  for item in pending:
    model_id = item.model_id or "default"
    if model_id not in by_model:
      by_model[model_id] = {
        "is_processing": False,
        "current_item": None,
        "pending_count": 0,
        "pending_items": [],
      }
    by_model[model_id]["pending_count"] += 1
    by_model[model_id]["pending_items"].append(item.to_dict())

  # Add all processing items to their model's status
  processing_models = []
  all_processing_quadrants = []

  for processing in processing_items:
    model_id = processing.model_id or "default"
    processing_models.append(model_id)

    # Collect all processing quadrants
    if processing.quadrants:
      all_processing_quadrants.extend(processing.quadrants)

    if model_id not in by_model:
      by_model[model_id] = {
        "is_processing": True,
        "current_item": processing.to_dict(),
        "pending_count": 0,
        "pending_items": [],
      }
    else:
      by_model[model_id]["is_processing"] = True
      by_model[model_id]["current_item"] = processing.to_dict()

  return {
    "by_model": by_model,
    "total_pending": len(pending),
    "processing_models": processing_models,
    "all_processing_quadrants": all_processing_quadrants,
  }


def get_queue_position_for_model(
  conn: sqlite3.Connection, item_id: int, model_id: str | None
) -> int:
  """
  Get the position of an item within its model's queue.

  Returns 0 if the item is currently processing,
  1 if it's first in the pending queue for this model, etc.
  Returns -1 if the item is not found or already complete.
  """
  cursor = conn.cursor()

  # Check if it's processing
  cursor.execute(
    "SELECT 1 FROM generation_queue WHERE id = ? AND status = ?",
    (item_id, QueueItemStatus.PROCESSING.value),
  )
  if cursor.fetchone():
    return 0

  # For model-specific position, count items with same model_id that are ahead
  # Use empty string for NULL model_id comparison
  if model_id is None:
    cursor.execute(
      """
      SELECT COUNT(*) + 1
      FROM generation_queue
      WHERE status = ?
        AND model_id IS NULL
        AND created_at < (
          SELECT created_at FROM generation_queue WHERE id = ?
        )
      """,
      (QueueItemStatus.PENDING.value, item_id),
    )
  else:
    cursor.execute(
      """
      SELECT COUNT(*) + 1
      FROM generation_queue
      WHERE status = ?
        AND model_id = ?
        AND created_at < (
          SELECT created_at FROM generation_queue WHERE id = ?
        )
      """,
      (QueueItemStatus.PENDING.value, model_id, item_id),
    )

  row = cursor.fetchone()
  if row:
    return row[0]

  return -1


def reset_all_processing_items(conn: sqlite3.Connection) -> int:
  """
  Reset ALL items in 'processing' state back to 'pending'.

  This should be called on server startup to ensure any items that were
  interrupted mid-processing (e.g., due to server shutdown) are retried.

  Returns the number of items reset.
  """
  cursor = conn.cursor()

  cursor.execute(
    """
    UPDATE generation_queue
    SET status = ?, started_at = NULL
    WHERE status = ?
    """,
    (QueueItemStatus.PENDING.value, QueueItemStatus.PROCESSING.value),
  )
  conn.commit()
  return cursor.rowcount


def cleanup_stale_processing(
  conn: sqlite3.Connection, max_age_seconds: float = 3600.0
) -> int:
  """
  Clean up items stuck in 'processing' state for too long.

  This handles cases where the server crashed during processing.
  Items older than max_age_seconds are reset to 'pending'.

  Note: For server startup, use reset_all_processing_items() instead,
  which resets all processing items regardless of age.

  Returns the number of items reset.
  """
  cursor = conn.cursor()
  cutoff_time = time.time() - max_age_seconds

  cursor.execute(
    """
    UPDATE generation_queue
    SET status = ?, started_at = NULL
    WHERE status = ? AND started_at < ?
    """,
    (QueueItemStatus.PENDING.value, QueueItemStatus.PROCESSING.value, cutoff_time),
  )
  conn.commit()
  return cursor.rowcount


def clear_completed_items(
  conn: sqlite3.Connection, max_age_seconds: float = 86400.0
) -> int:
  """
  Delete completed/errored items older than max_age_seconds.

  Returns the number of items deleted.
  """
  cursor = conn.cursor()
  cutoff_time = time.time() - max_age_seconds

  cursor.execute(
    """
    DELETE FROM generation_queue
    WHERE status IN (?, ?) AND completed_at < ?
    """,
    (QueueItemStatus.COMPLETE.value, QueueItemStatus.ERROR.value, cutoff_time),
  )
  conn.commit()
  return cursor.rowcount


def clear_pending_queue(conn: sqlite3.Connection) -> int:
  """
  Delete all pending items from the queue.

  Does NOT affect items that are currently processing.
  Returns the number of items deleted.
  """
  cursor = conn.cursor()

  cursor.execute(
    """
    DELETE FROM generation_queue
    WHERE status = ?
    """,
    (QueueItemStatus.PENDING.value,),
  )
  conn.commit()
  return cursor.rowcount


def cancel_queue_item_by_id(conn: sqlite3.Connection, item_id: int) -> bool:
  """
  Cancel a specific queue item by its ID.

  Can cancel items in 'pending' or 'processing' status.

  Returns True if an item was cancelled, False otherwise.
  """
  cursor = conn.cursor()

  cursor.execute(
    """
    UPDATE generation_queue
    SET status = ?, completed_at = ?, error_message = ?
    WHERE id = ? AND status IN (?, ?)
    """,
    (
      QueueItemStatus.ERROR.value,
      time.time(),
      "Cancelled by user",
      item_id,
      QueueItemStatus.PENDING.value,
      QueueItemStatus.PROCESSING.value,
    ),
  )
  conn.commit()

  return cursor.rowcount > 0


def cancel_processing_items(conn: sqlite3.Connection) -> int:
  """
  Mark all processing items as cancelled (error status).

  This is used when the user wants to cancel everything, including
  items that are currently being processed.

  Returns the number of items cancelled.
  """
  cursor = conn.cursor()

  cursor.execute(
    """
    UPDATE generation_queue
    SET status = ?, completed_at = ?, error_message = ?
    WHERE status = ?
    """,
    (
      QueueItemStatus.ERROR.value,
      time.time(),
      "Cancelled by user",
      QueueItemStatus.PROCESSING.value,
    ),
  )
  conn.commit()
  return cursor.rowcount


def clear_all_queue_items(conn: sqlite3.Connection) -> dict[str, int]:
  """
  Delete all items from the queue (pending, processing, complete, error).

  Returns a dict with counts by status.
  """
  cursor = conn.cursor()

  # Get counts first
  cursor.execute(
    """
    SELECT status, COUNT(*) FROM generation_queue GROUP BY status
    """
  )
  counts = {row[0]: row[1] for row in cursor.fetchall()}

  # Delete all
  cursor.execute("DELETE FROM generation_queue")
  conn.commit()

  return counts
