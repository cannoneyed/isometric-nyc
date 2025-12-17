# Can you write a quick script in @src/isometric_nyc/e2e_generation/ called "fill_water.py"
_Exported on 12/16/2025 at 15:42:50 EST from Cursor (2.2.20)_

---

**User**

Can you write a quick script in @src/isometric_nyc/e2e_generation called "fill_water.py" that takes a tl and br x,y quadrant parameters, and then fills those quadrant pngs with the water color #4A6372

---

**Cursor**

I'll look at some existing scripts to understand the conventions and patterns used for handling quadrant coordinates and file paths.

Let me also look at how the database path is set up and how the main function is structured:

Now I have a good understanding of the patterns. Let me check what size the quadrant images are:

Now I have all the information I need. Let me create the `fill_water.py` script:

```python
"""
Fill quadrants with solid water color.

Fills a rectangular region of quadrants with a solid water color (#4A6372).

Usage:
  uv run python src/isometric_nyc/e2e_generation/fill_water.py <generation_dir> --tl X,Y --br X,Y

Examples:
  # Fill quadrants from (0,0) to (5,5) with water
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5

  # Dry run to see what would be filled
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --dry-run

  # Overwrite existing quadrants
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --overwrite
"""

import argparse
import io
import sqlite3
import sys
from pathlib import Path

from PIL import Image

# Constants
QUADRANT_SIZE = 512
WATER_COLOR = (0x4A, 0x63, 0x72)  # #4A6372


def parse_coordinate(coord_str: str) -> tuple[int, int]:
    """Parse a coordinate string like '10,15' into (x, y) tuple."""
    try:
        parts = coord_str.split(",")
        if len(parts) != 2:
            raise ValueError()
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
        )


def create_water_tile() -> bytes:
    """Create a solid water color tile as PNG bytes."""
    img = Image.new("RGB", (QUADRANT_SIZE, QUADRANT_SIZE), WATER_COLOR)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def quadrant_has_data(conn: sqlite3.Connection, x: int, y: int) -> bool:
    """Check if a quadrant already has generation data."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT generation FROM quadrants
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (x, y),
    )
    row = cursor.fetchone()
    return row is not None and row[0] is not None


def save_water_quadrant(
    conn: sqlite3.Connection, x: int, y: int, png_bytes: bytes
) -> bool:
    """
    Save water tile to a quadrant at position (x, y).

    Creates the quadrant row if it doesn't exist.
    Returns True if successful.
    """
    cursor = conn.cursor()

    # Try to update existing row first
    cursor.execute(
        """
        UPDATE quadrants
        SET generation = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (png_bytes, x, y),
    )

    if cursor.rowcount > 0:
        return True

    # Row doesn't exist, insert it
    cursor.execute(
        """
        INSERT INTO quadrants (quadrant_x, quadrant_y, generation)
        VALUES (?, ?, ?)
        """,
        (x, y, png_bytes),
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill quadrants with solid water color (#4A6372)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory (contains quadrants.db)",
    )
    parser.add_argument(
        "--tl",
        type=parse_coordinate,
        required=True,
        metavar="X,Y",
        help="Top-left coordinate of the region to fill (e.g., '0,0')",
    )
    parser.add_argument(
        "--br",
        type=parse_coordinate,
        required=True,
        metavar="X,Y",
        help="Bottom-right coordinate of the region to fill (e.g., '5,5')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quadrants (default: skip existing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be filled without actually filling",
    )

    args = parser.parse_args()

    # Resolve paths
    generation_dir = args.generation_dir.resolve()
    tl = args.tl
    br = args.br

    # Validate inputs
    if not generation_dir.exists():
        print(f"❌ Error: Generation directory not found: {generation_dir}")
        return 1

    db_path = generation_dir / "quadrants.db"
    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    # Validate coordinate range
    if tl[0] > br[0] or tl[1] > br[1]:
        print(
            f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
        )
        return 1

    # Calculate total quadrants
    width = br[0] - tl[0] + 1
    height = br[1] - tl[1] + 1
    total = width * height

    print("=" * 50)
    print("🌊 Fill Water Quadrants")
    print("=" * 50)
    print(f"📁 Generation dir: {generation_dir}")
    print(f"📍 Region: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
    print(f"📐 Size: {width}×{height} = {total} quadrants")
    print(f"🎨 Water color: #{WATER_COLOR[0]:02X}{WATER_COLOR[1]:02X}{WATER_COLOR[2]:02X}")
    print()

    if args.dry_run:
        print("🔍 Dry run - no changes will be made")
        print(f"   Would fill {total} quadrants with water color")
        return 0

    # Create water tile
    water_png = create_water_tile()
    print(f"💧 Created water tile: {len(water_png)} bytes")
    print()

    # Fill quadrants
    conn = sqlite3.connect(db_path)
    try:
        filled = 0
        skipped = 0

        for y in range(tl[1], br[1] + 1):
            for x in range(tl[0], br[0] + 1):
                # Check if quadrant already has data
                if not args.overwrite and quadrant_has_data(conn, x, y):
                    skipped += 1
                    continue

                # Save water tile
                save_water_quadrant(conn, x, y, water_png)
                filled += 1

            # Commit after each row
            conn.commit()

            # Progress indicator
            row_num = y - tl[1] + 1
            print(f"   Row {row_num}/{height}: filled {filled}, skipped {skipped}")

        conn.commit()
    finally:
        conn.close()

    print()
    print("=" * 50)
    print("✅ Fill complete!")
    print(f"   Filled: {filled} quadrants")
    print(f"   Skipped: {skipped} quadrants (already had data)")
    print(f"   Total: {total} quadrants")

    return 0


if __name__ == "__main__":
    sys.exit(main())

```

Done! I've created `/Users/andycoenen/cannoneyed/isometric-nyc/src/isometric_nyc/e2e_generation/fill_water.py`. Here's what it does:

**Usage:**
```bash
# Fill quadrants from (0,0) to (5,5) with water
uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5

# Dry run to preview what would be filled
uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --dry-run

# Overwrite existing quadrants with water
uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --overwrite
```

**Features:**
- Takes `--tl X,Y` (top-left) and `--br X,Y` (bottom-right) coordinates to define the region
- Fills 512×512 quadrants with solid water color `#4A6372`
- By default skips quadrants that already have generation data (use `--overwrite` to replace them)
- `--dry-run` flag shows what would be filled without making changes
- Commits to the database after each row with progress indicators

```shellscript
# Fill quadrants from (0,0) to (5,5) with water
uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5

# Dry run to preview what would be filled
uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --dry-run

# Overwrite existing quadrants with water
uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --overwrite
```

---

**User**

sqlite3.IntegrityError: NOT NULL constraint failed: quadrants.lat

---

**Cursor**

Let me check the database schema and how other scripts handle creating quadrants:

I see - the database schema requires `lat`, `lng`, `tile_row`, `tile_col`, and `quadrant_index` columns. Let me update the script to use placeholder values for water tiles:

```python
"""
Fill quadrants with solid water color.

Fills a rectangular region of quadrants with a solid water color (#4A6372).

Usage:
  uv run python src/isometric_nyc/e2e_generation/fill_water.py <generation_dir> --tl X,Y --br X,Y

Examples:
  # Fill quadrants from (0,0) to (5,5) with water
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5

  # Dry run to see what would be filled
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --dry-run

  # Overwrite existing quadrants
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --overwrite
"""

import argparse
import io
import sqlite3
import sys
from pathlib import Path

from PIL import Image

# Constants
QUADRANT_SIZE = 512
WATER_COLOR = (0x4A, 0x63, 0x72)  # #4A6372


def parse_coordinate(coord_str: str) -> tuple[int, int]:
    """Parse a coordinate string like '10,15' into (x, y) tuple."""
    try:
        parts = coord_str.split(",")
        if len(parts) != 2:
            raise ValueError()
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
        )


def create_water_tile() -> bytes:
    """Create a solid water color tile as PNG bytes."""
    img = Image.new("RGB", (QUADRANT_SIZE, QUADRANT_SIZE), WATER_COLOR)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def quadrant_has_data(conn: sqlite3.Connection, x: int, y: int) -> bool:
    """Check if a quadrant already has generation data."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT generation FROM quadrants
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (x, y),
    )
    row = cursor.fetchone()
    return row is not None and row[0] is not None


def save_water_quadrant(
    conn: sqlite3.Connection, x: int, y: int, png_bytes: bytes
) -> bool:
    """
    Save water tile to a quadrant at position (x, y).

    Creates the quadrant row if it doesn't exist.
    Returns True if successful.
    """
    cursor = conn.cursor()

    # Try to update existing row first
    cursor.execute(
        """
        UPDATE quadrants
        SET generation = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (png_bytes, x, y),
    )

    if cursor.rowcount > 0:
        return True

    # Row doesn't exist, insert it with placeholder values for required fields
    # Water tiles don't need real geo coordinates, so we use dummy values
    cursor.execute(
        """
        INSERT INTO quadrants (quadrant_x, quadrant_y, lat, lng, tile_row, tile_col, quadrant_index, generation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (x, y, 0.0, 0.0, y, x, 0, png_bytes),
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill quadrants with solid water color (#4A6372)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory (contains quadrants.db)",
    )
    parser.add_argument(
        "--tl",
        type=parse_coordinate,
        required=True,
        metavar="X,Y",
        help="Top-left coordinate of the region to fill (e.g., '0,0')",
    )
    parser.add_argument(
        "--br",
        type=parse_coordinate,
        required=True,
        metavar="X,Y",
        help="Bottom-right coordinate of the region to fill (e.g., '5,5')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quadrants (default: skip existing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be filled without actually filling",
    )

    args = parser.parse_args()

    # Resolve paths
    generation_dir = args.generation_dir.resolve()
    tl = args.tl
    br = args.br

    # Validate inputs
    if not generation_dir.exists():
        print(f"❌ Error: Generation directory not found: {generation_dir}")
        return 1

    db_path = generation_dir / "quadrants.db"
    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    # Validate coordinate range
    if tl[0] > br[0] or tl[1] > br[1]:
        print(
            f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
        )
        return 1

    # Calculate total quadrants
    width = br[0] - tl[0] + 1
    height = br[1] - tl[1] + 1
    total = width * height

    print("=" * 50)
    print("🌊 Fill Water Quadrants")
    print("=" * 50)
    print(f"📁 Generation dir: {generation_dir}")
    print(f"📍 Region: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
    print(f"📐 Size: {width}×{height} = {total} quadrants")
    print(f"🎨 Water color: #{WATER_COLOR[0]:02X}{WATER_COLOR[1]:02X}{WATER_COLOR[2]:02X}")
    print()

    if args.dry_run:
        print("🔍 Dry run - no changes will be made")
        print(f"   Would fill {total} quadrants with water color")
        return 0

    # Create water tile
    water_png = create_water_tile()
    print(f"💧 Created water tile: {len(water_png)} bytes")
    print()

    # Fill quadrants
    conn = sqlite3.connect(db_path)
    try:
        filled = 0
        skipped = 0

        for y in range(tl[1], br[1] + 1):
            for x in range(tl[0], br[0] + 1):
                # Check if quadrant already has data
                if not args.overwrite and quadrant_has_data(conn, x, y):
                    skipped += 1
                    continue

                # Save water tile
                save_water_quadrant(conn, x, y, water_png)
                filled += 1

            # Commit after each row
            conn.commit()

            # Progress indicator
            row_num = y - tl[1] + 1
            print(f"   Row {row_num}/{height}: filled {filled}, skipped {skipped}")

        conn.commit()
    finally:
        conn.close()

    print()
    print("=" * 50)
    print("✅ Fill complete!")
    print(f"   Filled: {filled} quadrants")
    print(f"   Skipped: {skipped} quadrants (already had data)")
    print(f"   Total: {total} quadrants")

    return 0


if __name__ == "__main__":
    sys.exit(main())


```

Fixed! The INSERT statement now includes all the required NOT NULL fields (`lat`, `lng`, `tile_row`, `tile_col`, `quadrant_index`) with placeholder values. Water tiles don't need real geo coordinates, so they use `0.0` for lat/lng.

Try running it again:
```bash
uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5
```

```shellscript
uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5
```

---

**User**

no, i'm pretty sure the water quadrants have lat/lng - can you look at the logic in @src/isometric_nyc/e2e_generation/app.py 

---

**Cursor**

```python
@app.route("/api/water-fill", methods=["POST"])
def api_water_fill():
  """API endpoint to fill an entire quadrant with the water color."""

  from PIL import Image

  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation,
    image_to_png_bytes,
    png_bytes_to_image,
    save_quadrant_generation,
  )
```

Looking at the `api_water_fill` function in `app.py`, I can see it uses `save_quadrant_generation` from `shared.py`, which properly handles creating the quadrant with lat/lng via `ensure_quadrant_exists`. Let me update the script to use the same approach:

```python
"""
Fill quadrants with solid water color.

Fills a rectangular region of quadrants with a solid water color (#4A6372).

Usage:
  uv run python src/isometric_nyc/e2e_generation/fill_water.py <generation_dir> --tl X,Y --br X,Y

Examples:
  # Fill quadrants from (0,0) to (5,5) with water
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5

  # Dry run to see what would be filled
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --dry-run

  # Overwrite existing quadrants
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --overwrite
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from PIL import Image

from isometric_nyc.e2e_generation.shared import (
    get_generation_config,
    get_quadrant_generation,
    image_to_png_bytes,
    save_quadrant_generation,
)

# Constants
QUADRANT_SIZE = 512
WATER_COLOR = (0x4A, 0x63, 0x72)  # #4A6372


def parse_coordinate(coord_str: str) -> tuple[int, int]:
    """Parse a coordinate string like '10,15' into (x, y) tuple."""
    try:
        parts = coord_str.split(",")
        if len(parts) != 2:
            raise ValueError()
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
        )


def create_water_image(width: int, height: int) -> Image.Image:
    """Create a solid water color image."""
    return Image.new("RGB", (width, height), WATER_COLOR)


def quadrant_has_data(conn: sqlite3.Connection, x: int, y: int) -> bool:
    """Check if a quadrant already has generation data."""
    gen = get_quadrant_generation(conn, x, y)
    return gen is not None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill quadrants with solid water color (#4A6372)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory (contains quadrants.db)",
    )
    parser.add_argument(
        "--tl",
        type=parse_coordinate,
        required=True,
        metavar="X,Y",
        help="Top-left coordinate of the region to fill (e.g., '0,0')",
    )
    parser.add_argument(
        "--br",
        type=parse_coordinate,
        required=True,
        metavar="X,Y",
        help="Bottom-right coordinate of the region to fill (e.g., '5,5')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quadrants (default: skip existing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be filled without actually filling",
    )

    args = parser.parse_args()

    # Resolve paths
    generation_dir = args.generation_dir.resolve()
    tl = args.tl
    br = args.br

    # Validate inputs
    if not generation_dir.exists():
        print(f"❌ Error: Generation directory not found: {generation_dir}")
        return 1

    db_path = generation_dir / "quadrants.db"
    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    # Validate coordinate range
    if tl[0] > br[0] or tl[1] > br[1]:
        print(
            f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
        )
        return 1

    # Calculate total quadrants
    width = br[0] - tl[0] + 1
    height = br[1] - tl[1] + 1
    total = width * height

    print("=" * 50)
    print("🌊 Fill Water Quadrants")
    print("=" * 50)
    print(f"📁 Generation dir: {generation_dir}")
    print(f"📍 Region: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
    print(f"📐 Size: {width}×{height} = {total} quadrants")
    print(f"🎨 Water color: #{WATER_COLOR[0]:02X}{WATER_COLOR[1]:02X}{WATER_COLOR[2]:02X}")
    print()

    if args.dry_run:
        print("🔍 Dry run - no changes will be made")
        print(f"   Would fill {total} quadrants with water color")
        return 0

    # Create water tile
    water_png = create_water_tile()
    print(f"💧 Created water tile: {len(water_png)} bytes")
    print()

    # Fill quadrants
    conn = sqlite3.connect(db_path)
    try:
        filled = 0
        skipped = 0

        for y in range(tl[1], br[1] + 1):
            for x in range(tl[0], br[0] + 1):
                # Check if quadrant already has data
                if not args.overwrite and quadrant_has_data(conn, x, y):
                    skipped += 1
                    continue

                # Save water tile
                save_water_quadrant(conn, x, y, water_png)
                filled += 1

            # Commit after each row
            conn.commit()

            # Progress indicator
            row_num = y - tl[1] + 1
            print(f"   Row {row_num}/{height}: filled {filled}, skipped {skipped}")

        conn.commit()
    finally:
        conn.close()

    print()
    print("=" * 50)
    print("✅ Fill complete!")
    print(f"   Filled: {filled} quadrants")
    print(f"   Skipped: {skipped} quadrants (already had data)")
    print(f"   Total: {total} quadrants")

    return 0


if __name__ == "__main__":
    sys.exit(main())


```

```python
"""
Fill quadrants with solid water color.

Fills a rectangular region of quadrants with a solid water color (#4A6372).

Usage:
  uv run python src/isometric_nyc/e2e_generation/fill_water.py <generation_dir> --tl X,Y --br X,Y

Examples:
  # Fill quadrants from (0,0) to (5,5) with water
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5

  # Dry run to see what would be filled
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --dry-run

  # Overwrite existing quadrants
  uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5 --overwrite
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from PIL import Image

from isometric_nyc.e2e_generation.shared import (
    get_generation_config,
    get_quadrant_generation,
    image_to_png_bytes,
    save_quadrant_generation,
)

# Constants
QUADRANT_SIZE = 512
WATER_COLOR = (0x4A, 0x63, 0x72)  # #4A6372


def parse_coordinate(coord_str: str) -> tuple[int, int]:
    """Parse a coordinate string like '10,15' into (x, y) tuple."""
    try:
        parts = coord_str.split(",")
        if len(parts) != 2:
            raise ValueError()
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
        )


def create_water_image(width: int, height: int) -> Image.Image:
    """Create a solid water color image."""
    return Image.new("RGB", (width, height), WATER_COLOR)


def quadrant_has_data(conn: sqlite3.Connection, x: int, y: int) -> bool:
    """Check if a quadrant already has generation data."""
    gen = get_quadrant_generation(conn, x, y)
    return gen is not None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill quadrants with solid water color (#4A6372)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory (contains quadrants.db)",
    )
    parser.add_argument(
        "--tl",
        type=parse_coordinate,
        required=True,
        metavar="X,Y",
        help="Top-left coordinate of the region to fill (e.g., '0,0')",
    )
    parser.add_argument(
        "--br",
        type=parse_coordinate,
        required=True,
        metavar="X,Y",
        help="Bottom-right coordinate of the region to fill (e.g., '5,5')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quadrants (default: skip existing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be filled without actually filling",
    )

    args = parser.parse_args()

    # Resolve paths
    generation_dir = args.generation_dir.resolve()
    tl = args.tl
    br = args.br

    # Validate inputs
    if not generation_dir.exists():
        print(f"❌ Error: Generation directory not found: {generation_dir}")
        return 1

    db_path = generation_dir / "quadrants.db"
    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    # Validate coordinate range
    if tl[0] > br[0] or tl[1] > br[1]:
        print(
            f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
        )
        return 1

    # Calculate total quadrants
    width = br[0] - tl[0] + 1
    height = br[1] - tl[1] + 1
    total = width * height

    print("=" * 50)
    print("🌊 Fill Water Quadrants")
    print("=" * 50)
    print(f"📁 Generation dir: {generation_dir}")
    print(f"📍 Region: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
    print(f"📐 Size: {width}×{height} = {total} quadrants")
    print(f"🎨 Water color: #{WATER_COLOR[0]:02X}{WATER_COLOR[1]:02X}{WATER_COLOR[2]:02X}")
    print()

    if args.dry_run:
        print("🔍 Dry run - no changes will be made")
        print(f"   Would fill {total} quadrants with water color")
        return 0

    # Fill quadrants
    conn = sqlite3.connect(db_path)
    try:
        # Get generation config (needed for ensure_quadrant_exists)
        config = get_generation_config(conn)

        # Determine quadrant size from config
        quadrant_width = config.get("width_px", QUADRANT_SIZE) // 2
        quadrant_height = config.get("height_px", QUADRANT_SIZE) // 2

        # Create water image
        water_img = create_water_image(quadrant_width, quadrant_height)
        water_png = image_to_png_bytes(water_img)
        print(f"💧 Created water tile: {quadrant_width}×{quadrant_height}, {len(water_png)} bytes")
        print()

        filled = 0
        skipped = 0

        for qy in range(tl[1], br[1] + 1):
            for qx in range(tl[0], br[0] + 1):
                # Check if quadrant already has data
                if not args.overwrite and quadrant_has_data(conn, qx, qy):
                    skipped += 1
                    continue

                # Save water tile (this handles creating quadrant with proper lat/lng)
                save_quadrant_generation(conn, config, qx, qy, water_png)
                filled += 1

            # Progress indicator
            row_num = qy - tl[1] + 1
            print(f"   Row {row_num}/{height}: filled {filled}, skipped {skipped}")

    finally:
        conn.close()

    print()
    print("=" * 50)
    print("✅ Fill complete!")
    print(f"   Filled: {filled} quadrants")
    print(f"   Skipped: {skipped} quadrants (already had data)")
    print(f"   Total: {total} quadrants")

    return 0


if __name__ == "__main__":
    sys.exit(main())


```

Fixed! The script now uses `save_quadrant_generation` from `shared.py`, which calls `ensure_quadrant_exists` to properly calculate lat/lng from the generation config. This matches the approach used by `api_water_fill` in `app.py`.

Key changes:
1. Uses `get_generation_config` to load the config (needed for lat/lng calculation)
2. Uses `save_quadrant_generation` from shared.py which handles creating quadrants with proper geo coordinates
3. Gets quadrant size from config instead of hardcoding

Try running it again:
```bash
uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5
```

```shellscript
uv run python src/isometric_nyc/e2e_generation/fill_water.py generations/v01 --tl 0,0 --br 5,5
```

