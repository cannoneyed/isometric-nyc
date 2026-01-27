# Rectangle Generation Plan

**STATUS: COMPLETED**

We need to implement an algorithm to generate _all_ quadrants within a specified
rectangle defined by the `tl` and `br` (x,y) coordinates with the following
restrictions.

- For any given rectangle quadrant selection, there may be any number of
  generated quadrants already present within the bounds of the rectangle or
  along the edges of the rectangle.
- Quadrants may be generated in a 2x2, 2x1, 1x2, or 1x1 fashion
- For a 2x2 quadrant generation, no side of the 2x2 tile may touch any
  previously generated quadrants.
- For a 2x1 or 1x2 quadrant generation, both quadrants of the 2-long side must
  touch previously generated quadrants along that axis on _one_ side. Neither of
  the 2 generating quadrants may touch previously generated quadrants along the
  transverse (short) side.
- 1x1 quadrants should ideally be generated with the 3 other quadrants in a 2x2
  tile already generated.

The most efficient algorithm is to start with as many 2x2s as possible in the
given space, then 2x1/1x2s, then fill in the remaining gaps with 1x1s.

This logic has been explored in `make_strip_plan.py`, but I want it to be more
robust. Please design, implemement, and test a new plan that handles any grid
configuration and generates quadrants in the most efficient way possible.

The function will be used by a new `generate_rectangle` function/route in the
`app.py` web app, and should create a sequence of "generation steps" that can be
enqueued in the system.

## Implementation

Implemented in:

- `src/isometric_nyc/generation/make_rectangle_plan.py` - Core algorithm
- `tests/test_make_rectangle_plan.py` - Comprehensive test suite (71 tests)
- `src/isometric_nyc/generation/app.py` - API route
  `/api/generate-rectangle`

### Algorithm

The algorithm follows the pattern from task 019 (strip plan) and works in three
phases:

1. **Phase 1: Place 2x2 tiles** - Greedily places 2x2 tiles where no neighbor
   (in any of the 8 adjacent positions) is previously generated or scheduled.
   This naturally creates a spacing pattern (SS.SS.SS) with 1-column gaps.

2. **Phase 2: Place 2x1/1x2 tiles** - Places tiles in this order:

   - **1x2 (vertical) first**: Bridges between 2x2 tiles (can have generated
     content on both sides, like when bridging two 2x2s)
   - **2x1 (horizontal) second**: Connect to the generation edge
   - Rule: At least one long side must be fully generated, short sides must not
     have generated neighbors

3. **Phase 3: Fill with 1x1 tiles** - Fills remaining gaps with single
   quadrants, prioritizing those where more of the surrounding 2x2 block is
   already generated.

### API Usage

```bash
curl -X POST http://localhost:8080/api/generate-rectangle \
  -H "Content-Type: application/json" \
  -d '{"tl": [0, 0], "br": [5, 5], "model_id": "optional-model"}'
```

Response includes plan summary with step counts by type.
