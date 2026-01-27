# Strip plan

Create a script in `src/isometric_nyc/generation` called
`make_strip_plan.py` that takes a few parameters:

- generation_dir - the directory containing the generation db
- tl - the "(x,y)" coordinates of the top left of the strip to generate
- br - the "(x,y)" coordinates of the bottom right of the strip to generate

The script should create a json file in the generation dir called
"generate*strip*<tl>\_<br>.json" that follows the schema defined below:

```
[{
  "quadrants": "(x,y),(x,y)",
  "status": "pending"|"done"|"error"
}, ...]
```

Where the quadrants is a string representing a list of x,y tuples of the strip
to generate.

The algorithm for generation is as follows:

1. First, determine the "generation edge" of the strip - the generation edge is
   defined as the edge of the quadrant rectangle on which all exterior
   neighboring quadrants are generated. If no edge is found, then throw an
   error.

2. Once the generation edge has been found determine the direction of progress -
   the direction of progress will run along the edge from left to right (if
   horizontal) or top to bottom (if vertical)

3. Next determine the "depth" of the strip - this is defined as the height for a
   horizontal rectangle or the width for a vertical rectangle, ie the number of
   grid quadrants perpendicular to the direction of progress.

4. If the "depth" is 1, then follow the following formula. For the following
   grid of generated tiles with a tl=(0,0) and br=(0,10)

(G means Generated, x means empty, and S means selected)

Generate 2x1 quadrants with a 1 quadrant gap, e.g.

```step 1 - (0,0),(1,0)
S S x x x x x x x x x
G G G G G G G G G G G
```

```step 2 - (3,0),(4,0)
G G x S S x x x x x x
G G G G G G G G G G G
```

```step 3 - (6,0),(7,0)
G G x G G x S S x x x
G G G G G G G G G G G
```

```step 4 - (9,0),(10,0)
G G x G G x G G x S S
G G G G G G G G G G G
```

If there's an "overhang" and a 2x1 can't be selected, then continue to the next
step.

Once all of the 2x1 quadrants have been generated, generate the missing 1
quadrant slots from start to finish, e.g.

```step 5 - (2,0)
G G S G G x G G x G G
G G G G G G G G G G G
```

```step 5 - (5,0)
G G G G G S G G x G G
G G G G G G G G G G G
```

```step 5 - (8,0)
G G G G G G G G S G G
G G G G G G G G G G G
```

In this way, we'll generate the next "row" or "column" of quadrants.

5. For a 2 deep strip, follow the same algorithm but twice, once on the first
   1-quadrant strip and twice on the second.

6. For a 3 deep strip, we can be more efficient by generating 2x2 quadrant
   tiles. For example, for the following grid of generated tiles with a tl=(0,0)
   and br=(2,7)

```Step 1 - (0,0),(1,0),(0,1),(1,1)
S S x x x x x x
S S x x x x x x
x x x x x x x x
G G G G G G G G
```

```Step 2 - (3,0),(4,0),(3,1),(4,1)
G G x S S x x x
G G x S S x x x
x x x x x x x x
G G G G G G G G
```

```Step 3 - (6,0),(7,0),(6,1),(7,1)
G G x G G x S S
G G x G G x S S
x x x x x x x x
G G G G G G G G
```

Once the 2x2 tiles (one away from the generated edge) are generated, then we
generate the 1x2 "bridge" quadrants between them:

```Step 4 - (2,0),(2,1)
G G S G G x G G
G G S G G x G G
x x x x x x x x
G G G G G G G G
```

```Step 5 - (5,0),(5,1)
G G G G G S G G
G G G G G S G G
x x x x x x x x
G G G G G G G G
```

Then we generate the 2x1 "bridge" quadrants back towards the generated edge,
ensuring we keep a 1 quadrant gap between generations

```Step 6 - (0,2),(1,2)
G G G G G G G G
G G G G G G G G
S S x x x x x x
G G G G G G G G
```

```Step 7 - (3,2),(4,2)
G G G G G G G G
G G G G G G G G
G G x S S x x x
G G G G G G G G
```

```Step 8 - (6,2),(7,2)
G G G G G G G G
G G G G G G G G
G G x G G x S S
G G G G G G G G
```

Once all bridges are generated, then we fill in the missing gaps

```Step 9 - (2,2)
G G G G G G G G
G G G G G G G G
G G S G G x G G
G G G G G G G G
```

```Step 10 - (5,2)
G G G G G G G G
G G G G G G G G
G G G G G S G G
G G G G G G G G
```

7. If the strip is more than 3 deep, then do the first 3 using the formula for
   3-deep then continue outward using the formula for however more quadrants
   remain.

---

Please generate this script and add a small test suite.
