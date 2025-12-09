# Automated Generation

The following plan outlines a script that will automatically generate tiles in
an optimum fashion given a bounding box and existing tiles in the generations
db.

The script should take the following params:

- generation_dir - the directory containing the generation config

- top-left - an (x,y) tuple string that determines the top left of the bounding
  box

- bottom-right - an (x,y) tuple string that determines the bottom right of the
  bounding box

The script should then create a virtual in-memory grid of all x,y coordinates
within the bounding box (inclusive), and then look up which quadrants in the
grid have already been generated (looking in the sqlite db). The script should
then construct an optimum generation plan using the following logic:

- Start from the "center" - there should already be a vaguely rectangular shape
  that's already been generated - rather than starting from the corner, we'll
  want to generate outward from the middle (around what's already been
  generated).

- The most efficient generation is as follows:

* Generate four quadrants that are one quadrant away from the center rectangle.
  Generate these tiles on a one-quadrant offset, extending all the way to the
  center generated rectangle.

* Next, connect the offset tiles to the center rectangle by generating the two
  quadrants in between the tile and the center rectangle.

* Next, connect the offset tiles by generating the two quadrants in between the
  tile and its neighbor

* Finally, generate the single remaining tiles next to the the corner of the
  tiles between the tiles and the center rectangle.

Here's an example of the optimal algorithm:

(G means Generated, x means empty, and S means selected)

Step 1 - offset four-quadrant tile

```(sample grid of the quadrants)
x S S x x x x x x
x S S x x x x x x
x x x x x x x x x
G G G G G G G G G <- already generated
G G G G G G G G G
```

Step 2 - next offset four-quadrant tile

```
x G G x S S x x x
x G G x S S x x x
x x x x x x x x x
G G G G G G G G G
G G G G G G G G G
```

Step 3 - next offset four-quadrant tile

```
x G G x G G x S S
x G G x G G x S S
x x x x x x x x x
G G G G G G G G G
G G G G G G G G G
```

Step 4 - first "bridge" to the center block

```
x G G x G G x G G
x G G x G G x G G
x S S x x x x x x
G G G G G G G G G
G G G G G G G G G
```

Step 5 - second "bridge" to the center block

```
x G G x G G x G G
x G G x G G x G G
x G G x S S x x x
G G G G G G G G G
G G G G G G G G G
```

Step 6 - third "bridge" to the center block

```
x G G x G G x G G
x G G x G G x G G
x G G x G G x S S
G G G G G G G G G
G G G G G G G G G
```

Step 8 - first bridge off of the tiles

```
S G G x G G x G G
S G G x G G x G G
x G G x G G x G G
G G G G G G G G G
G G G G G G G G G
```

Step 9 - second bridge between the tiles

```
G G G S G G x G G
G G G S G G x G G
x G G x G G x G G
G G G G G G G G G
G G G G G G G G G
```

Step 10 - third bridge between the tiles

```
G G G G G G S G G
G G G G G G S G G
x G G x G G x G G
G G G G G G G G G
G G G G G G G G G
```

Step 10 - first gap

```
G G G G G G G G G
G G G G G G G G G
S G G x G G x G G
G G G G G G G G G
G G G G G G G G G
```

Step 10 - second gap

```
G G G G G G S G G
G G G G G G S G G
G G G S G G x G G
G G G G G G G G G
G G G G G G G G G
```

Step 11 - third gap

```
G G G G G G S G G
G G G G G G S G G
G G G G G G S G G
G G G G G G G G G
G G G G G G G G G
```

---

We'll gradually extend the rectangle in the following spiral pattern: top,
right, bottom, left, until we generate all of the quadrants in the entire
bounding box.

This algorithm implies that we have a rectangle in the center to extend. In
order to get that rectangle, we'll need to fill in all missing quadrants from
whatever shape exists in the middle. This is fairly comples, so I'll leave it to
you to figure out how to do it in a reasonably efficient manner given the
constraints of the quadrant-based generation system enumerated in infill
template.py.

Ensure that the script first and foremost has a dry run mode so we can test and
validate the plan, then once that's done we'll wire it up to the logic in
`generate_tile_omni.py` to automatically generate the quadrants!
