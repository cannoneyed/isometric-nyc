# Generation App

Let's wire up generation to the view_generations.py web app! If the user has one
or more quadrants selected, they can hit "generate" button to send a request to
the python app. For quadrant-based generation, there are a number of rules to
follow:

(G means Generated, x means empty, and S means selected)

1. An isolated generation - this is four adjacent square-shaped quadrants that
   have no neighbors. For example:

```
x x x x x x
G G G x S S
G G G x S S
x x x x x x
```

The above generation is LEGAL.

2. A missing quadrant, with no neighbors:

```
x x x x
G G G x
G G S x
x x x x
```

The above generation is LEGAL.

3. A missing half, with no neighbors:

```
x x x x
G G S x
G G S x
x x x x
```

The above generation is LEGAL.

4. A missing "middle", with no opposite side neighbors

```
x x x x x
G G S G G
G G S G G
x x x x x
```

The above generation is LEGAL.

---

For all LEGAL generations, the key to the formula is that we can fit all of the
content to be generated in one 4x4 quadrant square, and there's no generated
content from adjacent tiles that border the new generated content. This is
because unless content is generated either a) from scratch (no neighbors) or b)
by infilling/inpainting existing content, there may be "seams" in the generated
quadrants from the pixels being generated slightly differently.

--- Example ILLEGAL Generations:

1. A whole tile, with neighbors

```
x x x x x
G G S S x
G G S S x
G G x x x
```

Because the quadrants to be generated don't contain any existing generation and
touch existing generated pixels, the generated pixels may not perfectly match
and a seam may be generated, making this generation ILLEGAL.

2. A quadrant with neighbors

```
x x x x x
G G x x x
G S x x x
G G x x x
x x x x x
```
