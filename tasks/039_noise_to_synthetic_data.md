# Synthetic Data Noise

We want to add a new parameter to the `src/isometric_nyc/synthetic_data/create_omni_dataset.py` script to apply to terrain:

- `gamma_shift`: Darken the masked/rendered region of the template image (float 0 to 1)

`desaturation`: Desaturate the masked/rendered region (float 0 to 1)

`noise`: Add noise to the masked/rendered region (float 0 to 1)

The order of application should be noise -> desaturation -> gamma_shift.

We need these to be both parameters that can be added to *all* examples via the CLI flags, or as columns in an optional csv for each individual example.


