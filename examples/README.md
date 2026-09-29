# Synthetic demonstration data

`synthetic_specimens.csv` was generated for this repository with a fixed random
seed. All specimen IDs start with `SYN-`, and every row has
`sample_type=synthetic_demo`. Geometry, material-property and response values
are artificial; they do not represent the user's MMB experiments or support a
scientific conclusion. The target column name `Gc_raw` is retained solely so
the existing feature and validation functions can be exercised.
