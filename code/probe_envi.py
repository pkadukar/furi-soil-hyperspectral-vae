import rasterio, glob

paths = sorted(glob.glob("/scratch/mkhorram/Soil/VNIR2/*.bip"))
assert paths, "No VNIR .bip files found!"
p = paths[0]
print("Reading:", p)

with rasterio.open(p) as src:
    arr = src.read()  # (Bands, H, W)
    print("Shape (Bands, Height, Width):", arr.shape)
    print("Data type:", arr.dtype)
    print("Min/Max:", arr.min(), arr.max())
