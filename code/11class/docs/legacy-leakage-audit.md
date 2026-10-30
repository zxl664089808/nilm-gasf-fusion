# Legacy 11-Class Leakage Audit

## Confirmed title leakage

The legacy image generator contains:

```python
plt.title(f"{appliance_type} - RGB GADF (Cycle {i+1})")
plt.savefig(image_path, dpi=150, bbox_inches="tight")
```

The title includes the appliance class name. The training loader reads the full image:

```python
image = Image.open(self.image_paths[idx]).convert("RGB")
transforms.Resize((64, 64))
```

Therefore text pixels are model input. The evidence image is available at
`docs/evidence/legacy-title-band.png`.

## Confirmed source overlap

The legacy training script uses `train_test_split` on generated image paths.
Multiple adjacent cycles originate from the same CSV recording, so they can
appear in both training and test sets.

## Missing provenance

The saved `*_fundamental_cycles.pkl` files contain only voltage/current cycle
arrays. They do not retain which CSV sample generated each cycle. Grouped
evaluation therefore requires re-extraction from the source CSVs, which this
project performs.
