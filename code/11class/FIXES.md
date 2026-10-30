# Corrections

| Legacy problem | Corrected behavior |
|---|---|
| Appliance name burned into PNG | pure RGB array saved with Pillow |
| Whole PNG resized into CNN | image contains numerical Gramian pixels only |
| Image-level random split | source CSV sample grouped split |
| Source recording lost | `source_id` stored for every generated image |
| 55,000-image copy fallback | removed; samples are selected from real cycles |
| Summation field called GADF | standard 11-class pipeline fixed to GASF |

The historical 11-class training result cannot be reported as leakage-free appliance classification performance.
