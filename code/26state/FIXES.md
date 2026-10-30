# Corrections relative to the legacy project

## Confirmed leak

- Legacy generated PNGs contained class/state text in the top title.
- Legacy training read the whole PNG.
- The text was therefore part of the model input.

## Changes

| Problem | Legacy behavior | Corrected behavior |
|---|---|---|
| Label text in pixels | `plt.title` + `savefig` | `Image.fromarray(...).save` |
| Image-level split | random image split | source-recording grouped split |
| Lost source IDs | cycles stored without source | manifest stores `source_id` |
| Misnamed field | summation called GADF | explicit `gasf`/`gadf` |
| Copy fallback | duplicated images to reach 55,000 | no duplication |
| Missing provenance | manifest absent | CSV for every image |

## Scientific consequence

The old 99.67% and 99.96% results cannot be reported as waveform-only generalization performance. Re-run the corrected dataset and grouped evaluation before using metrics in a manuscript.
