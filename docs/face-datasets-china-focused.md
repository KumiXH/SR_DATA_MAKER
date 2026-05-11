# China-Focused Face Datasets for SR Data Maker

[中文版本 / Chinese Version](face-datasets-china-focused.zh-CN.md)

This note summarizes face datasets that are practical for this repository's face super-resolution, restoration, and degradation-teacher workflows, with emphasis on Chinese-majority or clearly East-Asian-heavy data sources.

## Recommendation Summary

| Dataset | Chinese / East-Asian relevance | Access | Best fit in this project | Priority |
| --- | --- | --- | --- | --- |
| `SCUT-FBP5500` | Strong; includes `2000 Asian females + 2000 Asian males` | Public academic-style release | Primary clean HR face source, alignment, degradation synthesis, teacher inference | Very high |
| `Tsinghua-FED` | Strong; Chinese subjects | Paper / academic access | High-quality HR reference set, validation set, showcase samples | Very high |
| `CAS-PEAL-R1` | Very strong; explicitly a Chinese face database | Official request / research access | Controlled evaluation for pose, illumination, occlusion, robustness checks | High |
| `SCUT-FBP` | Strong; Asian female subset | Public academic paper release | Small clean aligned-face supplement, pipeline smoke testing | Medium-high |
| `SZU-EmoDage` | Medium; useful mostly as synthetic augmentation | Paper / OSF | Distribution augmentation only, not primary ground truth | Medium |
| `DataTang` commercial sets | Medium to high depending on SKU | Commercial purchase | Scale expansion after public seed datasets are exhausted | High if budget exists |

## Dataset Notes

### `SCUT-FBP5500`

- Best public starting point for this repository.
- Useful as a clean HR pool for aligned face extraction, synthetic degradation generation, and face-teacher inference comparisons.
- Limitation: mostly frontal and relatively clean, so it does not cover enough in-the-wild diversity by itself.
- Sources:
  - [SCUT-FBP5500 GitHub release](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)
  - [SCUT-FBP5500 paper](https://arxiv.org/abs/1801.06345)

### `Tsinghua-FED`

- Strong candidate for high-quality HR references.
- Reported in the paper as high-resolution face-expression data, useful for premium HR supervision targets, validation samples, and qualitative comparisons across teacher models.
- Limitation: small scale and typically academic-access oriented.
- Source:
  - [Tsinghua-FED paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0231304)

### `CAS-PEAL-R1`

- Strong Chinese-face benchmark with controlled capture conditions.
- Valuable for pose robustness checks, illumination robustness checks, and occlusion evaluation.
- Limitation: too controlled to serve as the only training source.
- Sources:
  - [CAS-PEAL-R1 official site](https://www.jdl.link/peal/home.htm)
  - [Request information page](https://www.jdl.link/peal/Request%20Information.htm)

### `SCUT-FBP`

- Small but clean aligned-face dataset.
- Useful for first-pass pipeline debugging, YAML examples, and smoke tests for face-teacher integration.
- Source:
  - [SCUT-FBP paper](https://arxiv.org/abs/1511.02459)

### `SZU-EmoDage`

- Better treated as an auxiliary or augmentation-oriented source than as primary ground truth.
- Use it if the goal is to broaden expression or distribution patterns after the core real-photo datasets are already in place.
- Sources:
  - [Metadata record](https://d-nb.info/1321435401/34)
  - [OSF project](https://osf.io/7a5fs/)

### `DataTang` commercial options

- Practical when the project needs far more scale than public Chinese-majority datasets can offer.
- Treat these as licensed production inputs, not default public baselines.
- Caveat: "East Asian" or "yellow race" product wording is not identical to "Chinese faces"; confirm sample composition before procurement.
- Sources:
  - [DataTang dataset 173](https://datatang.com/dataset/173)
  - [DataTang dataset 1199](https://datatang.com/dataset/1199)

## Practical Selection Strategy

For this repository, a pragmatic data plan is:

1. Start with `SCUT-FBP5500` and `SCUT-FBP` to stabilize the pipeline.
2. Add `Tsinghua-FED` as a higher-quality HR supplement.
3. Use `CAS-PEAL-R1` mainly for evaluation and controlled ablations.
4. Add licensed commercial data only when public sources are no longer enough.

## Important Caveat

There is no single perfect, large, fully open, clearly Chinese-majority, super-resolution-ready face dataset that covers both quality and scale. In practice, production-grade training data for this project will likely require a mix of:

- public academic seed datasets
- self-collected or partner-provided data
- licensed commercial expansion data
