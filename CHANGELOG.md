# Changelog

## [0.8.1](https://github.com/popellab/maple/compare/v0.8.0...v0.8.1) (2026-07-23)


### Bug Fixes

* **cal:** bounded-observable logit_normal validator + n_biological_is_floor prompt ([#76](https://github.com/popellab/maple/issues/76)) ([4a65226](https://github.com/popellab/maple/commit/4a6522661cda586657c542aabeb8d28c923520e0))
* migrate extraction lit-search agent to pydantic-ai 2.x ([#72](https://github.com/popellab/maple/issues/72)) ([5fca667](https://github.com/popellab/maple/commit/5fca667f934c59d359a1cbee9394d47e10d3525e))
* **submodel:** allow 1.645 as an observation_code constant ([#75](https://github.com/popellab/maple/issues/75)) ([15d0e3e](https://github.com/popellab/maple/commit/15d0e3e50254b2e02d0e482311c21033ba7bc0aa))

## [0.8.0](https://github.com/popellab/maple/compare/v0.7.1...v0.8.0) (2026-07-23)


### ⚠ BREAKING CHANGES

* explicit observable time-series reduction; drop vector index ([#70](https://github.com/popellab/maple/issues/70))

### Features

* explicit observable time-series reduction; drop vector index ([#70](https://github.com/popellab/maple/issues/70)) ([0d83bbb](https://github.com/popellab/maple/commit/0d83bbbcc262b6265a202a20d7cce4052ca12b30))
* unit_group panels + logit_normal shape for population spread ([#62](https://github.com/popellab/maple/issues/62)) ([3013dcd](https://github.com/popellab/maple/commit/3013dcdc325ed57d22cf2e18cadfb59fa40c4838))


### Bug Fixes

* defer model-structure CalibrationTarget validators when no validation context ([#64](https://github.com/popellab/maple/issues/64)) ([6454612](https://github.com/popellab/maple/commit/6454612170937773d62a77822d5b14f2d468fd45))
* harden staged-extraction pipeline (PDF fetch, timeouts, DOI matching) ([#60](https://github.com/popellab/maple/issues/60)) ([41fcca9](https://github.com/popellab/maple/commit/41fcca97478c600f544947f1f034c7d74876e07e))
* run context-free observable signature check in the agent loop ([#65](https://github.com/popellab/maple/issues/65)) ([5be741e](https://github.com/popellab/maple/commit/5be741e94c7b8e7f0eb1197de5b80e04ce0dc8af))
* unit_group allows unbalanced n_biological across members ([#63](https://github.com/popellab/maple/issues/63)) ([648d5bf](https://github.com/popellab/maple/commit/648d5bf064b45570d6477e52410be0d4c689e7a9))

## [0.7.1](https://github.com/popellab/maple/compare/v0.7.0...v0.7.1) (2026-07-08)


### Documentation

* document FigureExcerpt/TableExcerpt schemas and denominator requirement in cal-target prompt ([#32](https://github.com/popellab/maple/issues/32)) ([0c1b6ac](https://github.com/popellab/maple/commit/0c1b6ac355d71bbeb13be17adda86ea1a432274a))

## [0.7.0](https://github.com/popellab/maple/compare/v0.6.0...v0.7.0) (2026-07-08)


### Features

* observed_distribution native moments form + observation_code center-only convention ([#53](https://github.com/popellab/maple/issues/53)) ([bfd4401](https://github.com/popellab/maple/commit/bfd44016124e833da186085ca8e3b81d83a9fd97))
* shared quantile-anchor variability schema (center vs population spread) ([#52](https://github.com/popellab/maple/issues/52)) ([1d7cb00](https://github.com/popellab/maple/commit/1d7cb00bf2092dc5724feeb3bfae8c75c8df2acd))


### Bug Fixes

* repair release-please config so releases actually cut ([#56](https://github.com/popellab/maple/issues/56)) ([7462348](https://github.com/popellab/maple/commit/7462348fa771804fa0815200eb7063c1909440bd))


### Documentation

* add workflow schematic and tighten README prose ([#55](https://github.com/popellab/maple/issues/55)) ([bbe3f0f](https://github.com/popellab/maple/commit/bbe3f0ff6dae220c660117143dad0ddc6d5e095f))
