# GTA VI trophy rarity forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/funtoons/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/funtoons/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun competition gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy, from the house's recorded reads up to the deadline.

## Run

```sh
python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `/output/<id>.out` (or `$MAKEFUN_OUTPUT_DIR`). Python 3.13, standard library only, no network, deterministic.

## Model

A trophy's rate is earners over players tracked. Earners only grow; players tracked grows as Exophase discovers more players, so a rate can **fall** when tracking outgrows earning (dilution — observed on launch night, when a refresh took players 638 → 3868 and almost every rate down). So per trophy the program fits, on the last 24h of reads with exponential recency weights (half-life 12h):

- (a) log earners vs log(1 + hours since release),
- (b) log players tracked vs log(1 + hours since release),
- (c) logit(rate) vs log(1 + hours since release) directly,

and forecasts the rate as the average in logit space of (a)/(b) and (c). Fits run on **distinct plateaus**, not on every read: between Exophase refreshes the page is frozen, so repeats carry no information and a long flat stretch must not drag slopes to zero.

Spreads are honest about refresh shocks: a 0.5 logit floor, plus a random-walk term in the horizon and the fit residual, with a longer downside tail than upside tail because refreshes drop rates. A trophy every tracked player holds (100% at the checkpoint) keeps a ~100% median with a long downside tail.

Structural priors on medians: chapter trophies decrease along the story (Ch1 ≥ Ch2 ≥ Ch3 ≥ Ch4 ≥ final mission), the platinum is at most the rarest trophy and at most the 100%-completion trophy, and 100% completion is at most the final mission. Ceilings (platinum ≤ 10%, 100%-completion ≤ 12%, grind trophies ≤ 25%) cap extreme extrapolation only.

## Local validation

- Practice cases p00/p01: valid, deterministic (byte-identical across runs), ~0.2s per case, pinball-logit loss **0.8455** vs the house persistence baseline's ~1.51 (answers reconstructed from the 04:30Z refresh read).
- Synthetic final-like tests (2-day, 10-minute reads from a power-law earners / growing-players ground truth with refresh jumps, held-out truth at +4.52 days): beats the persistence baseline on both scenarios (0.9530 vs 1.5519; 0.2803 vs 0.2911) and the trend baseline on the first (0.9530 vs 1.0679).
- Edge cases (single-read history, 0% and 100% trophies, null rates, empty history) all produce valid output.

## License

MIT — see [LICENSE](LICENSE).
