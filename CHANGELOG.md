# Changelog

## [0.2.0](https://github.com/ricardogr07/market-lab/compare/marketlab-v0.1.0...marketlab-v0.2.0) (2026-04-10)


### Features

* add benchmark-relative comparison reporting ([#48](https://github.com/ricardogr07/market-lab/issues/48)) ([84e4c18](https://github.com/ricardogr07/market-lab/commit/84e4c183373abc8b07ff707367fe8438b11fd6e9))
* add black-litterman config surface ([518229e](https://github.com/ricardogr07/market-lab/commit/518229efd4f3b81dd8228f568acb653dd41f2aea))
* add canonical phase 5 scenario configs ([c5a061b](https://github.com/ricardogr07/market-lab/commit/c5a061b6adce987e876f4caa3d05e6668754ef2e))
* add config-defined allocation baselines ([#45](https://github.com/ricardogr07/market-lab/issues/45)) ([41515c1](https://github.com/ricardogr07/market-lab/commit/41515c11b57342c16b86f045e610d804f231fa2a))
* add cost sensitivity config and analytics ([bb08cc3](https://github.com/ricardogr07/market-lab/commit/bb08cc3ce35989391cf5f0d0a9c96ad55e87bb0c))
* add embargo-aware fold diagnostics builder ([5f2fb73](https://github.com/ricardogr07/market-lab/commit/5f2fb7364d7f631aedc9d2ea4453b3d453a39cac))
* add exposure and concentration caps to ranking strategies ([#46](https://github.com/ricardogr07/market-lab/issues/46)) ([abf5516](https://github.com/ricardogr07/market-lab/commit/abf551629ddac615ef26a21437fb720ae2637902))
* add exposure-aware analytics and reporting ([#47](https://github.com/ricardogr07/market-lab/issues/47)) ([e028147](https://github.com/ricardogr07/market-lab/commit/e02814709f39d0fad5a409343e9240738693ba06))
* add factor-model config seam ([f527e46](https://github.com/ricardogr07/market-lab/commit/f527e46fe7f57e4aaf3a3e84718155d76fef4af4))
* add lightweight sklearn comparison models ([40a223b](https://github.com/ricardogr07/market-lab/commit/40a223bad2c6ea69ca91037f87d0a3f837e6a682))
* add mean-variance optimizer config and solver ([84a4533](https://github.com/ricardogr07/market-lab/commit/84a45330d5e2b1ffae100d08ecc819cb751b411c))
* add optimizer input and covariance scaffolding ([#50](https://github.com/ricardogr07/market-lab/issues/50)) ([2fd54ae](https://github.com/ricardogr07/market-lab/commit/2fd54aed43e0bfe6b600804a9ea5a42b75669d3d))
* add ranking strategy modes ([#30](https://github.com/ricardogr07/market-lab/issues/30)) ([4294b80](https://github.com/ricardogr07/market-lab/commit/4294b80d948a37a849778a2c86b42765f1e8cc41))
* add ranking-aware fold evaluation builders ([e5a126c](https://github.com/ricardogr07/market-lab/commit/e5a126c4d0c6d16178e8e71066deb0559e0a8fad))
* add risk diagnostics artifacts ([baa909d](https://github.com/ricardogr07/market-lab/commit/baa909d3cbd0f38ec2f82b9ce850f93eef31e651))
* add single-symbol VOO long-only evaluation support ([6981c7d](https://github.com/ricardogr07/market-lab/commit/6981c7d79d3a52c391d5e6ad29fe83051f4d38b3))
* add walk-forward guardrail config and template defaults ([4715291](https://github.com/ricardogr07/market-lab/commit/47152917d04194a2df6f0b97b298851e5cb5a033))
* expose phase 5 scenarios as packaged templates ([138784a](https://github.com/ricardogr07/market-lab/commit/138784a2b0c6fd0554d1d47e5bab83fe642a82b6))
* extend summaries with ranking-aware and downside metrics ([06817a5](https://github.com/ricardogr07/market-lab/commit/06817a576c23ac5d73f63b108f9b9e866c0f6157))
* implement black-litterman optimizer baseline ([dae1f3e](https://github.com/ricardogr07/market-lab/commit/dae1f3eac061d9be9e69b2b74d26127b47e8c71a))
* persist cost sensitivity artifacts and report section ([d9d99da](https://github.com/ricardogr07/market-lab/commit/d9d99da78e2f4b24ba61986cb31a96bd5e402475))
* persist fold diagnostics across training and experiment runs ([2756e02](https://github.com/ricardogr07/market-lab/commit/2756e02216700c64fd8358a1b3430ab26930ed69))
* persist ranking diagnostics in experiment artifacts and report headlines ([c5cfa64](https://github.com/ricardogr07/market-lab/commit/c5cfa64b7a2c6299bb4cf6164de17d6902e2f4f7))
* surface walk-forward diagnostics in reporting ([b53b273](https://github.com/ricardogr07/market-lab/commit/b53b2735aa0667d1f24af0b7e42c21aa6b8f5a1e))
* wire mean-variance baseline into experiment flows ([8317591](https://github.com/ricardogr07/market-lab/commit/8317591fbb007298984f7e9046bd71286465b872))
* wire risk parity baseline into baseline flows ([097f095](https://github.com/ricardogr07/market-lab/commit/097f0955182c94040c03a325e0a452b4c82b6e07))


### Bug Fixes

* correct mean-variance group cap expectation ([b31ccf8](https://github.com/ricardogr07/market-lab/commit/b31ccf857087ed75a3d92019ed4ca380eb6fcb85))
* handle non-positive cost sensitivity equity ([22aa252](https://github.com/ricardogr07/market-lab/commit/22aa252b1ba4d2dbdcf88c4d90eab2295b28e2cd))
* preserve empty mean-variance baselines as cash ([ab0b01f](https://github.com/ricardogr07/market-lab/commit/ab0b01fb3455d4827d100b96dd242242d4ce82d5))
* retain active covariance window at oos start ([80f9cee](https://github.com/ricardogr07/market-lab/commit/80f9cee3f6d638ff4182e0fb3a1af5588098a851))
* validate optimizer input paths before empty windows ([91c9f82](https://github.com/ricardogr07/market-lab/commit/91c9f827a2564a6ebc4071c5bf96b84368eb721b))


### Documentation

* add phase 5 comparison guide ([2487962](https://github.com/ricardogr07/market-lab/commit/2487962de8f224bbd8310246f6d036add6dc1bf5))
* consolidate public docs under docs ([ad7abb4](https://github.com/ricardogr07/market-lab/commit/ad7abb4b31d19128a9dc9ec87f38ae6694a3bdc7))
* document black-litterman baseline semantics ([90e9c80](https://github.com/ricardogr07/market-lab/commit/90e9c8038200388cb8763c17787288c86e19f44a))
* document cost sensitivity diagnostics ([c6abf23](https://github.com/ricardogr07/market-lab/commit/c6abf23299b5c8b565b4c72faea9964f2dd82276))
* document factor and covariance diagnostics ([fd49c1a](https://github.com/ricardogr07/market-lab/commit/fd49c1aa8d1968787f8fd076544a0a39c044fd69))
* document ranking-aware evaluation artifacts ([73c5ac9](https://github.com/ricardogr07/market-lab/commit/73c5ac96a6e1a17bd6b5b7b787e6a890db3c6dbf))
* document the mean-variance baseline ([3594a8b](https://github.com/ricardogr07/market-lab/commit/3594a8bc1c50dc258f93fc8e31e84b8851aabeb5))
* document the risk parity baseline ([87d6306](https://github.com/ricardogr07/market-lab/commit/87d6306b0aa1ac97de031eeb3bacdeed8dbbb4af))
* refresh project plan for phase 4 roadmap ([110b9fc](https://github.com/ricardogr07/market-lab/commit/110b9fc858b7c9f7358d2013805e963bc0eb6f91))

## 0.1.0 (2026-03-29)


### Features

* activate train-models command ([2a5d53a](https://github.com/ricardogr07/market-lab/commit/2a5d53a559e2146cda67921d89444d4c9b16d85a))
* add baseline features strategies and backtest engine ([2f41bcc](https://github.com/ricardogr07/market-lab/commit/2f41bccd369cbd301be88c2a40ccbe8002a309da))
* add config loading and market panel normalization ([ead14e9](https://github.com/ricardogr07/market-lab/commit/ead14e94a110f7d7e9531f1b74db1da44b0195ff))
* add experiment pipeline cli and reporting ([4e2ed55](https://github.com/ricardogr07/market-lab/commit/4e2ed55c9f1e11027cb794bc59583669c693deb1))
* add fold and model summary outputs ([2a8d232](https://github.com/ricardogr07/market-lab/commit/2a8d232110e6b50ea8681dc7bfbda8392c90db64))
* add fold metadata helpers ([8a4333e](https://github.com/ricardogr07/market-lab/commit/8a4333e919fddaa1729e160924678596bd377d2b))
* add github actions pr validation workflow ([d8960b9](https://github.com/ricardogr07/market-lab/commit/d8960b935debd72cbdbcfc7f8e9f0bdf782cf080))
* add GitHub release asset upload and PyPI publish job ([b010a0d](https://github.com/ricardogr07/market-lab/commit/b010a0d01a403621d9c66a6da997e82e8df4bc46))
* add local launcher and real-data smoke workflow ([7b604c7](https://github.com/ricardogr07/market-lab/commit/7b604c79a171f9c59c9a43ae107a8f0c6afc14af))
* add local tox preflight gate ([20fca5a](https://github.com/ricardogr07/market-lab/commit/20fca5a5cfc4bdd75ddde99140bac5545fb56912))
* add manual Docker runner workflow ([ff92d7c](https://github.com/ricardogr07/market-lab/commit/ff92d7c752e860be404f670f6b983a99caefb4a6))
* add mkdocs site and tox preflight envs ([c8e4ae6](https://github.com/ricardogr07/market-lab/commit/c8e4ae687b66b7391217750c67effa23c8bde029))
* add model registry for configured estimators ([7b1307d](https://github.com/ricardogr07/market-lab/commit/7b1307ddf38acffd8803d7a1c0a1c4ad6b3abf4d))
* add phase 3 ci foundation ([f66d897](https://github.com/ricardogr07/market-lab/commit/f66d897640ddd60b43cbdb54b9b0f613fd1ab2f6))
* add ranking strategy from model scores ([1d174d1](https://github.com/ricardogr07/market-lab/commit/1d174d190f21fbd76c43df995c5f620df5c5a079))
* add release-please workflow and config ([57813a5](https://github.com/ricardogr07/market-lab/commit/57813a576e6e52136586a0fbcba5a8ac82c95a33))
* add shared weekly rebalance helpers ([d4e063b](https://github.com/ricardogr07/market-lab/commit/d4e063b56fd9308441cddecbdc8d862b7d5c459d))
* add strategy analytics artifact builders ([3d530d8](https://github.com/ricardogr07/market-lab/commit/3d530d85331b39ce8bd5cf14e681e109dc9a1d74))
* add walk-forward fold engine ([4afe037](https://github.com/ricardogr07/market-lab/commit/4afe0376d0da76909465a404f4a4b2d22db32a3d))
* add walk-forward fold generator ([17de9d2](https://github.com/ricardogr07/market-lab/commit/17de9d21dcab4776c5e61ea4f712a1bc7f87b6df))
* add weekly target dataset builders ([92bba13](https://github.com/ricardogr07/market-lab/commit/92bba1351162606b328c0ad7168afd5c51a876e6))
* add weekly targets and modeling dataset ([948b0cf](https://github.com/ricardogr07/market-lab/commit/948b0cf2af25e89d6695a3bb0c827f51b6aaf6c1))
* bundle example configs for installed cli use ([5ce6612](https://github.com/ricardogr07/market-lab/commit/5ce661271d56099f534d7aeeee0d22b3ad59d327))
* expand phase 2 e2e validation ([6905619](https://github.com/ricardogr07/market-lab/commit/69056192f52494264f1d877d1c0df3617712cdfb))
* extend experiment report for baseline and ml comparison ([774b553](https://github.com/ricardogr07/market-lab/commit/774b553c4dd084de31c927ec8b64d90b3a246218))
* extend markdown report with analytics sections ([38ec7ee](https://github.com/ricardogr07/market-lab/commit/38ec7ee7e566b17b9bc9e793a0fb45c450fcb730))
* implement walk-forward training pipeline ([5db34bf](https://github.com/ricardogr07/market-lab/commit/5db34bf4f744a06649f5ec967a0790cde3270009))
* integrate ml strategies into run-experiment ([3f92321](https://github.com/ricardogr07/market-lab/commit/3f9232199b916a90bd2d680dc88831e3c708f01b))
* persist analytics artifacts and turnover plot ([5afba11](https://github.com/ricardogr07/market-lab/commit/5afba11be39432b6841244ab81c399debef620ef))
* source runtime version from installed metadata ([b69bff5](https://github.com/ricardogr07/market-lab/commit/b69bff5b8481100e7b56e05ae2df509a0698a137))
* upload Docker runner artifacts ([7e4d109](https://github.com/ricardogr07/market-lab/commit/7e4d109ee4f22eb85376c16452430d42c2cb4b9c))


### Bug Fixes

* derive turnover report from turnover costs ([a811bc8](https://github.com/ricardogr07/market-lab/commit/a811bc8435b03e536ff47b69deb771fc441a3293))
* prefer repo source in local launcher ([56a42a5](https://github.com/ricardogr07/market-lab/commit/56a42a58cd8eb9da3e5035c08cc672e65448b2de))
* restore tox ci environment setup ([7c06ac9](https://github.com/ricardogr07/market-lab/commit/7c06ac9217aba2a8dd8a8b8d5de0e97afaf5cef4))


### Documentation

* add architecture guide sprint review and codex workflow ([bd9d1d6](https://github.com/ricardogr07/market-lab/commit/bd9d1d67fe5438bb3bff5efb3864f6116898a756))
* add changelog scaffold for release automation ([65b710a](https://github.com/ricardogr07/market-lab/commit/65b710a1a1e91da0f5514d35865f0c3f087ef502))
* add phase 2 results review ([227cc85](https://github.com/ricardogr07/market-lab/commit/227cc85adf645abf1821ad7c155dcfe6c577126f))
* add repository community files ([ab8a7d5](https://github.com/ricardogr07/market-lab/commit/ab8a7d52017d502faa57755cffb5b52feec4eb37))
* clean public-facing project wording ([1568493](https://github.com/ricardogr07/market-lab/commit/15684932fe8b0cc2b7465efdbe7381abffac9c86))
* document Docker runner workflow ([2ca7a82](https://github.com/ricardogr07/market-lab/commit/2ca7a827b91b33f8765028fc11e29745b4450cc4))
* document installed package bootstrap flow ([1ad0ec7](https://github.com/ricardogr07/market-lab/commit/1ad0ec7eab1b5867c72abbd2a1ed44035d3084d8))
* document integration gate and preflight ([7785132](https://github.com/ricardogr07/market-lab/commit/7785132199c559cf27adebe03ec0154b18133eda))
* document local ci entrypoints ([2993742](https://github.com/ricardogr07/market-lab/commit/2993742639f45589041713eacd1d7d3e8643ae68))
* document monthly release batching and publish operations ([9699521](https://github.com/ricardogr07/market-lab/commit/969952191b043bcdc42c9646059dabc22f41bdb6))
* document public development workflow ([87b5d4d](https://github.com/ricardogr07/market-lab/commit/87b5d4d7e43d522de402e4fd81a24ab49e2ba5d9))
* narrow public docs site surface ([39fe1e6](https://github.com/ricardogr07/market-lab/commit/39fe1e6d338a887d8faef123da97362dad5bc6e3))
* sync architecture with phase 2 validation flow ([4b40fe3](https://github.com/ricardogr07/market-lab/commit/4b40fe354337626fb383d47a6c0dd34f8e57a135))
* sync phase 2 execution plan ([96b64ad](https://github.com/ricardogr07/market-lab/commit/96b64adb3338414a006dfbde549b28d090d5c12f))
* sync phase 2 execution plan ([45c738c](https://github.com/ricardogr07/market-lab/commit/45c738c7db984fd3c489dbbbe5337ea460c388ae))
* update analytics docs and smoke validation ([b399bb0](https://github.com/ricardogr07/market-lab/commit/b399bb0aa95faa61be36738886e73d84a92403ff))
* update architecture for ml experiment integration ([c173052](https://github.com/ricardogr07/market-lab/commit/c173052d42d7e17f7a7542a63ebcb9315792f935))
* update architecture for train-models flow ([57b680a](https://github.com/ricardogr07/market-lab/commit/57b680ad0f6e02100c70525a642180984ea25d2c))
* update architecture for walk-forward evaluation ([2d3b860](https://github.com/ricardogr07/market-lab/commit/2d3b860fc76d918d9119e86a992189658bdf8d0d))
* wrap up phase 2 documentation ([ede6532](https://github.com/ricardogr07/market-lab/commit/ede65321b40c9d236278c238e728065284f9b321))

## Changelog

All notable changes to this project will be documented in this file.

This file is managed by release automation. Normal feature PRs merge to `master`,
and release-please keeps a single Release PR updated with the unreleased batch
until maintainers merge that Release PR to cut a public release.
