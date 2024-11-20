# Changelog
Automatically updated by
[python-semantic-release](https://python-semantic-release.readthedocs.io/en/latest/)
with commit parsing of [angular commits](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#commits).

## Unreleased
### 🧰 Chores / Maintenance
### 🐛 Bug Fixes
* make xesmf optional dependency ([`664072a`](https://github.com/mdtanker/invert4geom/commit/664072aa32f18e4938148d0a04ba04402cee8018))


## v0.11.0 (2024-11-19)
### 🧰 Chores / Maintenance
* remove ruff commands and use pre-commit ([`7495204`](https://github.com/mdtanker/invert4geom/commit/74952047270f2f86ceb56fc907a4792623ee4ca3))
* bump binder env to v.0.10 ([`0604561`](https://github.com/mdtanker/invert4geom/commit/060456170989616a09aa5824815194ee60061804))
* update release checklist and make commands ([`8943bd0`](https://github.com/mdtanker/invert4geom/commit/8943bd04fbd076b48d379ab325e1e7969c740bf5))
### 📖 Documentation
* minor updates ([`82210b2`](https://github.com/mdtanker/invert4geom/commit/82210b27fe486ca7f0c205a567e9e4036cbb7dc8))
* rerun docs ([`9f7348f`](https://github.com/mdtanker/invert4geom/commit/9f7348f1aa71a49240b095896d7abd5731a20a2f))
* fix docstring ([`b0ec481`](https://github.com/mdtanker/invert4geom/commit/b0ec4815a07ab4595e81ca2b42c21f554a6bf135))
* add TLDR and conventional comments to contribute guide ([`1371096`](https://github.com/mdtanker/invert4geom/commit/13710969abe34b811303044ffe00a4fab0a63fff))
* saving progress ([`bdd6907`](https://github.com/mdtanker/invert4geom/commit/bdd69078d6784c40c4a97867f4127858da870c3d))
* restructure to Divio system ([`865983a`](https://github.com/mdtanker/invert4geom/commit/865983a8c380bfcbc799d1ae7fa90b6cac7fdedf))
* update some docs notebooks ([`6fe1488`](https://github.com/mdtanker/invert4geom/commit/6fe1488d7cda52615d7890cf11c4811b637d17fd))
### 🚀 Features
* return fig for `plot_cv_scores` ([`70e4a0d`](https://github.com/mdtanker/invert4geom/commit/70e4a0d29e22021481ee2ebf14549eccdbd2844f))
* add `fname` kwarg to some plotting functions ([`155f645`](https://github.com/mdtanker/invert4geom/commit/155f6450ae9ea58b56e7a9c3f32f299656c5b217))
* add plotting functions for latin hypercube sampling ([`2607284`](https://github.com/mdtanker/invert4geom/commit/26072840fb55c7fbad95ff77ab739add00928f3f))
* add `region` parameter for `regional_misfit_uncertainty` ([`ef5239e`](https://github.com/mdtanker/invert4geom/commit/ef5239e2c15d4f826bd4f851ed058fb3f3d05705))
* add uncertainty estimate for equivalent source interpolation ([`0cb4a21`](https://github.com/mdtanker/invert4geom/commit/0cb4a21973cc1457fc5551674f08917b0816d360))
* allow normalizing of sampled values in latin hypercube ([`b01c210`](https://github.com/mdtanker/invert4geom/commit/b01c210e4c05996f17a4a8171942f75ca84b9c74))
* include discrete uniform distribution for LHS sampling ([`7bd98e9`](https://github.com/mdtanker/invert4geom/commit/7bd98e918c086b0b0e58eb480b0db413f6112e21))
* allow choice of `criterion` in sampling of latin hyper cube ([`f7e94a1`](https://github.com/mdtanker/invert4geom/commit/f7e94a1137f89b25b81d0ed6668cb84cdd4ed0d2))
* allow `weight_by=None` in `merged_stats` ([`d96e5c2`](https://github.com/mdtanker/invert4geom/commit/d96e5c20b866c25da5343d1cec76ab4a5299ac44))
* add normalizing functions ([`bebaa67`](https://github.com/mdtanker/invert4geom/commit/bebaa67b74520a21d9d0ce69e392a9da272ec518))
### 🐛 Bug Fixes
* use duplicate filter on optimize for zref/density ([`cc825e0`](https://github.com/mdtanker/invert4geom/commit/cc825e0a4701ae1e0102e5b41b98c643b4b7f13e))
* exception for trail worsening warning ([`313f847`](https://github.com/mdtanker/invert4geom/commit/313f8474b22b557bd7706e72e0c59ff27673480e))
* raise warnings for invalid number of splits in test train data splitting ([`bcef273`](https://github.com/mdtanker/invert4geom/commit/bcef273b38f33cf564300b716559d3f8699b1932))
* raise warnings and use fallback for nan scores in eq source fitting ([`ad14dc0`](https://github.com/mdtanker/invert4geom/commit/ad14dc09d35c318d473aaed1981e976ab68b1015))
* re-create starting topography for each fold of training constraint points in  density/zref optimization. ([`91ffb71`](https://github.com/mdtanker/invert4geom/commit/91ffb71979dc9d87d8c0287edf426400098c9377))
* re-write long-wavelength contamination function, adding dependency `xesmf`. ([`335e11e`](https://github.com/mdtanker/invert4geom/commit/335e11e7a770e03a685abb54cc7fb20a6692c885))
###  🎨 Refactor
* make &#34;default&#34; the default depth for eq sources ([`6b68c2b`](https://github.com/mdtanker/invert4geom/commit/6b68c2bbcb5a3759aeb98722c4bc0cec52a3c36e))
* print more decimals for inversion info ([`05cbda9`](https://github.com/mdtanker/invert4geom/commit/05cbda902768c527c82550bdc62432f0f7781981))

## v0.10.0 (2024-10-09)
### 📦️ Build
* make xesmf optional dependency ([`8f3494c`](https://github.com/mdtanker/invert4geom/commit/8f3494c66d3cb3d0999c55b56fe148fcfcc81c62))
### 🧰 Chores / Maintenance
* set ruff rule ([`7cdb2b0`](https://github.com/mdtanker/invert4geom/commit/7cdb2b0b6dffd3567f1277e94f1a88fd0b24562f))
### 📖 Documentation
* update and rerun doc notebooks ([`7285281`](https://github.com/mdtanker/invert4geom/commit/728528117ce62b3536fc20cbcaf76c23b5080406))
### 🚀 Features
* allow passing column name for weights in create_topography ([`1815139`](https://github.com/mdtanker/invert4geom/commit/18151399629e2b0fda306171577510250d38fee1))
* add fname option to optimize functions ([`1bb69b6`](https://github.com/mdtanker/invert4geom/commit/1bb69b6044ed1c6149d7f3b083debf19d97d7040))
### 🐛 Bug Fixes
* minor changes ([`338306d`](https://github.com/mdtanker/invert4geom/commit/338306dd5d0f9b8349f5a93cc4585a0c65097eaa))
* re-write long-wavelength contamination function, adding dependency xesmf ([`6e96870`](https://github.com/mdtanker/invert4geom/commit/6e968709a11cc28039d08695a83206f64a7c51fb))
* add debug for eq source fitting ([`43e2855`](https://github.com/mdtanker/invert4geom/commit/43e28559184a38da218acf849bcbe02dae83a622))
* if too few points, reduce K folds until it works for fitting Splines ([`5d40917`](https://github.com/mdtanker/invert4geom/commit/5d40917f3f0bc3e470f4c13a7ebdc1994275a490))
* re-create starting topography for each fold of training constraint points in density/zref optimization ([`5da58fb`](https://github.com/mdtanker/invert4geom/commit/5da58fbcf8a091506636f19f7755853e506bfbf4))
* bug for checking gravity inside topo region with density/zref cv ([`e96099a`](https://github.com/mdtanker/invert4geom/commit/e96099a3b18e44a34885e852cabef9054209ee47))
### ✏️ Formatting
* pre-commit fixes ([`7802143`](https://github.com/mdtanker/invert4geom/commit/7802143f8bea1ee0101540b5ee88f21011b937ff))
* pre-commit fixes ([`28687c5`](https://github.com/mdtanker/invert4geom/commit/28687c597d4c956ac4adabf691084061c13313e8))

## v0.9.1 (2024-10-03)
### 📦️ Build
* add pooch and scikit-learn as dependencies ([`8bebf45`](https://github.com/mdtanker/invert4geom/commit/8bebf4545ad816fddc25d59d91dcc578019c86ca))
### 🧰 Chores / Maintenance
* update binder env to `v0.9` ([`0307bd9`](https://github.com/mdtanker/invert4geom/commit/0307bd9f38c5f20d83eeab8e73c9e5dd0b1f5f4e))
### 📖 Documentation
* update and rerun docs ([`1d39529`](https://github.com/mdtanker/invert4geom/commit/1d39529ee0b9bbf11eea7e32a87e2a710b1fa887))
### 🐛 Bug Fixes
* fixes error of using 1x not 4.5x default depth for eq sources ([`5746851`](https://github.com/mdtanker/invert4geom/commit/57468519286d34691f06855e9d278f8ee2a598d0))
* enable `default` eq source depth for regional constraints ([`d9301a3`](https://github.com/mdtanker/invert4geom/commit/d9301a3394e4a54b171a245d8bd0af7821c11c98))

## v0.9.0 (2024-08-10)
### 📦️ Build
* increase min supported botorch version to include logie_candidates_func ([`1491b24`](https://github.com/mdtanker/invert4geom/commit/1491b2438f4cc63a81377cf7d29799582e9f9530))
* restrict scipy to below `v1.14` until UQpy import issue is fixed ([`474e5b5`](https://github.com/mdtanker/invert4geom/commit/474e5b58c38c4f7c564aa021fb057c111e67da16))
### 🧰 Chores / Maintenance
* fix binder env ([`1066cc6`](https://github.com/mdtanker/invert4geom/commit/1066cc6fc56bd01f21cd5bb6e2d9173894092225))
* update binder env ([`bf45857`](https://github.com/mdtanker/invert4geom/commit/bf45857ed75b03c153f9ca04d2485d5b43490330))
### 📖 Documentation
* fix notebook error ([`f7783e2`](https://github.com/mdtanker/invert4geom/commit/f7783e22614675839286159e2bc42572d4603ef7))
* unindent docstring ([`e0715d6`](https://github.com/mdtanker/invert4geom/commit/e0715d63071ee543f8c53170708162d1da207d0d))
### 🚀 Features
* add `constraint_style` to plotting inversion grav resutls ([`c062b43`](https://github.com/mdtanker/invert4geom/commit/c062b4365b529d3fd76c280f2a968eb0bc5eb3af))
### 🐛 Bug Fixes
* raise error for NaNs in gravity dataframe ([`3ca5cca`](https://github.com/mdtanker/invert4geom/commit/3ca5cca879071af6701072772e742f2a02ac9e44))
* drop nans in constraints_df in `regional_constraints` ([`6ee2b6b`](https://github.com/mdtanker/invert4geom/commit/6ee2b6b29feebbea0931509f455e9cf28f38cb1a))
* add warning for missing `split_kwargs` argument ([`ef96fd9`](https://github.com/mdtanker/invert4geom/commit/ef96fd9dcfc00572076f04e4e3e546025621d579))
* use `copy.deepcopy()` for all dictionaries inside functions ([`10df82f`](https://github.com/mdtanker/invert4geom/commit/10df82f7e6db1b7748b49b1e678765e2b905f2d1))
* `warn` instead of `raise error` for constraints outside of gravity data ([`08a60b4`](https://github.com/mdtanker/invert4geom/commit/08a60b45662823668954ce98d3ca247395caf356))
###  🎨 Refactor
* manually calculate `default` eq source depth until harmonica `v0.7` is released. ([`aeb675a`](https://github.com/mdtanker/invert4geom/commit/aeb675acafc185f1e5a7bcd17fb86e0c518005a3))
* simplify UQpy import ([`783d3dc`](https://github.com/mdtanker/invert4geom/commit/783d3dcd08770b8eb95a134e7742d5382d14b1b7))

## v0.8.1 (2024-08-05)
### 🧰 Chores / Maintenance
* add test deps to conda_install for testing conda releases ([`c44d59b`](https://github.com/mdtanker/invert4geom/commit/c44d59ba2dceddfa15d48b2f9dd2315a20e2b304))
* update binder env to `v0.8` ([`8d9d294`](https://github.com/mdtanker/invert4geom/commit/8d9d29410d4b406070fcb2316673530b9cff7d61))
### 📖 Documentation
* add missing docstring ([`50591e3`](https://github.com/mdtanker/invert4geom/commit/50591e3a809c94c4d78851417515a79d51c6fa89))
* fix issues with autoapi and typehints. ([`af93c02`](https://github.com/mdtanker/invert4geom/commit/af93c024f023c5967ab70cd2909ee67cf8f24194))
### 🐛 Bug Fixes
* remove nptyping as dependency ([`d2534d0`](https://github.com/mdtanker/invert4geom/commit/d2534d083759f8769c42c1979054aa3b6a94aa5f))

## v0.8.0 (2024-08-02)
### 💥 Breaking Changes
* change equivalent source param names

BREAKING CHANGE: update parameter `eq_cv`-&gt;`cv` in `regional_eq_sources` and `regional_constraints`, and use parameter `cv_kwargs` to pass args to `optimize_eq_source_params` ([`10e47ef`](https://github.com/mdtanker/invert4geom/commit/10e47ef7b6bc19f0b303755d37bdc4ffde6d35cc))
* update `utils.best_spline_cv` to pass all kwargs directly to verde.SplineCV.

BREAKING CHANGE: update parameter `spline_damping` to `spline_dampings` in function `regional.regional_constraints` and all functions which feed into it (i.e. `regional.regional_separation`) ([`1546d06`](https://github.com/mdtanker/invert4geom/commit/1546d06437bfcb63f1c3abd6dfec283a5fea73dd))
* change all equivalent source parameter variable names.

BREAKING CHANGE: please update all variables names as follows: `eq_damping`-&gt; `damping`, `source_depth`-&gt;`depth`, `eq_damping_limits`-&gt;`damping_limits, `source_depth_limits`-&gt;`depth_limits` ([`1476e2a`](https://github.com/mdtanker/invert4geom/commit/1476e2a5233f38c4a36d3f3c7f9ba4889bf8db2e))
* overhaul `run_inversion_workflow` function to emit warnings at beginning, and enable kfolds CV for using constraints in regional separation with density/zref CV, and use better file names.

BREAKING CHANGE: removed `starting_prisms_kwargs`, put `density_contrast` and `zef` into standard kwargs now. All `regional_grav_kwargs` are now directly passed to the relavent functions, same for `starting_topography_kwargs`. ([`a72c8a4`](https://github.com/mdtanker/invert4geom/commit/a72c8a4738432587a9c7f29bc7dc39c408a4badc))
* internally calculate misfit and residual in regional estimation functions and use standardized names.

BREAKING CHANGE: The function `regional_dc_shift` has been removed, please use function `regional_constant` instead. Inputs to all regional functions have changed use to standardized column names. All functions will now automatically calculate the misfit and and residual, as well as the regional. ([`a501f54`](https://github.com/mdtanker/invert4geom/commit/a501f54afe814c6382d1d0f91a4f7a8ca423fa20))
* use standardized column names instead of setting with variables.

BREAKING CHANGE: To simplify the code, all instances of passing the column name for the various data type have been replace with preset names. Please update your code to use column names: `gravity_anomaly` instead of parameter grav_data_column`, `reg` instead and parameter `regional_column`. ([`a4f24ec`](https://github.com/mdtanker/invert4geom/commit/a4f24ecd3c84a9c545d13e54ad4cebf0e70794d1))
* deprecate `cross_validation.grav_optimal_parameter` in favor of new  Optuna-based function `optimization.optimize_inversion_damping`.

BREAKING CHANGE: please switch to the new function ([`3062ce3`](https://github.com/mdtanker/invert4geom/commit/3062ce3dcd239450bb9f6c1c1f9ee754dee19bf3))
* deprecate `cross_validation.zref_density_optimal_parameter` in favor of new  Optuna-based function `optimization.optimize_inversion_zref_density_contrast`.

BREAKING CHANGE: please switch to the new function ([`5059cd0`](https://github.com/mdtanker/invert4geom/commit/5059cd0aaaa943ddc7594a0700fda50b768fb00c))
* add delta L2 norms to convergence plots.

BREAKING CHANGE: `plot_convergence` now takes parameter `params` (output dict from run_inversion) and doesn&#39;t plot iteration times. `plot_dynamic_convergence` now takes parameters `l2_norms` and `delta_l2_norms` instead of `results`. ([`7cbdbb3`](https://github.com/mdtanker/invert4geom/commit/7cbdbb3c944aee4b96ca0c42569a7f6f25aebed4))
* update `regional_constraints` func.

BREAKING CHANGE: parameter `constraint_block_size` changed to `constraints_block_size`. Instead of supplying a lists of damping values to `dampings` for the `verde` method, now provide a single value with parameter `spline_damping`. For `eq_sources` method, instead of providing limits and trial numbers, just provide single parameter values with parameters `source_depth`, and `eq_damping`, `block_size`. ([`3496d5f`](https://github.com/mdtanker/invert4geom/commit/3496d5f0097da570cf79920f59d5f0bb5bec98eb))
### 📦️ Build
* remove unused optuna parallel functions and dependencies: joblib, psutil, and tqdm_joblib ([`56c3887`](https://github.com/mdtanker/invert4geom/commit/56c388776584291192c64d8556a14058bf1b0f12))
* remove `viz` optional deps and add to core deps ([`b152f1b`](https://github.com/mdtanker/invert4geom/commit/b152f1b057ebace6ab6d055444ba736fe3741e76))
* remove `opti` deps and include as part of core deps ([`a396318`](https://github.com/mdtanker/invert4geom/commit/a396318827001c1a7b45aad882ece92eb999ecee))
* add deprecation to deps ([`dec0ec8`](https://github.com/mdtanker/invert4geom/commit/dec0ec809ada9781e09ca8f2f0738dc61cd1262c))
### 🧰 Chores / Maintenance
* remove broken test and decrease timeout ([`3639b9c`](https://github.com/mdtanker/invert4geom/commit/3639b9c6778473e328a07b0809523727194a1d6e))
* remove pre-commit GHA in favor for pre-commit.ci ([`865c514`](https://github.com/mdtanker/invert4geom/commit/865c514551c2adf62b61c4d8955e48551fc6d216))
* delete unused release file ([`8118b97`](https://github.com/mdtanker/invert4geom/commit/8118b971f093d209bb3390525bcbea87a65783a9))
* remove check-manifest from pre-commit ([`71accc4`](https://github.com/mdtanker/invert4geom/commit/71accc43c119abafbe64847ad9a144caa1d24939))
* updates from learn-scientific-python ([`65b1815`](https://github.com/mdtanker/invert4geom/commit/65b1815ec13bebc1f1cf72f62e4e727ce8fcc40d))
* gitignore tmp folders ([`c228cb2`](https://github.com/mdtanker/invert4geom/commit/c228cb2460c526e02c5899ad1a29326cc259877e))
* misc small changes ([`b4d247d`](https://github.com/mdtanker/invert4geom/commit/b4d247dcc9581ab7bf80914c65a490ea73f602d0))
* make some functions private to declutter API documentation ([`e19c6f2`](https://github.com/mdtanker/invert4geom/commit/e19c6f241f3a85136a40215b78570c8ebe5ddefb))
* remove test optuna parallel ([`a1b2477`](https://github.com/mdtanker/invert4geom/commit/a1b247742ca30865da7a7230e79c43a85ac4087c))
* ignore optuna experimental warnings ([`e1fcb78`](https://github.com/mdtanker/invert4geom/commit/e1fcb7859f321401f85ca0fdb38d6938f1225840))
* remove check import for psutil, joblib and tqdm_joblib ([`0a3a925`](https://github.com/mdtanker/invert4geom/commit/0a3a9255cfbf20476c7b1a1b4b6f7ad27b188394))
* add debug logging throughout ([`70da273`](https://github.com/mdtanker/invert4geom/commit/70da27338edccc4104e72ee457fc574327d82eb3))
* ignore some ruff issues in notebooks ([`d18e866`](https://github.com/mdtanker/invert4geom/commit/d18e86620b8b1a535109fc9529b835418d823665))
* update make commands ([`50f1881`](https://github.com/mdtanker/invert4geom/commit/50f1881014c956fc6ddbbe245c437d22565b05a4))
* fix failing tests ([`7edfed2`](https://github.com/mdtanker/invert4geom/commit/7edfed2a3bd28bcba1b9605b6814ac5ab0e38033))
* add pickle and sqlite files to make clean ([`71e897f`](https://github.com/mdtanker/invert4geom/commit/71e897f641cb12fa801c9f391f6f7b952cd4dccb))
* add nodefaults to testing env ([`9a06f62`](https://github.com/mdtanker/invert4geom/commit/9a06f62142b8262a37117b33fd4ef8d070727d68))
* update invert4geom version in environment.yml ([`7762e06`](https://github.com/mdtanker/invert4geom/commit/7762e060577e91ac2397a931336e4d1e8d2eeacc))
* updates to ci.yml ([`b3bb4d4`](https://github.com/mdtanker/invert4geom/commit/b3bb4d4baf0cce2679c6e54ac6f8e53b8847afad))
* use micromamba for GHA test env and cache it ([`481a296`](https://github.com/mdtanker/invert4geom/commit/481a296594d8be25aa96fdb39a3a0e634723ae69))
* dont run test on docs or style commits ([`8b1065f`](https://github.com/mdtanker/invert4geom/commit/8b1065fc60d4bb085604bf65198a9f56b1b158cb))
* set min pylint version ([`f36ef0f`](https://github.com/mdtanker/invert4geom/commit/f36ef0f82636cc4b25e8a02d0afd9d307a807296))
* update make commands ([`e5bc907`](https://github.com/mdtanker/invert4geom/commit/e5bc907946f8c775ec7bbd447016ba3b4f40561a))
* precommit ignore doc md files ([`78461c3`](https://github.com/mdtanker/invert4geom/commit/78461c36dfe36789d3c43837d2f3e228f5574b08))
### 📖 Documentation
* update all doc notebooks ([`4cc0b69`](https://github.com/mdtanker/invert4geom/commit/4cc0b699e5c80e04c0986e1eb6ac856599a8268c))
* update some docstrings ([`ff932dd`](https://github.com/mdtanker/invert4geom/commit/ff932dd14c863f92252dba8398d57fc796f165b3))
* update overview and conventions ([`a225f45`](https://github.com/mdtanker/invert4geom/commit/a225f45f85015637e42cafe0b48db37765ecaca6))
* typo in contrib guide ([`32a8bdf`](https://github.com/mdtanker/invert4geom/commit/32a8bdf96e38ee4dfebaa3cdb5244272caf6cec5))
* add accidentally missing api files ([`b0b73d9`](https://github.com/mdtanker/invert4geom/commit/b0b73d96157a8bc10c9e434eaf1bca2d60f84355))
* update the docs ([`afab24d`](https://github.com/mdtanker/invert4geom/commit/afab24d99e9f88cee55883c78e3d19f694ef8ad1))
* move contrib guide into docs folder ([`3009e90`](https://github.com/mdtanker/invert4geom/commit/3009e9027ca21b5593cd55ed56f004f61f15f6c9))
### 🚀 Features
* add `uncertainty` module ([`3d624c7`](https://github.com/mdtanker/invert4geom/commit/3d624c75323c596601862bf8f1e04a8648cd8d55))
* add upward continuation option to eq_sources regional method ([`10e4c3e`](https://github.com/mdtanker/invert4geom/commit/10e4c3e7c4744235afa09926840e67bb2f8198f8))
* add back optuna parallel functions ([`4d53fde`](https://github.com/mdtanker/invert4geom/commit/4d53fde6627221a263f03ba495f2a76286f73168))
* add non-optuna function for find best eq source damping ([`3f9c6d1`](https://github.com/mdtanker/invert4geom/commit/3f9c6d15d7fe3754d4e76220b1f1e4a9232bc7a0))
* add class for filtering duplicate log messages ([`b8df17d`](https://github.com/mdtanker/invert4geom/commit/b8df17db811f86ece3e2fc9eb38a0e940414c89c))
* add damping value as dataarray attribute in `create_topography` ([`6e96f72`](https://github.com/mdtanker/invert4geom/commit/6e96f720248562d0097671ef1c045acf3cbbd497))
* add contextmanager function to temporarily set an env variable ([`b66bb59`](https://github.com/mdtanker/invert4geom/commit/b66bb596da4261ec20c3b7df5107f576b349a955))
* update fname for inversions and CV pickle files to be simpler ([`a8c3a1a`](https://github.com/mdtanker/invert4geom/commit/a8c3a1a057dcee860abba5d01f55b001eb5e7177))
* add buffer arg to `load_synthetic_model` ([`8ac6fb1`](https://github.com/mdtanker/invert4geom/commit/8ac6fb1e6777877df84dcb6b5710ad28e3a5707e))
* add checks for regional extent of gravity and constraints ([`997b416`](https://github.com/mdtanker/invert4geom/commit/997b416a2dff2218638ea5316b803b8dd7641356))
* add fig_height kwarg to inversion plots ([`40cb035`](https://github.com/mdtanker/invert4geom/commit/40cb035aac6c702f7682dca4da5ab35ec1bc0889))
* allow separate or combined metrics for regional hyperparameter optimizations ([`973d93d`](https://github.com/mdtanker/invert4geom/commit/973d93de0f33724a406111907e5a0358ee5dfa95))
* add function `regional_constraints_cv` to automatically perform a K-Folds cv for finding optimal parameter values for constraint point minimization ([`06ebe79`](https://github.com/mdtanker/invert4geom/commit/06ebe79006b5cdac5ad15b29a59c3a4be8e669fc))
* allow internal CVs for eq sources and splines for `regional_constraints` ([`4982cc3`](https://github.com/mdtanker/invert4geom/commit/4982cc3fe4d4903d5ecb5905402a172a6ed538a2))
* allow internal CV in `regional_eq_sources` ([`24fed1a`](https://github.com/mdtanker/invert4geom/commit/24fed1a216d9f8625c83eca9f61732d8cfd79271))
* add data weights to `regional_eq_sources` ([`e63e49d`](https://github.com/mdtanker/invert4geom/commit/e63e49d6f07efb9af3c345fe9c3dc31bc03ea604))
* add `grav_obs_height` parameter to regional_eq_sources ([`ed344ae`](https://github.com/mdtanker/invert4geom/commit/ed344ae550a803404a53c8846eb7188165431f49))
* add function for loading all synthetic topography and gravity ([`d4c8b07`](https://github.com/mdtanker/invert4geom/commit/d4c8b07e67cf37d5663040151456ccba99c29111))
* add function for contaminating data with long-wavelength noise ([`5d087a2`](https://github.com/mdtanker/invert4geom/commit/5d087a23a8030da4c00bc391c0387e9945fabcbc))
* new function `optimize_inversion_zref_density_contrast_kfolds` for use constraints within the CV via an inner kfolds-CV ([`f596f37`](https://github.com/mdtanker/invert4geom/commit/f596f37b63e9477b07e28e1ce1a6907027f48eca))
* remove default args for `optimize_eq_source_params` ([`df1fb14`](https://github.com/mdtanker/invert4geom/commit/df1fb14f16ff62ad8e1a0afcd59e8da1222bdaa9))
* add progressbar and callbacks to `optimize_eq_source_params` ([`7156c05`](https://github.com/mdtanker/invert4geom/commit/7156c05eedeb991ab99a3cb1248e3970f33899bc))
* all passing kfolds of constraints to `optimize_inversion_zref_density_constrast` ([`dc17ed3`](https://github.com/mdtanker/invert4geom/commit/dc17ed3b06d7b198ac93b7d0b5548b5266b5bed0))
* add plotting function for fixing hoverdata on plotly-optuna plots ([`da2af82`](https://github.com/mdtanker/invert4geom/commit/da2af8246e552f55c4f44614836c30752f853b63))
* add plotting function for stochastic uncertainty ([`88288a9`](https://github.com/mdtanker/invert4geom/commit/88288a90ed0cfb2d8239801c4ccb8841f6b5ff56))
* new function `random_split_test_train` to randomly split data for cross validations. ([`3b797ab`](https://github.com/mdtanker/invert4geom/commit/3b797ab212859ecb1116aa1f872a22b5b5aaf1ef))
* add optimization for eq source params ([`9a31851`](https://github.com/mdtanker/invert4geom/commit/9a318510d802a4ae45e8927cc8449a02dde463b2))
* add function to give score for regional estimations ([`276da98`](https://github.com/mdtanker/invert4geom/commit/276da98b44dd70c733472f7b7fd46100ee392bef))
* add function to convert format of test train folds dataframe ([`eec2a93`](https://github.com/mdtanker/invert4geom/commit/eec2a933bf5ccb60b23eaf1a8af35e3286124520))
* add function for splitting data into test and train sets with several methods ([`5e252f1`](https://github.com/mdtanker/invert4geom/commit/5e252f1deac287bd2fa9b4a08cfe4695444b1cc5))
* add optimization functions for all regional estimation techniques. ([`7db27cc`](https://github.com/mdtanker/invert4geom/commit/7db27cc61a3b161afddd13d912586e11babc43e1))
* add functions for optuna logback to warn about best parameter values being at limits. ([`53fafc7`](https://github.com/mdtanker/invert4geom/commit/53fafc75fc776c9e4cf16a8802bebc4efacf690d))
* include multi-objective studies in custom optuna logback func ([`89d4403`](https://github.com/mdtanker/invert4geom/commit/89d440354861882363170cd867921dd378f54d98))
* add `constraints_df`to plotting functions ([`64147bf`](https://github.com/mdtanker/invert4geom/commit/64147bf30e96c62f2bd52979db5d4b86afe806fd))
* add plotting func for non-grid search 2 parameter CV ([`b86ebf9`](https://github.com/mdtanker/invert4geom/commit/b86ebf96cad8627f6b07493066361a2409fe76be))
* add `constraints_df` arg to `plot_inversion_results` ([`57aefcc`](https://github.com/mdtanker/invert4geom/commit/57aefcc6a9326866ed1119493e7e4b6ba3342f75))
* allow already separated train/test sets for gravity data in run_inversion_workflow ([`9fa7f27`](https://github.com/mdtanker/invert4geom/commit/9fa7f271590652bd6a325ff329f0aca8116446dd))
* add fname option to run_inversion_workflow to save results ([`d393d7d`](https://github.com/mdtanker/invert4geom/commit/d393d7d12d6f983729819592629823c87fa3618e))
* add option to turn off inversion progressbar ([`17c6b32`](https://github.com/mdtanker/invert4geom/commit/17c6b32b7076e13b1d28e8a4fc1da49880564cac))
* add load_bishop_model function ([`8006cf5`](https://github.com/mdtanker/invert4geom/commit/8006cf5dced4fbe00435db1fdcd4aadda2f6d74c))
* add weights option to create_topography ([`17827b4`](https://github.com/mdtanker/invert4geom/commit/17827b4140d6a0660b21e48353f7b851eaf6ddec))
### 🐛 Bug Fixes
* ensure all dictionary `pop` calls are made on copies ([`1c80404`](https://github.com/mdtanker/invert4geom/commit/1c80404bc4b37948cd0471b3b25b4f97c84c2828))
* enable upward continuation option for CPM with equivalent source gridding ([`0c23eca`](https://github.com/mdtanker/invert4geom/commit/0c23ecafaeeea0716be0bce90599f061645db107))
* separate cmaps for starting and ending gravity residual plots ([`d6feb0e`](https://github.com/mdtanker/invert4geom/commit/d6feb0ecdc8d46c16a5b29171ac74e444893e2ed))
* explicitly save inversion results in `run_inversion_workflow` ([`5db892e`](https://github.com/mdtanker/invert4geom/commit/5db892e1b1fe9359123c0190d9f141562da29a93))
* bug in eq_sources_score with too many folds ([`c78354e`](https://github.com/mdtanker/invert4geom/commit/c78354e7669a8ddeb69e90413fbdc9d214b024f4))
* raise error if trying to get regional score from method `constraints_cv` ([`d35b703`](https://github.com/mdtanker/invert4geom/commit/d35b70304f45d5c8b17028794044ceb697814ed0))
* make `dampings` required for `create_topography` and `starting_topography_kwargs` ([`c3d407b`](https://github.com/mdtanker/invert4geom/commit/c3d407b3033ea7cdb2f59782304e6ce506923483))
* only plot optuna importances if &gt;1 parameter ([`ba17ed1`](https://github.com/mdtanker/invert4geom/commit/ba17ed14769b3ea04d16f5af633eb4e406d74dc8))
* update `regional_eq_sources` to work with new param names ([`68f85f2`](https://github.com/mdtanker/invert4geom/commit/68f85f2727a63e9151b30b7be1f9f5d0c468ce74))
* bug in `optimize_eq_source_params` ([`eb0ebbe`](https://github.com/mdtanker/invert4geom/commit/eb0ebbe73c8e22b62fdeba2af66453692287dfa9))
* add warnings about constraints and regional separation ([`c6bc503`](https://github.com/mdtanker/invert4geom/commit/c6bc5033f4b82bfd71a278052ac2e84851ae0b6a))
* add warning in run_inversion_workflow for using constraints with zref/density CV ([`88e9a86`](https://github.com/mdtanker/invert4geom/commit/88e9a86d985c4665ba3f6d44cd4be80ecf9b34ab))
* add assert thay run_inversion_workflow uses correct zref and density values ([`5692da4`](https://github.com/mdtanker/invert4geom/commit/5692da412346e9f811a1c9978bfcca27cf13914a))
* minor changes ([`dbe8ca5`](https://github.com/mdtanker/invert4geom/commit/dbe8ca5710653c7de8ce1c62c93ee2352b35e273))
* bug in run_inversion_workflow ([`bf875f8`](https://github.com/mdtanker/invert4geom/commit/bf875f8cc4c9ba2141768d52a464a9aff9311ecd))
* bug in `optimize_inversion_zref_density_contrast_kfolds` ([`8886438`](https://github.com/mdtanker/invert4geom/commit/8886438bb904a414adb67ee468572f2ddd4fb58f))
* suppress info logs for run inversionwith kfolds CV ([`3bb59d2`](https://github.com/mdtanker/invert4geom/commit/3bb59d27a48a9136ebe3d0beee32869aeb9c813b))
* suppress info logs for regional separations ([`e15d3de`](https://github.com/mdtanker/invert4geom/commit/e15d3de62e700e4a13aec9246a33126ff10f3922))
* change `regional_method` to `method` in `regional_grav_kwargs` ([`d52f01d`](https://github.com/mdtanker/invert4geom/commit/d52f01d0cdc43ba2c4ca6e2c7b8a282e9ac2393c))
* raise error for bug in `plot_2_param_cv_scores_uneven` ([`7ec08e7`](https://github.com/mdtanker/invert4geom/commit/7ec08e785a30822891a7ef9111c1f1bd14a0a2ad))
* add check for regional separation scores not being nans ([`e98bbfd`](https://github.com/mdtanker/invert4geom/commit/e98bbfd87962d59ec47c3fe00631f0fb7b793858))
* bug in regional test ([`c618097`](https://github.com/mdtanker/invert4geom/commit/c618097a794666962199a7c1fa228d9bcca6e821))
* bug in CPM kfolds optimization ([`30e6763`](https://github.com/mdtanker/invert4geom/commit/30e6763f19b2f3e734d5ea1f5023993573b52d6d))
* updating plotting of optuna results in `optimization` ([`af78985`](https://github.com/mdtanker/invert4geom/commit/af78985c1476e85e32879353e25f0870d4302608))
* add regional scores to trial user attrs in regional hyperparameter optimizations ([`9ac1bdf`](https://github.com/mdtanker/invert4geom/commit/9ac1bdf974e73bec857eabdc835a2fe716e5fcda))
* update `best_spline_cv` and use within `create_topography` ([`5c3d6c5`](https://github.com/mdtanker/invert4geom/commit/5c3d6c56e4a0baeeabab9493ac42ba8e61b1750f))
* raise error if index column already exists in `sample_grids` ([`7ca6c83`](https://github.com/mdtanker/invert4geom/commit/7ca6c83b3f992ae156c94c054615b8ce7b77fc60))
* update optuna plotting funcs ([`d3b1552`](https://github.com/mdtanker/invert4geom/commit/d3b15528988ee0b0602b0e37b7293d40adea7271))
* update `plot_2_parameter_cv_scores_unven` function ([`8b73d43`](https://github.com/mdtanker/invert4geom/commit/8b73d433ee4e5a3735d7817360cc64db63687c8b))
* enqueue trials for value limits in `optimize_eq_source_params` ([`af519e0`](https://github.com/mdtanker/invert4geom/commit/af519e051ead9f9a8e86fcb8effc42798dea53b5))
* add default value to source_depth in `OptimalEqSourceParams` ([`03b828e`](https://github.com/mdtanker/invert4geom/commit/03b828e2a900e001458ce2c42dba5c6774adeb84))
* use more startup trials in optimizing on both zref and density ([`789167b`](https://github.com/mdtanker/invert4geom/commit/789167b2cc1a7be7f999390a12132c2db8280a6e))
* remove warning about using constraints for finding constant regional value in CV as it doesn&#39;t seem to affect it. ([`14efcfc`](https://github.com/mdtanker/invert4geom/commit/14efcfc531b9e4cde17fc1c1e3d407d42615c91a))
* add warning if no supplied constraints in run_inversion_workflow if doing density/zref CV ([`2783ab4`](https://github.com/mdtanker/invert4geom/commit/2783ab476a6974da56a758eab9e215a4839962a3))
* allow no starting_prisms in `run_inversion_workflow` if doing a density/zref CV ([`da1674c`](https://github.com/mdtanker/invert4geom/commit/da1674c00f47c13794f9c9c1d715d1e1523f16b4))
* remove `data_column` from `random_split_test_train` and keep all non-coord columns in dataframe. ([`3584286`](https://github.com/mdtanker/invert4geom/commit/35842860e961b080cf28f360b63aa9cca239dde4))
* bug in `resample_with_test_points`, drop nans ([`feda68f`](https://github.com/mdtanker/invert4geom/commit/feda68fe611636c16cf6bb90dbad491664c51e75))
* bug fixed to explicitly remove prism_layer in zref/density CV ([`b70a893`](https://github.com/mdtanker/invert4geom/commit/b70a89302944b2ed8a5b7ecc326f28aa638b19db))
* update plotting for optuna optimization results ([`f2c60c4`](https://github.com/mdtanker/invert4geom/commit/f2c60c4f2b00c1f96f5008fc8e8ba940a8a4ac6f))
* log warning if using constraints points for finding constant regional value within a zref/density CV. ([`f7a135d`](https://github.com/mdtanker/invert4geom/commit/f7a135d1e9120483f64a3f4a687bf61d94650e55))
* bug in `plot_2_parameter_cv_scores_uneven` ([`b8ca0f4`](https://github.com/mdtanker/invert4geom/commit/b8ca0f491e8c61895dffbf91e49bc85d67c6ef91))
* bug in run_inversion_workflow ([`604729b`](https://github.com/mdtanker/invert4geom/commit/604729b4d127804f2146acf6fff04f1d3e194cae))
* minor fixes to logging ([`4481f5d`](https://github.com/mdtanker/invert4geom/commit/4481f5d8575d673cfa300fe16dac22593427b5a6))
* add crs parameter for `utils.nearest_grid_fill` ([`b2eec98`](https://github.com/mdtanker/invert4geom/commit/b2eec980332bcf66478ee780aaf016a241760427))
* turn off dynamic convergence plotting during CV ([`a09ad32`](https://github.com/mdtanker/invert4geom/commit/a09ad324ce4b611e73c483933d6900b00d551de0))
* remove unnecessary `depth_type` arg for eq sources ([`4998705`](https://github.com/mdtanker/invert4geom/commit/4998705e73de9020634595b7202c4d3679294148))
* add warning to run_inversion about unused weighting grid ([`e2676fa`](https://github.com/mdtanker/invert4geom/commit/e2676fa28a8065f86e37559d9737b772073621de))
###  🎨 Refactor
* clean up CPM optimization and cv. ([`f873490`](https://github.com/mdtanker/invert4geom/commit/f873490d6c6b96f7825f458164b9f0ffcbf61b0b))
* clean up zref/density CV functions ([`4ecf9ec`](https://github.com/mdtanker/invert4geom/commit/4ecf9ec4e0e4e9c565a7568611c9a0c0b2a83617))
* log info instead of warning for params at limits ([`24819a5`](https://github.com/mdtanker/invert4geom/commit/24819a5cc2de343d635614f81dfe83d5709e3d5e))
* redo regional separation with optimal parameters instead of storing results as attributes ([`ff369d8`](https://github.com/mdtanker/invert4geom/commit/ff369d8ac4eca7f1f5fcc6d3121723ecd897bd5f))
* use `run_optuna` for all optimizations to all for running in parallel. ([`c59f2c7`](https://github.com/mdtanker/invert4geom/commit/c59f2c7866f5197ef106f644acdbb7043c68172f))
* use `split_kwargs` in run_inversion_workflow and minor fixes ([`849bb39`](https://github.com/mdtanker/invert4geom/commit/849bb392f52b9ff31b7af4dd4f8ba2546231a9fb))
* move creation of regional sep studies to new function ([`be30d53`](https://github.com/mdtanker/invert4geom/commit/be30d537752f45a8252a702bb94b3b401ad486ef))
* default to separate metrics (multi-objective) for regional separation ([`91b2e4b`](https://github.com/mdtanker/invert4geom/commit/91b2e4b676cde5850ec218db7e28281cbaee0c56))
* update `optimize_eq_source_params` to work with recent changes ([`0e3ce7b`](https://github.com/mdtanker/invert4geom/commit/0e3ce7ba0e10e44837cda6e765d2055d0ab3ef4f))
* clean up `regional_constraints` and change default grid_method to &#34;eq_sources&#34; ([`7170ebd`](https://github.com/mdtanker/invert4geom/commit/7170ebd82ac529bfe39c4545eaf6f336311986a8))
* `cross_validation.eq_sources_score` now passes all kwargs directly to hm.EquivalentSources class ([`5f9f7c6`](https://github.com/mdtanker/invert4geom/commit/5f9f7c67cb604c3c753df419d49342c25a991dbc))
* consolidate optuna logging of results ([`68b8dc8`](https://github.com/mdtanker/invert4geom/commit/68b8dc8df26ff476d6457d71c7d70bebedf0096e))
* consolidate optuna warning for best parameter values at their limits ([`5a3a7d1`](https://github.com/mdtanker/invert4geom/commit/5a3a7d1378b57fce36def211c56651611656f070))
* use context manager to temporarily change logging level ([`c14aaca`](https://github.com/mdtanker/invert4geom/commit/c14aaca1763fbc5681f5be389b1cd9d4cdbb3e12))
* minor changes to `optimization` ([`18f2058`](https://github.com/mdtanker/invert4geom/commit/18f20580c459277b0c31ca1f4b2d8337f908332c))
* set default of `source_depth` to &#39;default&#39; in `regional_eq_sources` ([`ee7575f`](https://github.com/mdtanker/invert4geom/commit/ee7575fe5c7989ce3586694e6ffd041102b5914a))
* use pygmt instead of xarray for plotting some inversion results ([`cf377b3`](https://github.com/mdtanker/invert4geom/commit/cf377b399d146d45b908bceb5ba8cd2f2ba51173))
* use a logger specifically for invert4geom and update all logging calls ([`fa111cc`](https://github.com/mdtanker/invert4geom/commit/fa111cc40f1d3b187b04ae3583eda96f981c85f8))
* remove default value for `damping_cv_trials` and `zref_density_cv_trials`  in `run_inversion_workflow` ([`7dce943`](https://github.com/mdtanker/invert4geom/commit/7dce9433ed05df0b900179a0199171b14136bab3))
* cv score functions to return inversion results as well as scores ([`15e4369`](https://github.com/mdtanker/invert4geom/commit/15e43694d5c367e89115ad0455cbc04f2c811b52))
* remove buffer from delta l2 norm line in convergence  plots ([`cd85659`](https://github.com/mdtanker/invert4geom/commit/cd85659b8699a1c126cb32381aafc4a61b4b2aae))
* rename function `plot_optuna_inversion_figures` to `plot_optuna_figures` ([`4823cab`](https://github.com/mdtanker/invert4geom/commit/4823cab01a471481d175ee6ae39904c1c8eb3c3a))
* temporarily disable info-level logging in CV score functions ([`f5c329d`](https://github.com/mdtanker/invert4geom/commit/f5c329d51ca29d85a0dd409f3d65d45e41e727cf))
* use easier to understand names in `update_l2_norms` ([`1516795`](https://github.com/mdtanker/invert4geom/commit/15167954d111f2113f932b15c4946a1739e1b1ab))
* change point color to gray for `plot_2_parameter_cv_scores` ([`ccbd4ba`](https://github.com/mdtanker/invert4geom/commit/ccbd4baac736b8792501d3996f634e9b578a24e7))
* move eq_sources_score to cross_validation module ([`4def683`](https://github.com/mdtanker/invert4geom/commit/4def683b5c45957fd7fd20b6889e8f25f9e48498))
* explicitly set parameters instead of kwargs for `eq_sources_score` ([`2a8c133`](https://github.com/mdtanker/invert4geom/commit/2a8c1335e6fcf50fbeefb4bc99130afcbd1e99da))
### ✏️ Formatting
* auto style fixes ([`f49df6d`](https://github.com/mdtanker/invert4geom/commit/f49df6d6f519fdfeaf53ac1d1e9e1d5563f815b2))
* minor fixes ([`c8283be`](https://github.com/mdtanker/invert4geom/commit/c8283be1aa08a46da5bdd82b7f288c6145e1056b))
* minor fixes ([`bb4ef6f`](https://github.com/mdtanker/invert4geom/commit/bb4ef6fcc68b693571cbb0219b2572d998708fec))
* auto style fix ([`3eb9a4d`](https://github.com/mdtanker/invert4geom/commit/3eb9a4d1fd399f31e316bc1ab6fe728b6d620c0a))
* pre-commit fixes ([`ff4cd99`](https://github.com/mdtanker/invert4geom/commit/ff4cd993ea6187846f706b322ed07facb6c1fa11))
* minor fixes ([`dc23d05`](https://github.com/mdtanker/invert4geom/commit/dc23d05503dc78580208e0e00f45a56687c1de50))
* minor fixes to files ([`62e623a`](https://github.com/mdtanker/invert4geom/commit/62e623a203f8030123766e54a4b9fb76fc3bef5e))
* pre-commit fixes ([`e596e12`](https://github.com/mdtanker/invert4geom/commit/e596e12498a7f7b028bb2f9e99839c535c594b52))
* misc style fixes and missing imports ([`7ff25a4`](https://github.com/mdtanker/invert4geom/commit/7ff25a40a3fa10c19908617666f9fb9fad0f64d5))
* pre-commit fixes ([`ea59410`](https://github.com/mdtanker/invert4geom/commit/ea59410bea2bed62f93a59460449040baa88f6bf))

## v0.7.0 (2024-06-26)
### 💥 Breaking Changes
* remove `density_contrast` parameter from `run_inversion` and extract contrast instead

BREAKING CHANGE: the `run_inversion` function now doesn&#39;t take `density_contrast` parameter, please remove these parameters from your code! ([`2883c25`](https://github.com/mdtanker/invert4geom/commit/2883c25d2bbaa7d6555a33992abb01dd5ef557b7))
### 📦️ Build
### 🧰 Chores / Maintenance
* add codecov token ([`9d27f1b`](https://github.com/mdtanker/invert4geom/commit/9d27f1b2d861916c188ce47c4c5f91c1927002e3))
* pre-commit autoupdate to monthly ([`00189aa`](https://github.com/mdtanker/invert4geom/commit/00189aac0cbeabbffef9e15065c30a29c19c05a7))
### 📖 Documentation
* rerun the docs ([`322bcc8`](https://github.com/mdtanker/invert4geom/commit/322bcc8d0c5a954566b4dc9242d07ca86d004730))
* add variable density example to user guide ([`bdce402`](https://github.com/mdtanker/invert4geom/commit/bdce4020e497fe82a9ecbed4f3e285bc57fcc388))
* rerun docs ([`4635876`](https://github.com/mdtanker/invert4geom/commit/4635876b65bae44f575e8c4419c82710747c9986))
* add documentation to density contrast implementation ([`ca96f76`](https://github.com/mdtanker/invert4geom/commit/ca96f768a4ad4949fdd012774b0280b6db265787))
* fix error in readme ([`18b67c8`](https://github.com/mdtanker/invert4geom/commit/18b67c8e3cf738ffb3ace2b990c12d106932dda2))
* update binder env to v0.6 ([`19551d7`](https://github.com/mdtanker/invert4geom/commit/19551d763cf3a5c993f05a5b51b16b5b39af39ac))
### 🐛 Bug Fixes
* add warnings for run_inversion_workflow ([`54fc6b9`](https://github.com/mdtanker/invert4geom/commit/54fc6b9217c8864a46a39d41d90a9446bdcf135a))
* remove `zref` and `density_contrast`parameters, extract from prism layer instead

BREAKING_CHANGE: make sure to remove all `zref` and `density_contrast` parameters from the following functions: run_inversion, grav_cv_score, constraints_cv_score. ([`f995f00`](https://github.com/mdtanker/invert4geom/commit/f995f000b80718f81073a86e388800dcd653ec0e))
### ✏️ Formatting
* typo in docstring ([`d05861a`](https://github.com/mdtanker/invert4geom/commit/d05861aaf0463a099ddc976e39decf8c1864cb1e))
### Other
*  ([`117490c`](https://github.com/mdtanker/invert4geom/commit/117490cde21145a8a9b67e086736ffc9f4a3e639))

## v0.6.0 (2024-05-29)
### 💥 Breaking Changes
* make gravity CV return inversion results

BREAKING CHANGE: first return value of `grav_optimal_parameter` is ow a tuple of the inversion results. ([`1048295`](https://github.com/mdtanker/invert4geom/commit/1048295749e6b49accb96b78914c60e8d761e12f))
* regional separation methods only take gravity df not grid.

BREAKING CHANGE: use `grav_df` and `grav_data_column` arguments instead of passing a `grav_grid` to the various regional separation methods. ([`33fc5ba`](https://github.com/mdtanker/invert4geom/commit/33fc5baaffa5fae37787a44d6c9b13ff6a560ffd))
* switch keyword argument name

BREAKING CHANGE: make sure to switch all mentions of `regional_col_name` to `regional_column` in your code! ([`a71cba1`](https://github.com/mdtanker/invert4geom/commit/a71cba17b85dc864aef4dac3c15f0433b0441710))
* switch keyword argument for constraint points to `constraints_df`

BREAKING CHANGE: all use of constraint points need to be supplied via argument `constraints_df` now! ([`bf6cd34`](https://github.com/mdtanker/invert4geom/commit/bf6cd34839cbdae1f3db35b73c99ee5b47b3cefe))
* switch keyword argument name

BREAKING CHANGE: make sure to switch `run_inversion` argument  `weights_after_solver` to `apply_weighting_grid` and supply a xr.DataArray via parameter `weighting_grid`. ([`444c43d`](https://github.com/mdtanker/invert4geom/commit/444c43dce94edddd4fed490148f4ae2a229c7aa4))
* switch keyword argument name from `input_grav_column`  to `grav_data_column`

BREAKING CHANGE: make sure to switch all mentions of &#39;input_grav_column` to `grav_data_column` in your code! ([`f864c1f`](https://github.com/mdtanker/invert4geom/commit/f864c1f0fede206b2c65f0b9ac60814e8e8011f9))
* switch keyword argument name from `input_grav` to `grav_df`

BREAKING CHANGE: make sure to switch all mentions of `input_grav` to `grav_df` in your code! ([`b7656ee`](https://github.com/mdtanker/invert4geom/commit/b7656eef609b798eb2653bfd48b8157829c78a30))
### 📦️ Build
* add plotly as optional dep ([`0745e2a`](https://github.com/mdtanker/invert4geom/commit/0745e2ac3cc045bc8a6e0b1d670498eeceb7d5dc))
### 🧰 Chores / Maintenance
* add warning if using constraints for regional separation within zref/density CV ([`0e298aa`](https://github.com/mdtanker/invert4geom/commit/0e298aa331545bea917c6ce72b0105f6b29035d4))
* allow cyclic imports ([`0c8942b`](https://github.com/mdtanker/invert4geom/commit/0c8942bcb4a5ca771c1520f12f6165557d010b17))
* add optuna and plotly as optional imports ([`3545f71`](https://github.com/mdtanker/invert4geom/commit/3545f7104b6cdab6916784c93d72393139168514))
* update regional tests ([`9994cd5`](https://github.com/mdtanker/invert4geom/commit/9994cd534d37cd9c870a031901a369fec0535683))
* update tests to use `easting` and `northing` conventions ([`74eed3e`](https://github.com/mdtanker/invert4geom/commit/74eed3edda55a5318cf98bdd94788ec7a22cad55))
* check for valid registration type ([`6fb491a`](https://github.com/mdtanker/invert4geom/commit/6fb491aabd7ad9e478fa1f2ccf0e4da15d2e038a))
* raise error for wrong grid fill method ([`c7bc568`](https://github.com/mdtanker/invert4geom/commit/c7bc568ca0b1a8ef5a1db25723f97d359edcc4d0))
* codespell ignore word ([`1a8b169`](https://github.com/mdtanker/invert4geom/commit/1a8b1698f77d7dcfd1b9bfbbdae9a3eee8f4a1a7))
* pylint ignore changelog ([`b8a7030`](https://github.com/mdtanker/invert4geom/commit/b8a7030e327c52deb1371dce4901fc9a975e4ff8))
* git ignore pickle and log files ([`de4c738`](https://github.com/mdtanker/invert4geom/commit/de4c7386f1ee53fe334c0d265c56b1634b584a93))
* add make clean command ([`55fe0fa`](https://github.com/mdtanker/invert4geom/commit/55fe0fa5e62503736c215e4994024ea38b6decca))
* fix ruff linting compatibility issue ([`f82448c`](https://github.com/mdtanker/invert4geom/commit/f82448cacc0aca6dfd8e00442ca6f7d091da3f2c))
* remove comprehensive  ignore of specific md files in pre-commit ([`eb9b405`](https://github.com/mdtanker/invert4geom/commit/eb9b4050d9b580c25dde277a06cc732d6810e2e5))
* ignore md files in pre-commit prettier, ruff, blacken ([`0fade3b`](https://github.com/mdtanker/invert4geom/commit/0fade3b84b951431b0b5b3a38e4bb0bfc2c0e52f))
### 📖 Documentation
* update docstrings ([`ac9753c`](https://github.com/mdtanker/invert4geom/commit/ac9753c33d759e2bb86f9b2af2bb912624afd07d))
* add section on conventions for `Invert4Geom` ([`650e4e0`](https://github.com/mdtanker/invert4geom/commit/650e4e04614b31b3b3ec673207ad21893c2d3def))
* re-run all notebooks ([`a68431f`](https://github.com/mdtanker/invert4geom/commit/a68431f9091eab63cb9c09a41aaff7837c3a0e67))
* clarify intended use of invert4geom in README ([`0817954`](https://github.com/mdtanker/invert4geom/commit/08179547a53c992f562d55029657a6269b020742))
* fix spacing on md files ([`727c268`](https://github.com/mdtanker/invert4geom/commit/727c268fc6e51ab0d60ffe1c5d7bac55642ae370))
* update binder env ([`a6d9690`](https://github.com/mdtanker/invert4geom/commit/a6d96900a6703727bf649dcbbfb7d85579e5c502))
### 🚀 Features
* add warning for global min of optimizing eq source parameters ([`bbd5f10`](https://github.com/mdtanker/invert4geom/commit/bbd5f1054793fbc5ab51d5a08aa51abb727df237))
* add optuna optimization plotting function ([`8e64289`](https://github.com/mdtanker/invert4geom/commit/8e6428946e93511820817521034af02db1b2c8da))
* add function for performing constraint CV ([`96f74a0`](https://github.com/mdtanker/invert4geom/commit/96f74a0aaf08b0ab1eb43c04896f105fe0408149))
* updating logging in CV ([`45f1f77`](https://github.com/mdtanker/invert4geom/commit/45f1f7712d6ed583e08679529d6de0ed5d92c58d))
* save and reload best inversion results during grav CV ([`f390b7c`](https://github.com/mdtanker/invert4geom/commit/f390b7c1eb1d36f5ff8954735f7f7a3e4868c2de))
* add function to run entire inversion workflow at once ([`a7b87c7`](https://github.com/mdtanker/invert4geom/commit/a7b87c7bc45d60b18cfb8f041eddf583560330d1))
* add option to save inversion results ([`db9ef36`](https://github.com/mdtanker/invert4geom/commit/db9ef3690d12c51f95915e58b6eec960e97a7d46))
* add `regional_separation` function ([`3a8a626`](https://github.com/mdtanker/invert4geom/commit/3a8a6268828d50a053e5313dc0998937f6c3dd47))
* add equivalent sources options to `regional_constraints` ([`264c9f5`](https://github.com/mdtanker/invert4geom/commit/264c9f564ce2f51c52d6547e7d2ccb0d0750a678))
* add grav obs height option to `regional_constraints` ([`41f1a67`](https://github.com/mdtanker/invert4geom/commit/41f1a67ac7c54a650fc0418b068791285f06c44a))
* add function for creating starting topography ([`32bb475`](https://github.com/mdtanker/invert4geom/commit/32bb47599b8f9daf29853ece38929dea6132e5fc))
* add option to calculate CV scores as root median or mean square ([`818eed4`](https://github.com/mdtanker/invert4geom/commit/818eed4ad5aa125e27c902fcceeccd356ce466eb))
### 🐛 Bug Fixes
* remove numba jit decorator from `jacobian_prism` ([`b6166f5`](https://github.com/mdtanker/invert4geom/commit/b6166f5416e0c61054503ff764f1605b19f4a93d))
* add missing imports ([`bbc4825`](https://github.com/mdtanker/invert4geom/commit/bbc4825a4f66b575e3d4e3c0636603630776e726))
* update imports ([`362d79d`](https://github.com/mdtanker/invert4geom/commit/362d79ddfad41f9edab89b47cd6edfed83188716))
* use median not RMSE for constraint point minimization ([`6d1c686`](https://github.com/mdtanker/invert4geom/commit/6d1c686ee72c5752c897d621408d5009925c3e06))
* fix warning for best_spline_cv ([`abb5976`](https://github.com/mdtanker/invert4geom/commit/abb59766f4285353744f8c0f81797e69865178e3))
###  🎨 Refactor
* update plot 2 parameter CV function ([`365f5cf`](https://github.com/mdtanker/invert4geom/commit/365f5cfec04237cf7a22018730a6da77a342a7db))
* update plot convergence function ([`7efe1e7`](https://github.com/mdtanker/invert4geom/commit/7efe1e7ae029449d28ea62227b1ba3d1ceb43005))
* update plot inversion results function ([`ad2efa8`](https://github.com/mdtanker/invert4geom/commit/ad2efa87c7210ff07577914bc9a3ad698b3052c9))
* change default filename for `optimize_eq_source_params` ([`c7e4acd`](https://github.com/mdtanker/invert4geom/commit/c7e4acd9eecd8c7c580e67b2755bf71e1080b728))
* change names for saved inversion parameters ([`eaa7e55`](https://github.com/mdtanker/invert4geom/commit/eaa7e556d84ec197a0cc5615e655f6141f0c60f2))
* remove `inversion_region` from `run_inversion` ([`73c1c7c`](https://github.com/mdtanker/invert4geom/commit/73c1c7c1eb4451b53320e2f4cd93f313fe110742))
* misc changes to regional functions ([`446260e`](https://github.com/mdtanker/invert4geom/commit/446260e8e656f7615bdc215bd05e797dc3f35224))
* use `easting` and `northing` as  coordinate names ([`172b736`](https://github.com/mdtanker/invert4geom/commit/172b7368d79a5a66eca77feed4c5ed37a3391930))
* change default `regional_constraints` grid method to verde ([`ac1320b`](https://github.com/mdtanker/invert4geom/commit/ac1320bbd2e1e70bcdac93a66e0adf1f77e24be7))
* update inversion progressbar ([`d791fcb`](https://github.com/mdtanker/invert4geom/commit/d791fcb65a9a3d8af02d8f8018c1e228a410a50e))
* update progressbar for grav CV ([`1ebe1d8`](https://github.com/mdtanker/invert4geom/commit/1ebe1d8069fdfc4f0b686c02d75ab696b057267c))
* use list of l2 norms for end_inversion ([`f25e739`](https://github.com/mdtanker/invert4geom/commit/f25e739391f069fd250f534571e5548c4ebf12d9))
* change default perc_increase_limit ([`c456be8`](https://github.com/mdtanker/invert4geom/commit/c456be87fa60d2b29eeacd5a28c9822c82371f00))
* explicitly create density grid ([`7555f18`](https://github.com/mdtanker/invert4geom/commit/7555f18a61b2d2f1ca1e0d8c59c04b1781f68d8a))
* use `topo` variable for getting final topography grid ([`5596082`](https://github.com/mdtanker/invert4geom/commit/5596082967cd3f8953fc2e3df32c1586e7ccf847))
* enable CV progressbars by default ([`dc6646e`](https://github.com/mdtanker/invert4geom/commit/dc6646e121af29355c1ffa04308a0045bb67645f))
### ✏️ Formatting
* automatic style fixes ([`4a83bdd`](https://github.com/mdtanker/invert4geom/commit/4a83bddd2a27dd96b396989834a30f869b3c7934))

## v0.5.0 (2024-05-07)
### 📦️ Build
* make semantic release GHA need changelog success ([`7b7d1c4`](https://github.com/mdtanker/invert4geom/commit/7b7d1c40805617db9cf42fea3065eb32585bf035))
* add make command for binder env ([`3b5d44e`](https://github.com/mdtanker/invert4geom/commit/3b5d44e926a9925e70220344bba04c464cdbfb3d))
* add optuna-integration package ([`fa648d5`](https://github.com/mdtanker/invert4geom/commit/fa648d56e243edc934d746b8b77ac103ed0c7e10))
### 🧰 Chores / Maintenance
* ignore dtype in test utils sampling ([`f18e356`](https://github.com/mdtanker/invert4geom/commit/f18e356057a9d8b3889d3a275b273effa8993018))
* more specific dtypes in tests ([`1c587fd`](https://github.com/mdtanker/invert4geom/commit/1c587fda8de5f7cd7b2a51bda1615377de073439))
* fix test dtype issue ([`f16f955`](https://github.com/mdtanker/invert4geom/commit/f16f95541695a0ec0afdd6e4e7f70fcc68c0c284))
* use head commit for skip tests on docs ([`7f4c863`](https://github.com/mdtanker/invert4geom/commit/7f4c863560d69751a2bd09f3a18d102500666648))
* don&#39;t run test on docs commits [skip ci] ([`fedf871`](https://github.com/mdtanker/invert4geom/commit/fedf87102cb825a6054ff4e2d3d1c49a17a3e995))
* try stopping tests only for docs (not style) ([`f1cf974`](https://github.com/mdtanker/invert4geom/commit/f1cf9748c7f58cfba44f84166bfb69649ae24914))
* stop tests on docs/style [skip ci] ([`49de955`](https://github.com/mdtanker/invert4geom/commit/49de9550c07a77cefe5c40811090a87cbf07dc4c))
* still trying to stop GHA running on docs / style [skip ci] ([`7306b35`](https://github.com/mdtanker/invert4geom/commit/7306b352993af242e5c95f9566d178e29dc6390f))
* add missing &#34;)&#34; [skip-ci] ([`49918e9`](https://github.com/mdtanker/invert4geom/commit/49918e9169cf8f79779b7e53370426db50369492))
* fixing skip test on certain commits [skip ci] ([`4dfe4cf`](https://github.com/mdtanker/invert4geom/commit/4dfe4cf3bcd5dc9b212c3c75b239d414b4ea3d0c))
* stop tests running on doc / style commits ([`1a9e308`](https://github.com/mdtanker/invert4geom/commit/1a9e308ddbdfa78795b3055c466325b0c153629d))
* unpin binder env ([`e865ddc`](https://github.com/mdtanker/invert4geom/commit/e865ddc86d69b73da40fdb55fb042df792ce50ac))
* cleanup Make commands ([`482d95a`](https://github.com/mdtanker/invert4geom/commit/482d95af9f93ee79abd9e9ee51eee88ff93fb659))
* don&#39;t run tests on docs or style commits ([`30abcdf`](https://github.com/mdtanker/invert4geom/commit/30abcdf322ffb2c5bb8a29b9dfa8dfde26fb0641))
* remove pre-commit updates from changelog ([`f7ea9c8`](https://github.com/mdtanker/invert4geom/commit/f7ea9c8af19be4e432c70442482b3b692fb71618))
### 📖 Documentation
* remove duplicate thumbnail [skip ci] ([`507bd06`](https://github.com/mdtanker/invert4geom/commit/507bd06dad7479eba3266d428455c7ca96e70d4e))
* specify thumbnails ([`18fcac1`](https://github.com/mdtanker/invert4geom/commit/18fcac15d82fb1bf368de2b3d4fba7fb9db12a7e))
* remove linkcode for API ([`2d2eeff`](https://github.com/mdtanker/invert4geom/commit/2d2eeffee32173505c5e4cc66a48e2b075d38e20))
* add links to github code for API ([`883c706`](https://github.com/mdtanker/invert4geom/commit/883c706ebf83fe0265ea608cb76ee5d8e2190331))
* fix error in conf.py ([`2a7824d`](https://github.com/mdtanker/invert4geom/commit/2a7824de5c826b9e47204c12b468eea94783d490))
* fix links to code in API docs ([`7d1fa5f`](https://github.com/mdtanker/invert4geom/commit/7d1fa5fed5e0c6d6c986510ba1f4ff00e38e533c))
* update notebooks ([`a74dbee`](https://github.com/mdtanker/invert4geom/commit/a74dbeeff70660fe26c9b21f638abed18a266fb2))
* add bishop gallery example ([`7e55c7e`](https://github.com/mdtanker/invert4geom/commit/7e55c7e0874294b3d4809e3cdabeb4a494b18f4a))
* update contrib guide ([`36baf37`](https://github.com/mdtanker/invert4geom/commit/36baf37605f484c83b981c201b495bc9ea035a3e))
* update and rerun user guide notebooks ([`b1c572a`](https://github.com/mdtanker/invert4geom/commit/b1c572abbd5e69db74c0d2776eb27e32cd69566c))
* update contributing guide ([`761814c`](https://github.com/mdtanker/invert4geom/commit/761814c4eb7c0d7e4a5d9f79a18bf9c07628ba1f))
* clarify code comment ([`ef34f57`](https://github.com/mdtanker/invert4geom/commit/ef34f577fff6b38bc28b3ac68ef1d6ebed908a09))
* improve inversion docstrings ([`471a76f`](https://github.com/mdtanker/invert4geom/commit/471a76f62bbc12b7386b998cc8515dff7c29ae24))
* fix pinning issue ([`a8f0ea1`](https://github.com/mdtanker/invert4geom/commit/a8f0ea1a86de678733db806f2c499ab193c68dfb))
* pin python-semantic-release in GHA ([`c5b18da`](https://github.com/mdtanker/invert4geom/commit/c5b18da86d199c198ea5523c6b7f6a40de47b61e))
### 🚀 Features
* add termination reason to inversion progress bar ([`0274d7b`](https://github.com/mdtanker/invert4geom/commit/0274d7bc63ad24e1a83bbe10bdcee09496dd8fdc))
* add region arg to plot_inversion_topo_results ([`694b778`](https://github.com/mdtanker/invert4geom/commit/694b77898816ae7901bb3a59f4dde51ced0c1467))
* add DC-shift regional estimation method ([`2e91771`](https://github.com/mdtanker/invert4geom/commit/2e9177123a8018dcea16ddc3f2dd8be949b44e31))
* add scale and yoffset args to synthetic topography function ([`4801f33`](https://github.com/mdtanker/invert4geom/commit/4801f3358cbbcc83637aa5e7df6f5bc9aab2f750))
* add iteration progress bar to inversion ([`25a4e92`](https://github.com/mdtanker/invert4geom/commit/25a4e926e3150898eb66643b0af25881b6403edc))
* limit misfit rmse to within optional inversion region ([`b66d3c4`](https://github.com/mdtanker/invert4geom/commit/b66d3c4a66e77b79295fafd17c72083a42c4ac27))
* add dynamic convergence plot to inversion ([`7a5d5b2`](https://github.com/mdtanker/invert4geom/commit/7a5d5b2aa28c4821e610785171aaa7855879347c))
* add inversion_region to plot convergence ([`52eebdd`](https://github.com/mdtanker/invert4geom/commit/52eebddfe3871c624814e720ecd0b7ea495a8b20))
* add plot_title kwarg for plot_cv_scores ([`3cda4e0`](https://github.com/mdtanker/invert4geom/commit/3cda4e070671c10c17b3834279f3458a4c0dd283))
* fix colorscales for inversion grav results ([`9218e68`](https://github.com/mdtanker/invert4geom/commit/9218e68ded9c1e22c4945d8e948462e957755ab8))
### 🐛 Bug Fixes
* set optuna logging level to warning ([`0796fef`](https://github.com/mdtanker/invert4geom/commit/0796fef8340bc47d361e13d87dce794686283622))
* make ipython optional import ([`ce88f1a`](https://github.com/mdtanker/invert4geom/commit/ce88f1a61d09124cd8df52287e6099323644f0a3))
* allow wiggle room for enforcing confining surfaces ([`7e4e0b0`](https://github.com/mdtanker/invert4geom/commit/7e4e0b03cb0799fe5d810cc0cd7f1f37a98973ee))
* add warning message to annulus derivative calculation ([`99a7ade`](https://github.com/mdtanker/invert4geom/commit/99a7ade218a24a2e595961c1d4b5206a52520fcf))
* add noise to test regional eq sources ([`350baca`](https://github.com/mdtanker/invert4geom/commit/350baca5961beaa3dc95a6854e07e59bbd6196ac))
* replace pygmt gridding with simple set_index for plotting ([`ebf6732`](https://github.com/mdtanker/invert4geom/commit/ebf6732dbb1da0a770e3b446486336fde56b6606))
### ✏️ Formatting
* pre-commit fixes ([`116a612`](https://github.com/mdtanker/invert4geom/commit/116a612b0ff613375715e320444f85906984483e))
* pre-commit fixes ([`b110f23`](https://github.com/mdtanker/invert4geom/commit/b110f236277b60c6ed82035b5b58316885366997))
* pre-commit fixes ([`9fe3777`](https://github.com/mdtanker/invert4geom/commit/9fe377749e1918b587e630343d9e3d1baf8472ea))

## v0.4.0 (2024-02-22)
### 💥 Breaking Changes
* drop python 3.8 support

BREAKING CHANGE: ([`77cc15d`](https://github.com/mdtanker/invert4geom/commit/77cc15df813bde60a638b517d6c4d41ead76a7e0))
### 📦️ Build
* add support for Python 3.12 ([`b0b058a`](https://github.com/mdtanker/invert4geom/commit/b0b058a83cb54b1b2bc8a179842fd4dc02e55eb9))
* add binder env ([`60f15b7`](https://github.com/mdtanker/invert4geom/commit/60f15b7c84339bcf754b681257fbbfe599a02cbe))
* update pyproj specs ([`e70c4bc`](https://github.com/mdtanker/invert4geom/commit/e70c4bc1128575358603d711ca254ee041543ae2))
* add/remove dev dependencies ([`ec831f3`](https://github.com/mdtanker/invert4geom/commit/ec831f34146c611510c864bd2009da4b4ea617c5))
* add doc dependencies ([`d62d855`](https://github.com/mdtanker/invert4geom/commit/d62d8557e1c20e7a34c2d1df7204a08b9d5dc852))
### 🧰 Chores / Maintenance
* fixing semantic release action ([`cfcfdf1`](https://github.com/mdtanker/invert4geom/commit/cfcfdf1fd2e440128afd1312172d755feac672b1))
* trying to fix semantic-release action ([`75cadc3`](https://github.com/mdtanker/invert4geom/commit/75cadc332044e985bfcdc5dfbbbf3f02d7cff200))
* manually update GH action and pre-commit versions ([`46d9357`](https://github.com/mdtanker/invert4geom/commit/46d9357dc3c35f04f7c7854b0dcdcfa933892e69))
* update changelog template ([`d20c603`](https://github.com/mdtanker/invert4geom/commit/d20c603c7154e8df03a37cd130598caf2024c302))
* delete tmp optuna  file in test ([`9b266aa`](https://github.com/mdtanker/invert4geom/commit/9b266aac6b13a4c0fc771873666f45f4a08035b0))
* gitignore vscode settings ([`ad5ab8e`](https://github.com/mdtanker/invert4geom/commit/ad5ab8e9fd21e25c991192b40d3aa70868ce5e9a))
* remove unnecessary files ([`74b63ae`](https://github.com/mdtanker/invert4geom/commit/74b63ae61fe00f878c568137417646d5cf6ed96f))
* list packages after install in test GHA ([`35f05e5`](https://github.com/mdtanker/invert4geom/commit/35f05e5499abd2685ee590dee89a66fbe4b0aa22))
* include all optimization deps in testing env ([`cb2f091`](https://github.com/mdtanker/invert4geom/commit/cb2f091ce4de1e226b67e5efad91d1449ecf0294))
* add optuna to test deps ([`89e2447`](https://github.com/mdtanker/invert4geom/commit/89e2447d24934eafeb1d603d71c037cfdf7b9e90))
* update GHA test env ([`70fcbf7`](https://github.com/mdtanker/invert4geom/commit/70fcbf7ad1d08b2329b07f098eabd20e7f8dc5fd))
* revert to mamba for GHA test env ([`45c9b7b`](https://github.com/mdtanker/invert4geom/commit/45c9b7b6d7d6b7f0de4715977da9bc4a0995b846))
* add make command for GHA test env ([`9f598bd`](https://github.com/mdtanker/invert4geom/commit/9f598bd85a56684316d70bb5d8f108e82aa64f5e))
* upgrade pip for GHA test env ([`e89ba32`](https://github.com/mdtanker/invert4geom/commit/e89ba325e43db92ae66deb0a293e307478128431))
* include setuptools in GHA test env ([`aa9e569`](https://github.com/mdtanker/invert4geom/commit/aa9e569a1e7ceda842794759f2006a61e3f0b4e2))
* fix typo in pre-commit ([`b05bab5`](https://github.com/mdtanker/invert4geom/commit/b05bab5337ccf46ced49ad6c9904df860e64cb06))
* ignore bib in pre-commit ([`ecd6d93`](https://github.com/mdtanker/invert4geom/commit/ecd6d9398378e2f4c6f0e1432e6c9f0d0d22b76e))
* ignore changelog template in pre-commit ([`23ed7d4`](https://github.com/mdtanker/invert4geom/commit/23ed7d4b584db7df709d93ad9cb320e8615d6d9a))
* update changelog template ([`f3a3c9c`](https://github.com/mdtanker/invert4geom/commit/f3a3c9cc48d1698bf9e50802f58ba53d7e6bb5c5))
* remove old push GHA ([`9e2088d`](https://github.com/mdtanker/invert4geom/commit/9e2088de1f29500a412bfb9ab50ba62824461f93))
* replace mamba with pip for GHA test env install ([`421cffe`](https://github.com/mdtanker/invert4geom/commit/421cffe087d6e1a5c1b41f9d0a6cd32e7cd078f6))
* rename release GHA ([`32b6c3a`](https://github.com/mdtanker/invert4geom/commit/32b6c3a5ce7f372e0425eedb5e8fcc7735870efe))
* add semantic release GHA ([`204c7a7`](https://github.com/mdtanker/invert4geom/commit/204c7a785617a48a5aaaebf94ab38ffe6477a20b))
* update pre-commit config ([`e2dbcdd`](https://github.com/mdtanker/invert4geom/commit/e2dbcddfbc19ad3ee3d6d2cb8ba927e44db88fe8))
* add make commands ([`98d6069`](https://github.com/mdtanker/invert4geom/commit/98d60695ea85dafba46448d6fcc8e88d05f3090c))
* update RTD env and add make command ([`83d4199`](https://github.com/mdtanker/invert4geom/commit/83d419999f832efef1eb172656ecc81ffdcbaf8d))
* change dependa-bot updates to monthly ([`3284324`](https://github.com/mdtanker/invert4geom/commit/3284324bd778579871c9cbaf36db5e47991e7d43))
* update from antarctic-plots to polartoolkit ([`16bf313`](https://github.com/mdtanker/invert4geom/commit/16bf313d1b1b859750552e7b44bd2e2c031f1b92))
### 📖 Documentation
* rerun all user guide notebooks ([`d8688f2`](https://github.com/mdtanker/invert4geom/commit/d8688f2251510037c8068eb04f6a896707317078))
* fix incorrect reference styles ([`93b6cfc`](https://github.com/mdtanker/invert4geom/commit/93b6cfc7cc3dff1f79f2705eb0ebf873a8d4b3e9))
* fix bibliography references ([`606c936`](https://github.com/mdtanker/invert4geom/commit/606c936aa2a5835e1d0f5242041d0c1b21f542c2))
* fix rtd.yml ([`36376d7`](https://github.com/mdtanker/invert4geom/commit/36376d7788bde28647e4f2c392299bfab4709160))
* use pip to install RTD env ([`27c063a`](https://github.com/mdtanker/invert4geom/commit/27c063a46680910641579b7abe765ce25e131048))
* switch from autodoc to autoapi ([`e182bac`](https://github.com/mdtanker/invert4geom/commit/e182bacc09f4630b8aaa594ff3576c44c0ccb4ec))
* add binder link to docs ([`ddb80d3`](https://github.com/mdtanker/invert4geom/commit/ddb80d3a35eecba21a52186224893950ec128bfd))
* add reference .bib and md file ([`048933e`](https://github.com/mdtanker/invert4geom/commit/048933e640d0cf6ee699c15694d8db85097e76f1))
* update install instructions ([`5c051a4`](https://github.com/mdtanker/invert4geom/commit/5c051a4d1af117a0fb5c510c6cf962e3fe691f6b))
* add a discretization user guide ([`8562c06`](https://github.com/mdtanker/invert4geom/commit/8562c0663e2f8d11433073904869beca8720f0b2))
* move contributing guide ([`8d30cda`](https://github.com/mdtanker/invert4geom/commit/8d30cda624eeebd56d7297d60270d90587e83591))
* enable nbgallery for user guide ([`2bcc0aa`](https://github.com/mdtanker/invert4geom/commit/2bcc0aaadbb0a3e915f453d04388570191d6fe76))
* enable binder links ([`e134b4a`](https://github.com/mdtanker/invert4geom/commit/e134b4aa8c495ae7d77b3fbd1b46756fcd1cb583))
### 🚀 Features
* add plotting options to show_prism_layers ([`8c1874e`](https://github.com/mdtanker/invert4geom/commit/8c1874e746a19a07f4cf5815316a88ddab6c034e))
### 🐛 Bug Fixes
* resolve permission denied windows error in optuna test ([`1084b56`](https://github.com/mdtanker/invert4geom/commit/1084b569e13f8ac8cb5660be4dbd56e66af2e093))
### ✏️ Formatting
* pre-commit fixes ([`ed6fed9`](https://github.com/mdtanker/invert4geom/commit/ed6fed98b1e302546e31a6fffb0ecb420a13e164))
* remove unused type ignore ([`6313e7b`](https://github.com/mdtanker/invert4geom/commit/6313e7b1bdb56cca8c5490aa0ab0c1235bf68796))
* fix styling ([`15d96c7`](https://github.com/mdtanker/invert4geom/commit/15d96c7cc8299ea1536f77fa9ea0d92fc9353ed3))
* formatting ([`ccb890c`](https://github.com/mdtanker/invert4geom/commit/ccb890c0154186fdd11e45ed1c2ee49c9ae4daac))

## v0.3.1 (2023-12-06)
### 📦️ Build
* only install doc optional deps for nox ([`da69dcb`](https://github.com/mdtanker/invert4geom/commit/da69dcbc46c0e8895e380dc6b1696761789b4ad7))
### 🧰 Chores / Maintenance
* adds license file to pyproject.toml ([`ed882d8`](https://github.com/mdtanker/invert4geom/commit/ed882d85b8bdf790126d5a638d14b4bcdd511716))
* adds github issue templates amd comments ([`721b391`](https://github.com/mdtanker/invert4geom/commit/721b391d94f9adbdeb28aa65dbea5615231cc0ce))
### 📖 Documentation
* remove optimization deps from RTD env ([`dad0330`](https://github.com/mdtanker/invert4geom/commit/dad033019a0d78421bdc94fe928e51500624d41d))
* fix docstring for jacobian_annular ([`7b7be72`](https://github.com/mdtanker/invert4geom/commit/7b7be7286476795fbf882dcee74ba9ef98eef97d))
* fix intersphinx mapping links ([`1ca89af`](https://github.com/mdtanker/invert4geom/commit/1ca89affd93f93b365727bb014620c2fd4c4bd4a))
* add docs link to contributing file ([`9d7f830`](https://github.com/mdtanker/invert4geom/commit/9d7f830d92fc05f514766fc964c85eabe17f76dd))
* fixes minor issues in docs ([`0bca578`](https://github.com/mdtanker/invert4geom/commit/0bca578550b1bf3d3cbefa25784692032d12aac6))
* add cover figure ([`5c0d029`](https://github.com/mdtanker/invert4geom/commit/5c0d029d44c257c51856711b3823a011a6db6c25))
### 🐛 Bug Fixes
* adds encoding to open calls ([`110ff36`](https://github.com/mdtanker/invert4geom/commit/110ff368f91cc1138a48922798f42982cd352b82))
### ✏️ Formatting
* formatting ([`3ab7fb3`](https://github.com/mdtanker/invert4geom/commit/3ab7fb3bd9691816e4bc18d2205d46a37053d383))

## v0.3.0 (2023-11-30)
### 📦️ Build
* restrict xrft version

Seems to be cause issues in conda-forge feedstock, harmonica requires &gt;= 1.0, match this ([`f9a17ff`](https://github.com/mdtanker/invert4geom/commit/f9a17ff8effbbd18261434619df1b9d514badc04))
* combine optional dependencies into `all` ([`d5f038b`](https://github.com/mdtanker/invert4geom/commit/d5f038ba675c5ad2c95f26e1e589a9ab0b825489))
* add optimization optional deps ([`871a870`](https://github.com/mdtanker/invert4geom/commit/871a870acba5db7adbc51297e3c8bed274990be4))
* alter dependency versions ([`82695f5`](https://github.com/mdtanker/invert4geom/commit/82695f574fa0118af5a8874dcc53d811d99b1272))
* add new dependencies ([`8d9982c`](https://github.com/mdtanker/invert4geom/commit/8d9982cd1b5d04804cd2c9f245c3d48aabafa896))
### 🧰 Chores / Maintenance
* add Make changelog command ([`d433a9d`](https://github.com/mdtanker/invert4geom/commit/d433a9d9a4b881e7525ee44d9303a60312fae591))
* numba-progress install from pip to conda ([`59099bf`](https://github.com/mdtanker/invert4geom/commit/59099bfa07dd082d77010c4d0e10fd78524f5527))
* move changelog template location ([`2bdeccb`](https://github.com/mdtanker/invert4geom/commit/2bdeccbf43d3c4b6b494b38f4d4724d25207d729))
* fix env.yml files ([`2735f09`](https://github.com/mdtanker/invert4geom/commit/2735f093543e59f660e8bb974de14f5ba3c6dcd8))
* seperate RTD env and testing env ([`0d1621d`](https://github.com/mdtanker/invert4geom/commit/0d1621df983e3c6eedb36afdb7395974dd3f5012))
* remove build api docs to ci.yml ([`0bf7f87`](https://github.com/mdtanker/invert4geom/commit/0bf7f8780819a9376615a027c8c6924e2cc6e372))
* add sphinx-apidoc call to ci.yml ([`9d0ff3a`](https://github.com/mdtanker/invert4geom/commit/9d0ff3a231d199a30943f8b05b065693d9c1ba76))
### 📖 Documentation
* replace `Unknown` with `Other` in changelog ([`e7f274a`](https://github.com/mdtanker/invert4geom/commit/e7f274a8975b7a3ab577af0be1665fd769318389))
* customize jinja changelog template ([`203c1c8`](https://github.com/mdtanker/invert4geom/commit/203c1c8775f7ec8754f4a2b137a50c0da499c16f))
* add rst files for new modules ([`b9010e5`](https://github.com/mdtanker/invert4geom/commit/b9010e562990879cf0570db2c4380bf630982c3c))
* add estimating regional field user guide ([`7e5b69b`](https://github.com/mdtanker/invert4geom/commit/7e5b69bdb4be483909530ce0c89310474ef874dc))
* add testing instructions to contrib guide ([`9c2732d`](https://github.com/mdtanker/invert4geom/commit/9c2732d6d9949895f29f0008f287c60fa7576d99))
* add module descriptions to overview ([`ae7469d`](https://github.com/mdtanker/invert4geom/commit/ae7469d52dbaea67851e28c21ca3e033484a639f))
* fix some docstrings ([`920e129`](https://github.com/mdtanker/invert4geom/commit/920e12997354b91b2d4659f75d834e47a73686c4))
* add cross_validation to pre-commit ignore ([`b8e8e86`](https://github.com/mdtanker/invert4geom/commit/b8e8e86174442c1b16ed4b268097cc56b95c84ac))
### 🚀 Features
* add test files for all modules ([`2c11f56`](https://github.com/mdtanker/invert4geom/commit/2c11f56fdc4e349f485295f3a616c7ef3a7b0e1e))
* add regional module ([`2d52c19`](https://github.com/mdtanker/invert4geom/commit/2d52c1924d578140a47679d30a7f37b3dcc5fea5))
* add optimization module ([`a398e12`](https://github.com/mdtanker/invert4geom/commit/a398e129a243014a0aa78ac92d7c8a9fdea24cbb))
* add synthetic regional field function ([`aeb81b2`](https://github.com/mdtanker/invert4geom/commit/aeb81b2083d477a7bbd19b93dcad3c7006d91059))
* add eq_sources_score function ([`6c132d8`](https://github.com/mdtanker/invert4geom/commit/6c132d8533fac4f30e4a537059ef2adca423283b))
* add best_spline_cv function ([`3965dcb`](https://github.com/mdtanker/invert4geom/commit/3965dcbeb7ab333d8578b74cf3c686ad96908e86))
### 🐛 Bug Fixes
* optional optuna Journal import ([`e0a159f`](https://github.com/mdtanker/invert4geom/commit/e0a159f6c175419fb42728ca61fba305ab66db84))
* make optimization dep imports optional ([`e7ed8a1`](https://github.com/mdtanker/invert4geom/commit/e7ed8a16fdf052688d0fd1d1dd868a55abfb1744))
* use lock_obj to fix file store for windows ([`f835c74`](https://github.com/mdtanker/invert4geom/commit/f835c74bfaeb47491f4a3f74be6b7428e2ef63a8))
* replace psutil cpu affinity with new function

used a function from a stack overlow answer which is able to get cpu core numbers for many operating systems since psutil doesn&#39;t seem to work for MacOS or Windows. ([`f3bf61e`](https://github.com/mdtanker/invert4geom/commit/f3bf61e6f497d6a70551b098f2b7e3e257fa39d4))
* typos and formatting ([`9ce69c2`](https://github.com/mdtanker/invert4geom/commit/9ce69c21921ad0584f1fee6fc932ed24fd989864))
### Other
*  ([`99bcfac`](https://github.com/mdtanker/invert4geom/commit/99bcfac948ca9e5a169d46b1af4d8daf55b7193d))
*  ([`daf3c9c`](https://github.com/mdtanker/invert4geom/commit/daf3c9ccb32c5dbee137d20b61e959247399b0dc))
*  ([`cade158`](https://github.com/mdtanker/invert4geom/commit/cade1589062388546c6dd79b0471f04758d37111))

## v0.2.0 (2023-11-27)
### 🧰 Chores / Maintenance
* add ignore option for codespell ([`bc4f597`](https://github.com/mdtanker/invert4geom/commit/bc4f5973b89f4b27885b2037b8d861732cf6240f))
* add semantic-release changelog template ([`bc93159`](https://github.com/mdtanker/invert4geom/commit/bc93159f5cd3261d5049c9b1c12c228abec3ebd9))
* add semantic release check to makefile ([`d3bb077`](https://github.com/mdtanker/invert4geom/commit/d3bb077c038394a8c66e21d9916cf38ab9c48c96))
### 📖 Documentation
* minor changes ([`c29c742`](https://github.com/mdtanker/invert4geom/commit/c29c74239237ed08013f6eaf1df2e90c874971c4))
*  new user guide notebook

adds notebook which combines damping, density, and zref cross validations, as well as using a starting model and weighting grid. ([`1ad026a`](https://github.com/mdtanker/invert4geom/commit/1ad026a188413934c298aaff20f7eda306a3cf59))
* update user guide notebooks

To run faster, this lowers the resolution of the user guide notebooks. It also adds synthetic noise to all the examples. ([`d63c410`](https://github.com/mdtanker/invert4geom/commit/d63c410c5083078c0d20e1e4f1f840a83b58bbdf))
* update some docstrings ([`0bf4d7b`](https://github.com/mdtanker/invert4geom/commit/0bf4d7bd35900285ea5c42c29d2c23fa451f8f68))
* add references to docstrings ([`d0357a8`](https://github.com/mdtanker/invert4geom/commit/d0357a819db975295302d6aada0ba14206258aaf))
* add emojis to homepage ([`eda2938`](https://github.com/mdtanker/invert4geom/commit/eda29385c90692600469a5d9a48c123c9b2c5842))
* set nb execute to never for docs ([`ef58202`](https://github.com/mdtanker/invert4geom/commit/ef58202e4ab1e853be4924ddda3355bc902ffd00))
* edit the user guide notebooks ([`aec1fe6`](https://github.com/mdtanker/invert4geom/commit/aec1fe67409dd2db0da602483bda98c3cc6e3df0))
* add cross validation api docs ([`03c38d6`](https://github.com/mdtanker/invert4geom/commit/03c38d6fd89d8b1d3506cc865596dbaf75afb108))
* remove tqdm mapping ([`088409e`](https://github.com/mdtanker/invert4geom/commit/088409e3ae897e213b6c51d4d59ba5c5a2300b8b))
### 🚀 Features
* add 2 parameter cross validation plotting ([`2d1269e`](https://github.com/mdtanker/invert4geom/commit/2d1269ebacbe4149642c8cd3484b7ffa65554b83))
* add contaminate function for synthetic noise ([`3a1cf8d`](https://github.com/mdtanker/invert4geom/commit/3a1cf8d1b081057d0f8e19e611bab30f807fb4a5))
### 🐛 Bug Fixes
* add reference level to iteration plots ([`960eb44`](https://github.com/mdtanker/invert4geom/commit/960eb44cc5956b9795dbc9b18bcdd627eec7ea17))
### ✏️ Formatting
* formatting ([`66b0a56`](https://github.com/mdtanker/invert4geom/commit/66b0a56aaba2b590e2594fd2bd31735823959128))
### Other
*  ([`86c611b`](https://github.com/mdtanker/invert4geom/commit/86c611b322de101c79fa8072718418c8a0b1a56f))

## v0.1.20 (2023-11-24)
### 🐛 Bug Fixes
* forcing a new patch ([`9a73194`](https://github.com/mdtanker/invert4geom/commit/9a73194ea5b0e22e4422ed04f2c180c15de29867))
### Other
*  ([`2546ee4`](https://github.com/mdtanker/invert4geom/commit/2546ee401201509409c932cf8111370142911599))

## v0.1.19 (2023-11-24)
### 🧰 Chores / Maintenance
* remove scm version files ([`1f99c55`](https://github.com/mdtanker/invert4geom/commit/1f99c558617a85c050c56aaab0b6c799013f6fc8))
### 🐛 Bug Fixes
* add zref and density args to inversion (#24)

* fix: add zref and density args to inversion

* fix: removes references to density and zref

* fix: remove python 3.12 support, add test for 3.10

* chore: specify semantic release options

* chore: remove pypy from testing versions ([`c62f189`](https://github.com/mdtanker/invert4geom/commit/c62f1891fbbf14b9d9f092758fbf73f7b542621a))
### Other
*  ([`fb5f7b2`](https://github.com/mdtanker/invert4geom/commit/fb5f7b277d0b99fc901ca07b75bae4a9853ff008))

## v0.1.18 (2023-11-23)
### 🐛 Bug Fixes
* use PAT instead of GH tocken ([`f2cb0d1`](https://github.com/mdtanker/invert4geom/commit/f2cb0d16e23535d56a9378113639814f2140a640))

## v0.1.17 (2023-11-23)
### 🧰 Chores / Maintenance
* finally have release.yml fixed?! ([`6654e98`](https://github.com/mdtanker/invert4geom/commit/6654e98181e57a433f55870a0470dfa9ea56caf0))
* fixing release.yml ([`2a381b3`](https://github.com/mdtanker/invert4geom/commit/2a381b33087fcb5d74550231fd2d57816c1bb105))
### 🐛 Bug Fixes
* force version bump: ([`7c46909`](https://github.com/mdtanker/invert4geom/commit/7c469091dd542ce6ef5eee319f4005c1ca01d5d5))

## v0.1.16 (2023-11-23)
### 🧰 Chores / Maintenance
* fixing GH actions ([`1517933`](https://github.com/mdtanker/invert4geom/commit/1517933c19f1cfe8b2b156409bdee25aca0c683c))
### 🐛 Bug Fixes
* force version bump: ([`f357da5`](https://github.com/mdtanker/invert4geom/commit/f357da57bab602c5d6ef2e7eb81ffe60d479265a))

## v0.1.15 (2023-11-23)
### 🐛 Bug Fixes
* seperate push and release jobs ([`2f0284a`](https://github.com/mdtanker/invert4geom/commit/2f0284a1b25c831212c17ad32d45a5ba6a00217b))

## v0.1.14 (2023-11-23)
### 🐛 Bug Fixes
* refine concurrency ([`a54541f`](https://github.com/mdtanker/invert4geom/commit/a54541f4b92b8fddccbb2ad4911ed2b6bc87dfae))

## v0.1.13 (2023-11-23)
### 🐛 Bug Fixes
* remove permission from inside steps ([`42cadef`](https://github.com/mdtanker/invert4geom/commit/42cadef257b43a891566bc19fa4b4eb33a4eb3c8))
* move environment outside of steps ([`9773658`](https://github.com/mdtanker/invert4geom/commit/9773658adc392c6a90f93bb36e5a6a261a7d4b21))
* reconfigure cd.yml ([`5fa82ed`](https://github.com/mdtanker/invert4geom/commit/5fa82ed7285fbfbeb29431001b050f8fc64c322b))

## v0.1.12 (2023-11-23)
### 🐛 Bug Fixes
* fixing cd.yml ([`ee7efde`](https://github.com/mdtanker/invert4geom/commit/ee7efdefad907bf9eab1904ef0a18a08be12df74))
* fixing publish action ([`182d8a9`](https://github.com/mdtanker/invert4geom/commit/182d8a9e6481db4294f216f211d41c68091742a1))

## v0.1.11 (2023-11-23)
### 🐛 Bug Fixes
* publish pypi only if tag ([`1186e3e`](https://github.com/mdtanker/invert4geom/commit/1186e3ee650dfcc705789dc3c2cbf603d0a5cd24))

## v0.1.10 (2023-11-23)
### 🧰 Chores / Maintenance
* fixing cd.yml ([`6fcf6d9`](https://github.com/mdtanker/invert4geom/commit/6fcf6d99e69cb6d34aa79396e2401e205b5e5bee))
* fixing cd.yml ([`d63dd4c`](https://github.com/mdtanker/invert4geom/commit/d63dd4cf18ec743ee76c1f9cf8ff20082de351a2))
### 🐛 Bug Fixes
* force a version increment ([`8dfd424`](https://github.com/mdtanker/invert4geom/commit/8dfd424ce38f8eb417582130ddf87ce910933f93))

## v0.1.9 (2023-11-23)
### 🐛 Bug Fixes
* trying to fix PSR making github release ([`b777890`](https://github.com/mdtanker/invert4geom/commit/b77789046db794a21eaa09d97c8079f15f371199))

## v0.1.8 (2023-11-23)
### 🐛 Bug Fixes
* replace hynek build with PSR build ([`f40cfe9`](https://github.com/mdtanker/invert4geom/commit/f40cfe9cce1fef2ee27d88c67862937b13b6bbc5))

## v0.1.7 (2023-11-23)
### 🧰 Chores / Maintenance
* create github release in cd.yml ([`7b717e7`](https://github.com/mdtanker/invert4geom/commit/7b717e7dfa1d82eb8d7bb3d98a4ab898f5f61461))
* create github release in cd.yml ([`4212fdc`](https://github.com/mdtanker/invert4geom/commit/4212fdc2b5a9f510b49e34aa6849bc5d5ea96e5d))
* create github release in cd.yml ([`1e78ab3`](https://github.com/mdtanker/invert4geom/commit/1e78ab3bf191e67a764b649e37ad86de4fd8ca86))
* create github release in cd.yml ([`b6888dd`](https://github.com/mdtanker/invert4geom/commit/b6888dd27f3a897caf3f0710b095354f68211526))
* create github release in cd.yml ([`65a4b93`](https://github.com/mdtanker/invert4geom/commit/65a4b93952369036899912188f154d4d490d703c))
* create github release in cd.yml ([`eb1775d`](https://github.com/mdtanker/invert4geom/commit/eb1775d677baf9c43b5861bc27e1a4dd7631508c))
* create github release in cd.yml ([`9f08208`](https://github.com/mdtanker/invert4geom/commit/9f082087a8aa1f83c60dab2a9d16fe8cf1f72df4))
* fixing cd.yml ([`074739f`](https://github.com/mdtanker/invert4geom/commit/074739f620b3e8de5af299b8df807b2b37081735))
* fixing cd.yml ([`da6eaa6`](https://github.com/mdtanker/invert4geom/commit/da6eaa69c9b7d53a5326bdc6849082425386e5f0))
* fixing cd.yml ([`0ae1fbe`](https://github.com/mdtanker/invert4geom/commit/0ae1fbecf01bd85ab7e98694d06a878a90883d28))
* fixing cd.yml ([`43528b4`](https://github.com/mdtanker/invert4geom/commit/43528b46ed6fff295a21835ddf6b063750f07dae))
* fixing cd.yml ([`afa57ea`](https://github.com/mdtanker/invert4geom/commit/afa57ea031c4f51f2e956a12a3182837d27a8a96))
* fixing cd.yml ([`0aa148f`](https://github.com/mdtanker/invert4geom/commit/0aa148f602229e847b4552fc5074b9b295866908))
* fixing cd.yml ([`16b83a9`](https://github.com/mdtanker/invert4geom/commit/16b83a9e5c543e2b25174d3aee7476345820a182))
* fixing cd.yml ([`695909f`](https://github.com/mdtanker/invert4geom/commit/695909f7de895f8b77d56245db6fa5b8b5a57e4d))
* fixing cd.yml ([`1a2fea3`](https://github.com/mdtanker/invert4geom/commit/1a2fea33b7b406461c1855e4ca13fa8cae4f3345))
* fixing cd.yml ([`32b63a3`](https://github.com/mdtanker/invert4geom/commit/32b63a3c3b4eae6535f3cf5796b3d3f2d2f4875e))
* fixing cd.yml ([`ff243dd`](https://github.com/mdtanker/invert4geom/commit/ff243ddf61963451155cdecb11b5f9f99cfb98f9))
* fixing cd.yml ([`276f3e2`](https://github.com/mdtanker/invert4geom/commit/276f3e2ab0ca74735f38deef939cbcb112ef65e7))
* fixing cd.yml ([`e1084ba`](https://github.com/mdtanker/invert4geom/commit/e1084ba74d028b0b9bfe47639cfb85a946344c88))
* fixing cd.yml ([`c20b9ab`](https://github.com/mdtanker/invert4geom/commit/c20b9ab40f7aee2f66684bbd2c5f0e81f554b7ad))
* fixing cd.yml ([`9957ce6`](https://github.com/mdtanker/invert4geom/commit/9957ce608c00044155d8bb4aa0f3db1b6ac3bfef))
* fix release.yml ([`283b0d6`](https://github.com/mdtanker/invert4geom/commit/283b0d6367cfde096b7322cd5d7655e505cc95ec))
* fixing release.yml ([`498164d`](https://github.com/mdtanker/invert4geom/commit/498164d8fa889f7e03801b2bc7f58081195bd64d))
* fixing release.yml ([`a30266f`](https://github.com/mdtanker/invert4geom/commit/a30266f0bdafbb8ab592f9a650bdc60e24836dbb))
### 🐛 Bug Fixes
* fake fix commit to test semantic-release ([`ffcd3ed`](https://github.com/mdtanker/invert4geom/commit/ffcd3ed2b2efaf434731710d8c8d23f94c8f121a))
### Other
*  ([`3be81dc`](https://github.com/mdtanker/invert4geom/commit/3be81dc32b700dcbc5600fb3c81a09c07064e809))
*  ([`8ad6482`](https://github.com/mdtanker/invert4geom/commit/8ad648201fd41c395852716c8f938250c4839fb7))

## v0.1.6 (2023-11-22)
### 🧰 Chores / Maintenance
* publish to pypi on all commits to main (#18) ([`5b0b9f3`](https://github.com/mdtanker/invert4geom/commit/5b0b9f3cac9453b7c5aeeb45ba0ac3be0d4fca6e))
### 🐛 Bug Fixes
* add personal access token to github action (#20) ([`cc0f1ed`](https://github.com/mdtanker/invert4geom/commit/cc0f1ed4983103195b342c27cf85b1ea9245317e))
* enable semantic release (#19) ([`b318687`](https://github.com/mdtanker/invert4geom/commit/b318687220afa97e19f7b50d969000de9e367a1b))

## v0.1.5 (2023-11-23)
### 🧰 Chores / Maintenance
* only publish to pypi on tags (#17) ([`ae9cbac`](https://github.com/mdtanker/invert4geom/commit/ae9cbaca2b0b7c81a13669ea58eb6b6dfcf0d1ad))

## v0.1.4 (2023-11-23)
### 🧰 Chores / Maintenance
* manually increment version to test GA (#16) ([`dc18fdb`](https://github.com/mdtanker/invert4geom/commit/dc18fdb0fb6dd7f2148392fc637e929d3c156bbb))

## v0.1.3 (2023-11-23)
### 🧰 Chores / Maintenance
* enable publish to pypi without testpypi (#15) ([`c0d2969`](https://github.com/mdtanker/invert4geom/commit/c0d29695c826ac99c06e7c41ca121a6a76c35f25))

## v0.1.2 (2023-11-23)
### 🧰 Chores / Maintenance
* fixing release github action (#14)

manually increment version to test pypi release action is triggered ([`c9fc4c7`](https://github.com/mdtanker/invert4geom/commit/c9fc4c7e7f6cad6c0d9d948b806b51453adaeb33))
* still fixing release.yml (#12)

update pypi and testpypi environment name info ([`db9132a`](https://github.com/mdtanker/invert4geom/commit/db9132a5216385cbc67dbf14dda1a03f9e07cfd3))
* still fixing release.yml (#11)

refine if statements ([`24d3fca`](https://github.com/mdtanker/invert4geom/commit/24d3fca9aa1a50bda04cb95b0436bb425dce9a2f))
* trying to fix release.yml (#9)

makes release.yml run on all pushes to main instead of just published pushes. ([`5554585`](https://github.com/mdtanker/invert4geom/commit/55545854a3cd2cda06a2de503cb62470c63964ce))
### Other
*  ([`debe987`](https://github.com/mdtanker/invert4geom/commit/debe9879a55ed08509a3a901e0297d1c356fbacc))
*  ([`9afcbfd`](https://github.com/mdtanker/invert4geom/commit/9afcbfd2f55511e95fff478b304766363287614b))

## v0.1.1 (2023-11-23)
### 🧰 Chores / Maintenance
* trying to fix release.yml issues (#8)

* chore: trying to fix reseal.yml issues

* style: pre-commit fixes

---------

Co-authored-by: pre-commit-ci[bot] &lt;66853113+pre-commit-ci[bot]@users.noreply.github.com&gt; ([`aa8760a`](https://github.com/mdtanker/invert4geom/commit/aa8760ab455060abd3998fc3ce154563080e7ec9))
* sets up python-semantic-release (#7)

Changes dynamic version to a manually typed version 0.1.0, and tells python-semanitc-release where this version is specified. Removes setuptools_scm. Configures semanitic-release. ([`612b319`](https://github.com/mdtanker/invert4geom/commit/612b31914490201e99019537c3e9997b1bea74d2))
### Other
*  ([`5a8ea9d`](https://github.com/mdtanker/invert4geom/commit/5a8ea9de80b4a104bb32f65e8c8067919a34b3ce))
*  ([`4a08da8`](https://github.com/mdtanker/invert4geom/commit/4a08da8ae87e03a747fbc58df804dfc6d6c43304))
*  ([`315f8f7`](https://github.com/mdtanker/invert4geom/commit/315f8f77d0d2c16a4e6ce7455e878ce6973b2364))
*  ([`d8ece67`](https://github.com/mdtanker/invert4geom/commit/d8ece67eda2833d8ed8b102a34cd5da9831af96c))
*  ([`0304b87`](https://github.com/mdtanker/invert4geom/commit/0304b87235e8d652c18fd7a2df962277cd65ad91))
*  ([`9eecfea`](https://github.com/mdtanker/invert4geom/commit/9eecfeaf1ee42f735f469e073fc4281e52b798e9))
*  ([`8c5ed74`](https://github.com/mdtanker/invert4geom/commit/8c5ed748b5963322a1f36a11f8579f5a4363f3f9))
*  ([`440dcf0`](https://github.com/mdtanker/invert4geom/commit/440dcf0d534d825685094e5cd35a425dbe2237e0))
*  ([`993ba2c`](https://github.com/mdtanker/invert4geom/commit/993ba2cf6f90e3da88deecb4d84c5b6fbe2feeed))
*  ([`6d9b9df`](https://github.com/mdtanker/invert4geom/commit/6d9b9df9794d0598fcd9a0717801979fdd21894a))
*  ([`4f812a3`](https://github.com/mdtanker/invert4geom/commit/4f812a3a102c88ef15415c414894e4b680234b43))
*  ([`55976da`](https://github.com/mdtanker/invert4geom/commit/55976dac8deedda6626923e83f77e90184434b3c))
*  ([`c045ead`](https://github.com/mdtanker/invert4geom/commit/c045ead603e7a65956376ff3b847ae18de6ccca1))
*  ([`5fa1cd0`](https://github.com/mdtanker/invert4geom/commit/5fa1cd0b90a6df4019075b33606e5097e0c4bd8f))
*  ([`93b4bb6`](https://github.com/mdtanker/invert4geom/commit/93b4bb613870ccfbdc671ea0a7babf38ff5f50cf))
*  ([`d6a238c`](https://github.com/mdtanker/invert4geom/commit/d6a238c3bbc2decff6602db218828801bef93120))
*  ([`9dd1e2a`](https://github.com/mdtanker/invert4geom/commit/9dd1e2ac9daaf4fd37e665abd9713de928696919))
*  ([`5d43c41`](https://github.com/mdtanker/invert4geom/commit/5d43c41dc2bb71721ff78861b651d1d51ee184f6))
*  ([`f732336`](https://github.com/mdtanker/invert4geom/commit/f7323362c0dba800a37099f01a0085c6f6ddb098))
*  ([`5df4203`](https://github.com/mdtanker/invert4geom/commit/5df4203e5deebb8a806ad7901c2bb0febb3bfdbc))
*  ([`1dcd264`](https://github.com/mdtanker/invert4geom/commit/1dcd264bb12a61cf4280cb11412ed605c897ff10))
*  ([`f9cda79`](https://github.com/mdtanker/invert4geom/commit/f9cda79c8705574621c9590f21444189d4446653))
*  ([`70b98ac`](https://github.com/mdtanker/invert4geom/commit/70b98acf91dde7cbe98b565848eeec959e267672))
*  ([`996b841`](https://github.com/mdtanker/invert4geom/commit/996b84107572cb7d1c3579625384f1e008ccfbcf))
*  ([`5e3226d`](https://github.com/mdtanker/invert4geom/commit/5e3226d671ad68bfb8eaac6e413490755d59eded))
*  ([`4004082`](https://github.com/mdtanker/invert4geom/commit/400408266d23a80f9d623141b2cb49db7570b5ff))
*  ([`e329670`](https://github.com/mdtanker/invert4geom/commit/e329670858005078ca5a43076f9751321c0e16a8))
*  ([`9a87891`](https://github.com/mdtanker/invert4geom/commit/9a878919aa55a8c8a133d2bfdb2975179f2d73bf))
*  ([`b7d3849`](https://github.com/mdtanker/invert4geom/commit/b7d384918d712ff57093fded8ce1c7356a7f556f))
*  ([`11a397e`](https://github.com/mdtanker/invert4geom/commit/11a397eb65ccd523879e8985907f639c15072892))
*  ([`0746f53`](https://github.com/mdtanker/invert4geom/commit/0746f53716be4bd31a035bf1cf3488bbd9e9815b))
*  ([`8636362`](https://github.com/mdtanker/invert4geom/commit/8636362f15212d2a3f0f45e5fbd9a93a850d0949))
*  ([`6cfd46c`](https://github.com/mdtanker/invert4geom/commit/6cfd46cf4213801454e2bc68e4d61a25fa1f83f1))
*  ([`11ada50`](https://github.com/mdtanker/invert4geom/commit/11ada501226062bceb85ce333ef91fbc4a4c7489))
*  ([`ed45047`](https://github.com/mdtanker/invert4geom/commit/ed45047bec99a4d82ba01a9ef4cbc32cc76368e8))
*  ([`b51f180`](https://github.com/mdtanker/invert4geom/commit/b51f1800f703cd3bacec3d4fd7d7e1c96b14ccca))
*  ([`a0884b1`](https://github.com/mdtanker/invert4geom/commit/a0884b1d7751cbf959422b030e274f0f521e7e94))
*  ([`a37ffff`](https://github.com/mdtanker/invert4geom/commit/a37ffff06119fee42e07db295c8a8a757cd29cdf))
*  ([`36545a7`](https://github.com/mdtanker/invert4geom/commit/36545a7fddbf3ddf966c6f8be1242f4402db0d87))
*  ([`9541c32`](https://github.com/mdtanker/invert4geom/commit/9541c32f987a6598687e3726e9e27535eec62ebb))
*  ([`0a3c520`](https://github.com/mdtanker/invert4geom/commit/0a3c5207a20412fa3a5b14d334d6ffa990e4f207))
*  ([`3c97a79`](https://github.com/mdtanker/invert4geom/commit/3c97a79acb0be8107561ded73a91653861796748))
*  ([`7f6ac72`](https://github.com/mdtanker/invert4geom/commit/7f6ac7229f41f2c88d310af918e24e9c6f4df061))
*  ([`0cd2ed3`](https://github.com/mdtanker/invert4geom/commit/0cd2ed3c99efc3e6546cf2792e9449772c85a125))
*  ([`825941d`](https://github.com/mdtanker/invert4geom/commit/825941dc19994495ba2bc885eb2faaa1b598361d))
*  ([`1ae95bd`](https://github.com/mdtanker/invert4geom/commit/1ae95bd5be06d7e9eb7503037df1b8a775e6f7bf))
*  ([`6ad903c`](https://github.com/mdtanker/invert4geom/commit/6ad903c043ab520aa6caff176ebaf756802f886f))
*  ([`6c50649`](https://github.com/mdtanker/invert4geom/commit/6c5064988e0b6b892e068311709a57ca981bcdd3))
*  ([`eecc95e`](https://github.com/mdtanker/invert4geom/commit/eecc95ec7d82f2b6624395f456907515959aa415))
*  ([`467a428`](https://github.com/mdtanker/invert4geom/commit/467a4280bdd40dca019991b44836024443b215cb))
*  ([`6f626ae`](https://github.com/mdtanker/invert4geom/commit/6f626aea9f217dc01a3f7cc35e58a14a3863fbd5))
*  ([`3841709`](https://github.com/mdtanker/invert4geom/commit/3841709179d3c8f8b2f2ddaff92e91b39f18cf41))
*  ([`500f8f4`](https://github.com/mdtanker/invert4geom/commit/500f8f4e50b6e5caa2652984d0b3991feb6d699e))
*  ([`473e087`](https://github.com/mdtanker/invert4geom/commit/473e08762088212de43aa5e3e3dc1e3946dfc59a))
*  ([`ad4d6a1`](https://github.com/mdtanker/invert4geom/commit/ad4d6a186885a21f329198e18403b4b4a96207d6))
*  ([`6b78411`](https://github.com/mdtanker/invert4geom/commit/6b784114c7093a2d99d87c44375c21c99f19451f))
*  ([`809bc4d`](https://github.com/mdtanker/invert4geom/commit/809bc4d23c7833c6d140d6a9c33a3f516996e680))
*  ([`3c4c13d`](https://github.com/mdtanker/invert4geom/commit/3c4c13d1c1da694a11fe96ed01087f58ce1734db))
*  ([`5625c13`](https://github.com/mdtanker/invert4geom/commit/5625c1311223563bf0a1e2805db967182087a9ec))
*  ([`9713502`](https://github.com/mdtanker/invert4geom/commit/9713502e2af55bb27a6841930ac8625e1d01078f))
*  ([`d97006e`](https://github.com/mdtanker/invert4geom/commit/d97006e29e92356b9d451c1319c0f2bdbb53c967))
*  ([`b2bc056`](https://github.com/mdtanker/invert4geom/commit/b2bc056cf8b0413a3b85936abf12094a49fda5f9))
*  ([`cf8d0ca`](https://github.com/mdtanker/invert4geom/commit/cf8d0ca430f4f603d36f87fbbd8369b4f99ef442))
*  ([`bc1e445`](https://github.com/mdtanker/invert4geom/commit/bc1e44538787e94d749a37f8fe868505b8adbf96))
*  ([`f51370a`](https://github.com/mdtanker/invert4geom/commit/f51370a85623819f085682546e1097b165842f39))
*  ([`5b1801b`](https://github.com/mdtanker/invert4geom/commit/5b1801bc386d8ba59784ee875abd322b52a093a4))
