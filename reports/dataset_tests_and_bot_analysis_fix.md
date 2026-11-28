# Dataset-Dependent Tests & Analyze Post Fix Report

**Date:** 2026-11-28  
**Repository Root:** `./` (project root)  
**Git Operations:** None performed (working tree modifications only; no commit, push, merge, reset, or branch switch).

---

## 1. Root Cause of Dataset-Dependent Test Failures (`tests/test_models.py`)

Five tests in `tests/test_models.py` (`test_preprocess_returns_dataframe`, `test_preprocess_has_bot_label`, `test_account_features_extracted`, `test_get_features_and_labels`, and `test_prepare_rnn_input`) previously failed at fixture setup (`sample_df -> load_bot_dataset()`) with:

```text
FileNotFoundError: No bot dataset found. Please place 'twitter_human_bots_dataset.csv' in the project root directory.
```

Inspection identified three contributing causes:
1. **Dataset Filename Variant:** The dataset file placed in the project root is named `twitter_human_bots_dataset-1.csv` (`37,438` rows, `20` columns, `11.3 MB`), whereas `src/data/loader.py` only checked `"twitter_human_bots_dataset.csv"` and `"bot_detection_data.csv"`.
2. **Working-Directory Dependency:** `src/data/loader.py` checked bare relative strings via `os.path.exists(...)` against `os.getcwd()` rather than resolving paths relative to the project root (`pathlib.Path(__file__).resolve().parents[2]`). Invoking `pytest` or scripts from any directory other than the project root failed to locate both datasets and saved model artifacts.
3. **Git Exclusion Risk for Cross-Machine Sharing:** `.gitignore` previously ignored all `*.csv` files without exempting `twitter_human_bots_dataset*.csv`, which prevented the dataset from being tracked when sharing the project with another machine. Additionally, `tests/test_meta_classifier.py` assumed `Data/processed/meta_dataset_synthetic.csv` already existed on disk even on fresh clones where `/Data/processed/` is gitignored.

---

## 2. Root Cause of Streamlit "Analyze Post" Failure

When clicking **Analyze Post**, `app/streamlit_app.py` displayed:

```text
An unexpected error occurred while running the analysis pipeline.
```

Tracing `app/streamlit_app.py` → `SocialThreatPipeline.run()` → `FakeNewsDetector` / `BotDetector` / `MetaSupervisorAgent` / `MetaLogisticRegression` revealed:

1. **Cross-Machine `scikit-learn` / `NumPy` Deserialization Skew (`AttributeError` / `ModuleNotFoundError`):**
   - `models/saved/meta_logreg_synthetic.joblib` was serialized under **`scikit-learn 1.9.1`** and **`NumPy 2.5.4`**, while Agent 1 (`fake_news_model.joblib`, `tfidf_vectorizer.joblib`) was serialized under `scikit-learn 1.6.1` and Agent 2 (`bot_scaler.joblib`) under `scikit-learn 1.7.2`.
   - In `scikit-learn >= 1.8`, `LogisticRegression` removed the `multi_class` attribute from instance `__dict__` and set `penalty = 'deprecated'`.
   - On another computer running `scikit-learn 1.6.x` or `1.7.x` (permitted by `requirements.txt`), `joblib.load()` succeeded at app startup (showing all 3 agents as loaded), **but clicking "Analyze Post"** invoked `LogisticRegression.predict_proba()`, which accessed `self.multi_class` and raised:
     `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`.
   - Because `app/streamlit_app.py` only caught `(TypeError, ValueError, FileNotFoundError, RuntimeError)` in its primary handler, `AttributeError` fell into the bare `except Exception:` block — showing `"An unexpected error occurred while running the analysis pipeline."` while suppressing the traceback completely (no `logger.exception()` call existed).
   - Similarly, unpickling NumPy 2.x `.joblib` files on NumPy 1.24–1.26 raises `ModuleNotFoundError: No module named 'numpy._core'`.
2. **PyTorch Lazy CUDA/cuDNN Initialization Risk on Other Machines (`BotDetector`):**
   - `BotDetector.__init__()` previously defaulted `self.device` to `"cuda"` whenever `torch.cuda.is_available()` was `True`. On Windows machines with mismatched GPU architectures or missing cuDNN DLLs, weight loading can succeed at startup while the first `nn.GRU` forward pass on clicking **Analyze Post** fails (`OSError` / CUDA kernel error).
3. **Working-Directory Fragility & Edge-Case Guards:**
   - Default model paths in `FakeNewsDetector`, `BotDetector`, `MetaLogisticRegression`, and `SocialThreatPipeline` were relative to `os.getcwd()` rather than the project root.
   - `validate_account_metadata()` did not catch `OverflowError` on arbitrarily large integer inputs (`float(val)`), and `effective_case_id` deduplication in `app/streamlit_app.py` only checked `analysis_history` once rather than looping across both `analysis_history` and `ReviewQueue`.
4. **Confirmation Regarding Training CSV During Inference:**
   - Normal Bot inference (`BotDetector.predict()` / `SocialThreatPipeline.run()`) uses `engineer_account_features()`, `bot_scaler.joblib`, `bot_input_meta.joblib`, and `bot_rnn_model.pth`. It **never** loads `twitter_human_bots_dataset*.csv` during inference.

---

## 3. Files Modified & Exact Fixes

| File | Exact Fix Applied |
|---|---|
| `src/utils/helpers.py` | Added `PROJECT_ROOT` (`Path(__file__).resolve().parents[2]`), cross-platform `resolve_project_path()`, and `ensure_numpy_pickle_compat()` (`numpy._core` alias shim for NumPy 1.x/2.x compatibility) inside `load_model()`. |
| `src/data/loader.py` | Added `resolve_bot_dataset_path()` resolving paths relative to `PROJECT_ROOT` via `pathlib.Path`; added support for `twitter_human_bots_dataset.csv`, `twitter_human_bots_dataset-1.csv` (and `twitter_human_bots_dataset*.csv` variants), `Data/raw/` subdirectories, and `bot_detection_data.csv`. |
| `src/detection/bot_detector.py` | Resolved model, scaler, and metadata paths via `resolve_project_path()`; defaulted inference `device` to `"cpu"` for deterministic cross-machine compatibility (with forward-pass verification and automatic CPU fallback if `"cuda"` is explicitly requested); enabled `ensure_numpy_pickle_compat()`. |
| `src/features/temporal_features.py` | Added exact `(X - mean_) / scale_` fallback in `apply_scaler()` if a cross-version `StandardScaler` raises `AttributeError`. |
| `src/detection/fake_news_detector.py` | Resolved model and vectorizer paths via `resolve_project_path()`, added `ensure_numpy_pickle_compat()`, and added `logger.exception()` on artifact load failure. |
| `src/models/logistic_regression.py` | Made `_build_pipeline()` inspect `LogisticRegression.__init__` signature across `scikit-learn 1.6–1.9+`; patched missing cross-version attributes (`multi_class`, `penalty`) in memory on `load()` (without modifying disk artifacts); added exact fitted-parameter softmax fallback (`_manual_pipeline_proba`, verified to `0.0` max absolute difference) in `predict_proba()` on `AttributeError`. |
| `src/moderation/pipeline.py` | Resolved `meta_model_path` via `resolve_project_path()` and converted `OverflowError` in `validate_account_metadata()` into a clean `ValueError`. |
| `app/streamlit_app.py` | Configured `logging` (`logger = logging.getLogger(__name__)`); added `logger.exception()` across model loading, input validation, `pipeline.run()` errors, and HRQ resolution while keeping friendly user-facing messages in the Streamlit UI; hardened `effective_case_id` uniqueness loop across both `analysis_history` and `ReviewQueue`. |
| `.gitignore` | Added `!twitter_human_bots_dataset.csv` and `!twitter_human_bots_dataset-*.csv` so the bot dataset is included when sharing the project across machines while keeping `/Data/processed/` excluded. |
| `requirements.txt` | Updated `streamlit>=1.45` (required for `width="stretch"` UI parameters) alongside explicit `altair>=5.0` and all core dependencies. |
| `tests/test_models.py` | Resolved model path via `resolve_project_path()`; added `test_load_dataset_independent_of_cwd` and `test_bot_inference_does_not_load_training_csv`. |
| `tests/test_agents.py` | Updated direct artifact path checks to use `resolve_project_path()` so tests pass from any working directory. |
| `tests/test_meta_classifier.py` | Updated `test_saved_synthetic_dataset_and_three_way_training_load` to resolve via `resolve_project_path()` and generate into `tmp_path` if `/Data/processed/` is absent on a fresh clone; added `test_cross_version_sklearn_predict_proba_fallback`. |
| `tests/test_end_to_end_pipeline.py` | Updated `test_agent3_loads_synthetic_artifact` to use `resolve_project_path()` for subdirectory test runs. |
| `tests/test_streamlit_app.py` | Added `test_all_four_core_scenarios_and_no_csv_load_during_inference` and `test_unexpected_pipeline_exception_logs_traceback_and_shows_friendly_error`. |

---

## 4. Test Execution Results

All test suites were executed without skipping any tests:

| Test Suite | Passed | Failed | Errors | Skipped | Notes |
|---|---|---|---|---|---|
| `tests/test_models.py` | **13** | 0 | 0 | 0 | Includes all 5 previously failing dataset tests + CWD & CSV-free inference tests |
| `tests/test_agents.py` | **58** | 0 | 0 | 0 | Agent 1 & Agent 2 loading, preprocessing, batching, and schema contracts |
| `tests/test_meta_classifier.py` | **21** | 0 | 0 | 0 | Includes cross-version `scikit-learn` fallback & 5,000-row dataset tests |
| `tests/test_supervisor.py` | **39** | 0 | 0 | 0 | Rule-Based Supervisor baseline, policy matrix, and HRQ lifecycle |
| `tests/test_end_to_end_pipeline.py` | **14** | 0 | 0 | 0 | Live Agents 1+2+3 pipeline & strict input validation |
| `tests/test_streamlit_app.py` | **5** | 0 | 0 | 0 | All 5 pages, all 4 Analyze Post scenarios, HRQ resolve, and `logger.exception()` |
| `tests/test_preprocessing.py` | 0 | 0 | 0 | 0 | Empty placeholder file (`0` bytes) |
| **Total Active Tests (from root)** | **150** | **0** | **0** | **0** | **100% passing** |
| **Subdirectory CWD Run (`cd tests && pytest ...`)** | **145** | **0** | **0** | **0** | Verified independent of current working directory |

---

## 5. Analyze Post & Bot Workflow Verification

Verified end-to-end through `tests/test_streamlit_app.py` (`AppTest`) and `SocialThreatPipeline.run()` with `src.data.loader.load_bot_dataset` monkeypatched to fail if called:

| Scenario | Agent 1 Output | Agent 2 Output | Agent 3 Decision | Queued in HRQ | Result |
|---|---|---|---|---|---|
| **Human + Real News** | `real` | `human` | **`APPROVE`** | `False` | ✅ PASS |
| **Bot + Real News** | `real` | `bot` | **`REVIEW`** | `True` | ✅ PASS |
| **Human + Fake News** | `fake` | `human` | **`REVIEW`** | `True` | ✅ PASS |
| **Bot + Fake News** | `fake` | `bot` | **`RESTRICT`** | `True` | ✅ PASS |

- **No Training CSV Loaded During Inference:** Confirmed via automated monkeypatch assertion on `load_bot_dataset` during both `SocialThreatPipeline.run()` and Streamlit **Analyze Post** execution.
- **Exception Logging:** Confirmed via `caplog` test that unexpected internal exceptions log full tracebacks with `logger.exception()` while rendering `"An unexpected error occurred while running the analysis pipeline."` cleanly in the UI.
- **Human Review Queue:** Confirmed pending submission, reviewer resolution (`APPROVE` / `REJECT` / `RESTRICT`), and preservation of `ai_decision`.

---

## 6. Model Artifact Integrity Confirmation

No model was retrained, overwritten, or altered. All 6 saved model, vectorizer, scaler, and metadata files retain their exact SHA-256 checksums:

| Artifact | SHA-256 Checksum | Status |
|---|---|---|
| `models/saved/fake_news_model.joblib` | `03373CEE93980160BD01817E09546B044466614C6E6405FA2CA42489E4A9CA7C` | ✅ Unchanged |
| `models/vectorizers/tfidf_vectorizer.joblib` | `7FD6A1BE2E4355F3221ED368BE8E7F9D01364890CB6196D7C8EEBEE5A5E5FDDB` | ✅ Unchanged |
| `models/saved/bot_rnn_model.pth` | `050E69A32FD39A19030629C1EDF88FF7194358A376C1C2FB8E5A9AB3CD972F44` | ✅ Unchanged |
| `models/vectorizers/bot_scaler.joblib` | `89507EA4160CFEDD9BD468515F74FF16401584BC7E2BE7BD45E9EA69406FAB5D` | ✅ Unchanged |
| `models/vectorizers/bot_input_meta.joblib` | `288729F7786E397E3786F1DF8E8AD9055D5FF327E38A0D41D45D3E25955622E2` | ✅ Unchanged |
| `models/saved/meta_logreg_synthetic.joblib` | `B044F6836D8B25E33F8A934B8ED28971A6B5E394B048490680B1F40ABDBA476D` | ✅ Unchanged |

---

## 7. Cross-Machine Compatibility & Remaining Limitations

### Cross-Machine Readiness
- **No Personal Paths or CWD Assumptions:** Zero personal filesystem paths exist in code, config, or reports; all artifact and dataset paths resolve relative to `PROJECT_ROOT` via `pathlib.Path`.
- **Shared Files Inclusion:** All 6 model/vectorizer artifacts under `models/` and `twitter_human_bots_dataset-1.csv` (`11.3 MB`) are unignored in `.gitignore` and ready to be shared across computers.
- **Cross-Version Library Resilience:** In-memory `scikit-learn` (`1.6–1.9+`) and `NumPy` (`1.24–2.x`) deserialization shims plus default CPU PyTorch execution prevent environment-mismatch crashes on other machines.

### Remaining Limitations
1. **Synthetic Agent 3 Training Provenance:** `models/saved/meta_logreg_synthetic.joblib` remains a proof-of-concept artifact trained on synthetic policy data; real-world moderation accuracy requires future annotation of real cases.
2. **In-Memory Review Queue:** `ReviewQueue` stores cases in process memory per Streamlit session and does not persist to a database across restarts.
3. **Empty Baseline Placeholders:** `tests/test_preprocessing.py`, `reports/technical_report.md`, `notebooks/01_data_exploration.ipynb`, and `notebooks/02_data_preprocessing.ipynb` remain `0`-byte baseline files.
