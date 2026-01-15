# Validator Bugs Analysis Notes

**Date:** 2026-01-15
**Context:** Analyzing Logfire traces for failed LLM extractions

## Summary of Current Issues

After fixing the first batch of bugs, two new traces show persistent issues:

### Trace 1: `019bc2bfce496ae284e63f44093679cf`
**Final error:** `Unknown parameters in submodel.parameters: ['k_psc_const', 'k_psc_encounter']`

**Retry history errors (6 retries):**
1. Missing `biological_basis` field on `ObservableConstant` (repeats ALL retries)
2. Hardcoded constant `1.0 * ureg.milliliter` for V_T (compartment volume)
3. `setting an array element with a sequence` (distribution_code)
4. `median_obs must be Pint Quantity`
5. `Computed CI95[0] lower (4.633e+06) does not match reported (4e+06)` - 15% off, rejected by 1% tolerance

### Trace 2: `019bc2bfce5863f752fd2144bf73c0b3`
**Final error:** `setting an array element with a sequence`

**Retry history:** This error repeats ALL 5 retries. LLM cannot fix it despite improved error message.

---

## Root Causes Identified

### 1. Hardcoded Constant False Positive (V_T compartment volume)

**Problem:** The LLM writes `V_T = 1.0 * ureg.milliliter` in distribution_code, which gets flagged as a hardcoded constant.

**Analysis:**
- `distribution_code` is for STATISTICAL derivation (mean → samples → median/CI)
- It should NOT need model compartment volumes
- The LLM is confused about what distribution_code is for
- But even if it did need V_T, it should come from `inputs` or `constants`, not be hardcoded

**Options:**
1. Turn off `check_hardcoded=True` for distribution_code (pragmatic fix)
2. Improve prompt to clarify distribution_code purpose
3. Whitelist compartment volume names (V_T, V_C, V_P)

**Location:** `calibration_target_models.py:762` - `check_hardcoded=True`

### 2. `setting an array element with a sequence` - LLM Can't Self-Correct

**Problem:** Despite improved error message, LLM keeps producing bad ci95_obs structure across ALL retries.

**Analysis:**
- Error message is good but fix is too complex
- LLM pattern: `ci95_obs = np.array([[lo * units, hi * units]])` - numpy can't handle Pint in nested arrays
- Correct pattern: `ci95_obs = [[lo, hi]]` where both are Pint Quantities (not wrapped in np.array)

**Options:**
1. Add working code example directly in the error message
2. Pre-process ci95_obs in validator to handle common mistakes
3. Add more explicit guidance in prompt about ci95_obs format

### 3. Missing `biological_basis` Field

**Problem:** LLM keeps forgetting the required `biological_basis` field on `ObservableConstant`.

**Analysis:**
- This field is required but LLM doesn't include it
- Happens on EVERY retry - LLM isn't learning from feedback

**Options:**
1. Make `biological_basis` optional with a default
2. Better error message explaining what `biological_basis` should contain
3. Add example in prompt

### 4. 1% Tolerance Too Strict for CI Bounds

**Problem:** `4.633e+06` vs `4e+06` is ~15% off but biologically reasonable.

**Analysis:**
- CI bounds from Monte Carlo sampling have inherent variance
- 1% tolerance is appropriate for median but too strict for CI bounds
- Literature often rounds CI values (4e+06 vs 4.633e+06)

**Options:**
1. Use looser tolerance for CI bounds (e.g., 10%)
2. Allow rounding to significant figures in comparison
3. Only warn (not error) for CI mismatches

---

## Proposed Fixes (Priority Order)

### High Priority
1. **Turn off hardcoded constant check for distribution_code** - False positives causing failures
2. **Relax CI95 tolerance to 10%** - Too strict for Monte Carlo bounds
3. **Make biological_basis optional** - Or provide better guidance

### Medium Priority
4. **Add working code example to array structure error** - Help LLM self-correct
5. **Improve prompt guidance for distribution_code** - Clarify it's statistical only

### Low Priority
6. **Whitelist compartment volumes** - If hardcoded check stays on

---

## Code Locations

- Hardcoded constant check: `calibration_target_models.py:762`
- CI95 tolerance: `calibration_target_models.py:843-871` (rel_tol = 0.01)
- biological_basis field: `observable.py` (ObservableConstant class)
- Array error message: `calibration_target_models.py:881-892`
