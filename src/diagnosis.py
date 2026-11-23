"""Frozen dual-error definitions and inclusive threshold rules."""
from __future__ import annotations
import numpy as np
import pandas as pd
MAIN = (.02, .05, .50)


def metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the user's exact area definitions with explicit numerical-zero treatment."""
    out = frame.copy()
    assert (out.observed_m2 > 0).all()
    assert np.allclose(out.fp_m2 + out.fn_m2, out.error_m2, atol=1e-7, rtol=0)
    assert np.allclose(out.pred_m2 - out.gt_m2, out.fp_m2 - out.fn_m2, atol=1e-7, rtol=0)
    out['E_net'] = abs(out.pred_m2 - out.gt_m2) / out.observed_m2
    out['E_spatial'] = (out.fp_m2 + out.fn_m2) / out.observed_m2
    out['Q'] = out.observed_m2 / out.scale_m.astype(float)**2
    wrong = out.fp_m2 + out.fn_m2
    out['C'] = np.where(wrong > 0, 1 - abs(out.fp_m2 - out.fn_m2) / wrong.where(wrong > 0), np.nan)
    out['hidden_excess'] = out.E_spatial - out.E_net
    assert np.allclose(out.hidden_excess, 2*np.minimum(out.fp_m2, out.fn_m2)/out.observed_m2, atol=1e-10)
    assert (out.E_net <= out.E_spatial + 1e-10).all()
    assert out.Q.between(0, 1+1e-6).all()
    return out


def classify(frame: pd.DataFrame, net: float, spatial: float, support: float) -> pd.DataFrame:
    """Keep task tolerances and support status separate, with inclusive pass boundaries."""
    out = frame.copy()
    out['net_pass'] = out.E_net <= net
    out['spatial_pass'] = out.E_spatial <= spatial
    out['support_low'] = out.Q < support
    out['hidden_spatial_error'] = out.net_pass & ~out.spatial_pass
    out['diagnosis'] = np.select(
        [out.net_pass & out.spatial_pass, out.hidden_spatial_error, ~out.net_pass & out.spatial_pass],
        ['low_both', 'net_low_spatial_high', 'net_high_spatial_low'], default='high_both')
    out['support_status'] = np.where(out.support_low, 'insufficient_observation', 'scenario_support_sufficient')
    out['joint_scenario_pass'] = out.net_pass & out.spatial_pass & ~out.support_low
    out['interpretation'] = 'POST_HOC_GT_BASED_NOT_UNLABELED_PREDICTION'
    return out


def summarize(frame: pd.DataFrame) -> dict:
    """Report conditional count fractions and unambiguous observed-area denominators."""
    n = len(frame)
    area = float(frame.observed_m2.sum())
    low, hidden = frame.net_pass, frame.hidden_spatial_error
    low_area = float(frame.loc[low, 'observed_m2'].sum())
    hidden_area = float(frame.loc[hidden, 'observed_m2'].sum())
    return dict(cells=n, observed_m2=area, baseline_net_low_cells=int(low.sum()),
        hidden_cells=int(hidden.sum()), hidden_fraction_of_net_low=float(hidden.sum()/low.sum()) if low.sum() else None,
        hidden_area_m2=hidden_area, hidden_area_fraction_total=hidden_area/area if area else None,
        hidden_area_fraction_net_low=hidden_area/low_area if low_area else None,
        net_low_area_fraction=low_area/area if area else None,
        low_support_cells=int(frame.support_low.sum()),
        low_support_area_fraction=float(frame.loc[frame.support_low,'observed_m2'].sum()/area) if area else None,
        hidden_low_support_cells=int((hidden & frame.support_low).sum()),
        joint_pass_cells=int(frame.joint_scenario_pass.sum()),
        weighted_E_net=float(frame.E_net @ frame.observed_m2/area) if area else None,
        weighted_E_spatial=float(frame.E_spatial @ frame.observed_m2/area) if area else None,
        C_defined_cells=int(frame.C.notna().sum()),
        mean_C=float(frame.C.mean()) if frame.C.notna().any() else None)
