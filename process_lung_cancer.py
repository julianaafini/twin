"""
Detailed Data Analysis for Lung Cancer Trial (NCT01439568)
IMPROVED DATA PROCESSING FOR CLINICAL TRIAL DIGITAL TWINS
Optimized for small datasets and simplified diffusion model
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyreadstat
import re

# ==========================================================
# 0. Custom date parser + scalar (NO pandas datetime64)
# ==========================================================


def parse_sdtm_datetime(x):
    """Parse SDTM-like datetime strings. Returns a Python datetime or None."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None

    # Full datetime with time
    if "T" in s:
        m_dt = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})", s)
        if not m_dt:
            return None
        year, month, day, hour, minute = map(int, m_dt.groups())
        try:
            return datetime(year, month, day, hour, minute)
        except ValueError:
            return None

    # Date / partial date (no time component)
    s = s.replace("--", "-")

    year = month = day = None

    m_full = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    m_ym = re.fullmatch(r"(\d{4})-(\d{2})-?", s)
    m_y = re.fullmatch(r"(\d{4})-?", s)

    if m_full:
        year, month, day = map(int, m_full.groups())
    elif m_ym:
        year, month = map(int, m_ym.groups())
        day = 1
    elif m_y:
        year = int(m_y.group(1))
        month, day = 1, 1
    else:
        return None

    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def datetime_to_scalar(d):
    """Map a Python datetime to a numeric 'time' that is strictly increasing."""
    if d is None:
        return np.nan
    minutes = d.hour * 60 + d.minute
    return d.toordinal() * 1440 + minutes


# ==========================================================
# 1. IDENTIFY DIGITAL TWIN COHORT
# ==========================================================


def identify_digital_twin_cohort(dm, ds):
    """
    Identify the cohort for digital twin generation (DM - screen failures).
    """
    all_dm_ids = set(dm["USUBJID"])

    if "DSDECOD" in ds.columns:
        screen_mask = ds["DSDECOD"].astype(str).str.upper().str.contains("SCREEN", na=False)
        screen_fail_ids = set(ds.loc[screen_mask, "USUBJID"])
        print(f"[Digital Twin Cohort] Identified {len(screen_fail_ids)} screen failures")
    else:
        screen_fail_ids = set()
        print("[Digital Twin Cohort] WARNING: No DSDECOD column, keeping all DM patients")

    digital_twin_ids = sorted(all_dm_ids - screen_fail_ids)

    print(f"[Digital Twin Cohort] Total DM patients: {len(all_dm_ids)}")
    print(f"[Digital Twin Cohort] Final cohort: {len(digital_twin_ids)} patients")

    return digital_twin_ids


def restrict_to_cohort(df, cohort_ids, domain_name):
    """Restrict a dataframe to the digital twin cohort."""
    if "USUBJID" not in df.columns:
        print(f"[{domain_name}] No USUBJID column; leaving as-is.")
        return df

    out = df[df["USUBJID"].isin(cohort_ids)].copy()
    print(f"[{domain_name}] Filtered to {out['USUBJID'].nunique()} patients, {len(out)} rows")
    return out


# ==========================================================
# 2. BUILD VISIT TIMELINE (VISITNUM_MODEL alignment)
# ==========================================================


def build_visit_timeline(ex_df, vs_df, mh_df, cm_df):
    """
    Build unified visit timeline with VISITNUM_MODEL.
    """

    def process_domain_visits(df, domain_name, date_col):
        """Extract visit timeline from a domain."""
        if df.empty or "USUBJID" not in df.columns or "VISITNUM" not in df.columns:
            print(f"[{domain_name}] Cannot contribute to visit timeline")
            return pd.DataFrame(columns=["USUBJID", "VISITNUM_MODEL", "_TIME", "VISIT"])

        tmp = df.copy()
        tmp = tmp.rename(columns={"VISITNUM": "VISITNUM_MODEL"})

        # Parse dates if available
        if date_col and date_col in df.columns:
            tmp["_parsed_dt"] = tmp[date_col].map(parse_sdtm_datetime)
            tmp["_TIME"] = tmp["_parsed_dt"].map(datetime_to_scalar)

            if tmp["_TIME"].notna().any():
                print(f"[{domain_name}] Using {date_col} for timeline")
            else:
                tmp = tmp.sort_values(["USUBJID", "VISITNUM_MODEL"])
                tmp["_TIME"] = tmp.groupby("USUBJID").cumcount().astype("int64")
                print(f"[{domain_name}] Using VISITNUM order for timeline")
        else:
            tmp = tmp.sort_values(["USUBJID", "VISITNUM_MODEL"])
            tmp["_TIME"] = tmp.groupby("USUBJID").cumcount().astype("int64")
            print(f"[{domain_name}] Using VISITNUM order (no date column)")

        visit_col = "VISIT" if "VISIT" in tmp.columns else None

        keep_cols = ["USUBJID", "VISITNUM_MODEL", "_TIME"]
        if visit_col:
            keep_cols.append("VISIT")

        result = tmp[keep_cols].drop_duplicates(subset=["USUBJID", "VISITNUM_MODEL"])
        return result

    visit_parts = []

    if not ex_df.empty:
        ex_visits = process_domain_visits(ex_df, "EX", "EXSTDTC")
        visit_parts.append(ex_visits)

    if not vs_df.empty:
        vs_visits = process_domain_visits(vs_df, "VS", "VSDTC")
        visit_parts.append(vs_visits)

    if not mh_df.empty:
        mh_date = "MHDTC" if "MHDTC" in mh_df.columns else "MHSTDTC"
        mh_visits = process_domain_visits(mh_df, "MH", mh_date)
        visit_parts.append(mh_visits)

    if not cm_df.empty:
        cm_date = "CMSTDTC" if "CMSTDTC" in cm_df.columns else None
        cm_visits = process_domain_visits(cm_df, "CM", cm_date)
        visit_parts.append(cm_visits)

    if not visit_parts:
        raise ValueError("No domains available to build visit timeline!")

    visit_ref = pd.concat(visit_parts, ignore_index=True)

    visit_ref = (
        visit_ref
        .sort_values(["USUBJID", "_TIME", "VISITNUM_MODEL"])
        .drop_duplicates(subset=["USUBJID", "VISITNUM_MODEL"], keep="first")
    )

    print(f"\n[Visit Timeline] Built timeline for {visit_ref['USUBJID'].nunique()} patients")
    print(f"[Visit Timeline] Total unique visits: {len(visit_ref)}")
    print(f"[Visit Timeline] Avg visits per patient: {len(visit_ref) / visit_ref['USUBJID'].nunique():.1f}")

    return visit_ref


# ==========================================================
# 3. EXTRACT KEY FEATURES FROM EACH DOMAIN
# ==========================================================


def extract_baseline_features(dm_df, ds_df, mh_df, ph_df, digital_twin_ids):
    """Extract baseline (time-invariant) features."""
    baseline = dm_df[dm_df["USUBJID"].isin(digital_twin_ids)].copy()

    keep_cols = ["USUBJID"]
    for col in [
        "SITEID",
        "SITENO",
        "COUNTRY",
        "AGE",
        "SEX",
        "RACE",
        "ETHNIC",
        "ARM",
        "ARMCD",
        "RANDFL",
        "DTHFL",
        "DTHDTC",
        "RFSTDTC",
    ]:
        if col in baseline.columns:
            keep_cols.append(col)

    baseline = baseline[keep_cols].copy()

    # Medical history: PRIOR_CONDITIONS = list of unique conditions per patient (no dates)
    if not mh_df.empty and "USUBJID" in mh_df.columns:
        mh_term_col = None
        for col in ["MHTERM", "MEDXDESC", "MHDECOD", "MHBODSYS"]:
            if col in mh_df.columns:
                mh_term_col = col
                break

        if mh_term_col:
            mh_clean = mh_df[["USUBJID", mh_term_col]].dropna(subset=[mh_term_col]).copy()
            mh_clean[mh_term_col] = mh_clean[mh_term_col].astype(str)

            mh_summary = (
                mh_clean.groupby("USUBJID")[mh_term_col].apply(lambda x: sorted(set(x))).to_frame(name="PRIOR_CONDITIONS")
            )

            baseline = baseline.merge(
                mh_summary,
                left_on="USUBJID",
                right_index=True,
                how="left",
            )

            baseline["PRIOR_CONDITIONS"] = baseline["PRIOR_CONDITIONS"].apply(
                lambda x: x if isinstance(x, list) else []
            )

            print(f"[Baseline] Using {mh_term_col} to build list PRIOR_CONDITIONS")
        else:
            baseline["PRIOR_CONDITIONS"] = [[] for _ in range(len(baseline))]
            print("[Baseline] Missing medical history term column, PRIOR_CONDITIONS set to empty lists")
    else:
        baseline["PRIOR_CONDITIONS"] = [[] for _ in range(len(baseline))]
        print("[Baseline] No MH domain, PRIOR_CONDITIONS set to empty lists")

    # Histology: rename HIST -> HISTOPATHOLOGY
    if not ph_df.empty and "HIST" in ph_df.columns:
        ph_hist = ph_df.groupby("USUBJID")["HIST"].first().reset_index()
        ph_hist = ph_hist.rename(columns={"HIST": "HISTOPATHOLOGY"})
        baseline = baseline.merge(ph_hist, on="USUBJID", how="left")

    # Final disposition
    if not ds_df.empty and "DSDECOD" in ds_df.columns:
        ds_sorted = ds_df.sort_values(["USUBJID"])
        last_ds = ds_sorted.groupby("USUBJID").tail(1)
        baseline = baseline.merge(
            last_ds[["USUBJID", "DSDECOD"]].rename(columns={"DSDECOD": "FINAL_DISPOSITION"}),
            on="USUBJID",
            how="left",
        )

    print(f"\n[Baseline Features] Extracted {len(baseline.columns)} baseline features")
    print(f"[Baseline Features] {len(baseline)} patients")

    return baseline


def map_domain_to_visits_by_date(df, domain_name, date_col, visit_ref):
    """Map domain records to VISITNUM_MODEL using nearest date matching."""
    if df.empty or "USUBJID" not in df.columns:
        return df

    df = df.copy()

    if "VISITNUM" in df.columns:
        df = df.rename(columns={"VISITNUM": "VISITNUM_MODEL"})
        print(f"[{domain_name}] Using existing VISITNUM as VISITNUM_MODEL")
        return df

    if date_col not in df.columns:
        print(f"[{domain_name}] No {date_col} column; cannot map to visits")
        df["VISITNUM_MODEL"] = np.nan
        return df

    df["_parsed_dt"] = df[date_col].map(parse_sdtm_datetime)
    df = df[df["_parsed_dt"].notna()].copy()

    if df.empty:
        print(f"[{domain_name}] No valid dates; cannot map to visits")
        df["VISITNUM_MODEL"] = np.nan
        return df

    df["_TIME_DOMAIN"] = df["_parsed_dt"].map(datetime_to_scalar).astype("int64")

    ref_grouped = {u: sub for u, sub in visit_ref.groupby("USUBJID")}

    mapped_parts = []
    for usubjid, sub in df.groupby("USUBJID", sort=False):
        vref = ref_grouped.get(usubjid)
        if vref is None or vref.empty:
            sub = sub.assign(VISITNUM_MODEL=np.nan)
            mapped_parts.append(sub)
            continue

        vref_clean = vref.dropna(subset=["_TIME"])
        if vref_clean.empty:
            sub = sub.assign(VISITNUM_MODEL=np.nan)
            mapped_parts.append(sub)
            continue

        vt = vref_clean["_TIME"].to_numpy(dtype="int64")
        vvis = vref_clean["VISITNUM_MODEL"].to_numpy()
        tvals = sub["_TIME_DOMAIN"].to_numpy(dtype="int64")

        idx = np.abs(tvals[:, None] - vt[None, :]).argmin(axis=1)
        sub = sub.assign(VISITNUM_MODEL=vvis[idx])
        mapped_parts.append(sub)

    result = pd.concat(mapped_parts, ignore_index=True)
    print(f"[{domain_name}] Mapped to VISITNUM_MODEL using {date_col}")

    return result


def extract_visit_features_efficient(
    visit_ref, ex_df, vs_df, lbcen_df, ae_df, cm_df, rs_df, pe_df
):
    """Extract visit-level features efficiently (one row per patient per visit)."""
    visits = visit_ref.copy()

    # TIME FEATURES
    first_time = visit_ref.groupby("USUBJID")["_TIME"].min().to_dict()
    visits["_FIRST_TIME"] = visits["USUBJID"].map(first_time)
    visits["TIME_FROM_FIRST_DOSE_DAYS"] = (visits["_TIME"] - visits["_FIRST_TIME"]) / 1440.0

    visits = visits.sort_values(["USUBJID", "_TIME"])
    visits["VISIT_INDEX"] = visits.groupby("USUBJID").cumcount()
    visits["CYCLE_INDEX"] = np.floor(visits["TIME_FROM_FIRST_DOSE_DAYS"] / 21.0).astype("Int64")

    # EXPOSURE
    if not ex_df.empty:
        ex_mapped = ex_df.rename(columns={"VISITNUM": "VISITNUM_MODEL"})

        trt_col = None
        for col in ["EXTRT", "EXDECOD", "EXTRTDCD"]:
            if col in ex_mapped.columns:
                trt_col = col
                break

        agg_dict = {}
        if trt_col:
            agg_dict[trt_col] = "first"
        if "EXSTDTC" in ex_mapped.columns:
            agg_dict["EXSTDTC"] = "first"
        if "EXENDTC" in ex_mapped.columns:
            agg_dict["EXENDTC"] = "last"

        if agg_dict:
            ex_agg = ex_mapped.groupby(["USUBJID", "VISITNUM_MODEL"]).agg(agg_dict).reset_index()

            if trt_col and trt_col != "EXTRT":
                ex_agg = ex_agg.rename(columns={trt_col: "EXTRT"})

            visits = visits.merge(ex_agg, on=["USUBJID", "VISITNUM_MODEL"], how="left")

    # ADVERSE EVENTS (SUMMARY)
    if not ae_df.empty:
        ae_mapped = map_domain_to_visits_by_date(ae_df, "AE", "AESTDTC", visit_ref)

        ae_term_col = None
        for col in ["AEDECOD", "AETERM", "AELLT"]:
            if col in ae_mapped.columns:
                ae_term_col = col
                break

        if ae_term_col:
            ae_mapped[ae_term_col] = ae_mapped[ae_term_col].astype(str)

            ae_summary = (
                ae_mapped.groupby(["USUBJID", "VISITNUM_MODEL"]).agg(
                    NUM_AES=(ae_term_col, "count"),
                    ALL_AES=(ae_term_col, lambda x: "; ".join(sorted(set(x.dropna())))),
                )
            ).reset_index()

            if "AETOXGR" in ae_mapped.columns:
                ae_grade = (
                    ae_mapped.groupby(["USUBJID", "VISITNUM_MODEL"]).agg(
                        MAX_GRADE=("AETOXGR", lambda x: x.max() if x.notna().any() else np.nan)
                    )
                ).reset_index()
                ae_summary = ae_summary.merge(
                    ae_grade, on=["USUBJID", "VISITNUM_MODEL"], how="left"
                )

            if "AESER" in ae_mapped.columns:
                ae_ser = (
                    ae_mapped.groupby(["USUBJID", "VISITNUM_MODEL"])["AESER"]
                    .apply(lambda x: 1 if (x == "Y").any() else 0)
                    .reset_index()
                    .rename(columns={"AESER": "HAS_SERIOUS_AE"})
                )
                ae_summary = ae_summary.merge(
                    ae_ser, on=["USUBJID", "VISITNUM_MODEL"], how="left"
                )

            visits = visits.merge(
                ae_summary, on=["USUBJID", "VISITNUM_MODEL"], how="left"
            )

            if "NUM_AES" in visits.columns:
                visits["NUM_AES"] = visits["NUM_AES"].fillna(0)

    # CONMEDS (SUMMARY)
    if not cm_df.empty:
        cm_mapped = cm_df.rename(columns={"VISITNUM": "VISITNUM_MODEL"})

        cm_term_col = None
        for col in ["CMDECOD", "CMTRT", "CMTERM"]:
            if col in cm_mapped.columns:
                cm_term_col = col
                break

        if cm_term_col:
            cm_mapped[cm_term_col] = cm_mapped[cm_term_col].astype(str)

            cm_summary = (
                cm_mapped.groupby(["USUBJID", "VISITNUM_MODEL"]).agg(
                    NUM_CONMEDS=(cm_term_col, "count"),
                    ALL_CONMEDS=(cm_term_col, lambda x: "; ".join(sorted(set(x.dropna())))),
                )
            ).reset_index()

            visits = visits.merge(
                cm_summary, on=["USUBJID", "VISITNUM_MODEL"], how="left"
            )

            visits["NUM_CONMEDS"] = visits["NUM_CONMEDS"].fillna(0)

    # RESPONSE
    if not rs_df.empty:
        rs_mapped = rs_df.rename(columns={"VISITNUM": "VISITNUM_MODEL"})

        keep_cols = ["USUBJID", "VISITNUM_MODEL"]

        for col in [
            "RSSTRESC",
            "RSORRES",
            "RSTESTCD",
            "RSCAT",
            "RSEVAL",
            "RSDTC",
            "RSDY",
            "BESTRESP",
        ]:
            if col in rs_mapped.columns:
                keep_cols.append(col)

        rs_clean = rs_mapped[keep_cols].copy()

        rs_pivot = rs_clean.pivot_table(
            index=["USUBJID", "VISITNUM_MODEL"],
            columns="RSTESTCD",
            values="RSSTRESC",
            aggfunc="first",
        ).reset_index()

        if "RSDTC" in rs_clean.columns:
            dt = rs_clean.groupby(["USUBJID", "VISITNUM_MODEL"])["RSDTC"].first().reset_index()
            rs_pivot = rs_pivot.merge(dt, on=["USUBJID", "VISITNUM_MODEL"], how="left")

        visits = visits.merge(rs_pivot, on=["USUBJID", "VISITNUM_MODEL"], how="left")

    # PHYSICAL EXAM
    if not pe_df.empty and "PEABNL" in pe_df.columns:
        pe_mapped = pe_df.rename(columns={"VISITNUM": "VISITNUM_MODEL"})

        pe_mapped["PEABNL_NUM"] = pd.to_numeric(pe_mapped["PEABNL"], errors="coerce")

        pe_mapped["PESPY_ABN"] = np.where(
            pe_mapped["PEABNL_NUM"] == 1,
            pe_mapped.get("PESPY", np.nan),
            np.nan,
        )

        pe_summary = (
            pe_mapped.groupby(["USUBJID", "VISITNUM_MODEL"]).agg(
                HAS_PE_ABNORMALITY=("PEABNL_NUM", lambda x: "Y" if (x == 1).any() else "N"),
                PE_ABNORMALITY_DESC=("PESPY_ABN", lambda s: "; ".join(s.dropna().astype(str).unique()) if s.notna().any() else ""),
            )
        ).reset_index()

        visits = visits.merge(pe_summary, on=["USUBJID", "VISITNUM_MODEL"], how="left")

    # VITAL SIGNS
    if not vs_df.empty and "VSTESTCD" in vs_df.columns:
        vs_mapped = vs_df.rename(columns={"VISITNUM": "VISITNUM_MODEL"})

        key_vs = ["TEMP", "ECOG", "HGT", "DIABP", "PRT", "SYSBP", "WGT"]

        vs_res_col = None
        for col in ["VSORRES", "VSSTRESC", "VSSTRESN"]:
            if col in vs_mapped.columns:
                vs_res_col = col
                break

        if vs_res_col:
            vs_pivot_data = []
            for vs_code in key_vs:
                vs_subset = vs_mapped[vs_mapped["VSTESTCD"] == vs_code]
                if not vs_subset.empty:
                    vs_agg = (
                        vs_subset.groupby(["USUBJID", "VISITNUM_MODEL"])[vs_res_col]
                        .first()
                        .reset_index()
                    )
                    vs_agg = vs_agg.rename(columns={vs_res_col: f"VS_{vs_code}"})
                    vs_pivot_data.append(vs_agg)

            for vs_data in vs_pivot_data:
                visits = visits.merge(vs_data, on=["USUBJID", "VISITNUM_MODEL"], how="left")

    visits = visits.drop(columns=["_TIME", "_FIRST_TIME"], errors="ignore")

    print(f"\n[Visit Features] Extracted {len(visits.columns)} visit-level features")
    print(f"[Visit Features] {len(visits)} total visit records")
    print(f"[Visit Features] {visits['USUBJID'].nunique()} unique patients")

    return visits


# ==========================================================
# 4. EXTRACT OUTCOMES
# ==========================================================


def extract_outcomes(dm_df, pfs_df, os_df, dor_df, rs_df, digital_twin_ids):
    """Extract outcome variables (PFS, OS, DOR, Best Response)."""
    outcomes = dm_df[dm_df["USUBJID"].isin(digital_twin_ids)][["USUBJID"]].copy()

    if not pfs_df.empty:
        pfs_cols = [c for c in pfs_df.columns if c.upper() in ["USUBJID", "PFS_DUR", "PFS_EVENT", "PFS_STATUS", "PFS_CENS"]]
        if pfs_cols:
            outcomes = outcomes.merge(pfs_df[pfs_cols], on="USUBJID", how="left")

    if not os_df.empty:
        os_cols = [c for c in os_df.columns if c.upper() in ["USUBJID", "OS_DUR", "OS_EVENT", "OS_STATUS", "OS_CENS"]]
        if os_cols:
            outcomes = outcomes.merge(os_df[os_cols], on="USUBJID", how="left")

    if not dor_df.empty:
        dor_cols = [c for c in dor_df.columns if c.upper() in ["USUBJID", "DOR", "DOR_DUR", "DOR_EVENT", "DOR_STATUS"]]
        if dor_cols:
            outcomes = outcomes.merge(dor_df[dor_cols], on="USUBJID", how="left")

    if not rs_df.empty and "RSSTRESC" in rs_df.columns:
        def rs_score(cat):
            if pd.isna(cat):
                return -1
            c = str(cat).upper().strip()
            if "CR" in c and "NON" not in c:
                return 5
            if "PR" in c:
                return 4
            if "NON-CR/NON-PD" in c or "NON CR/NON PD" in c:
                return 3
            if "SD" in c:
                return 2
            if "NE" in c or "NOT EVALUABLE" in c:
                return 1
            if "PD" in c:
                return 0
            return -1

        rs_tmp = rs_df[["USUBJID", "RSSTRESC"]].copy()
        rs_tmp["RS_SCORE"] = rs_tmp["RSSTRESC"].map(rs_score)

        best_rs = (
            rs_tmp.sort_values(["USUBJID", "RS_SCORE"], ascending=[True, False])
            .drop_duplicates(subset=["USUBJID"], keep="first")
            .rename(columns={"RSSTRESC": "BEST_RS"})
        )

        outcomes = outcomes.merge(best_rs[["USUBJID", "BEST_RS"]], on="USUBJID", how="left")

    print(f"\n[Outcomes] Extracted outcome data for {len(outcomes)} patients")
    print(f"[Outcomes] Features: {list(outcomes.columns)}")

    return outcomes


# ==========================================================
# 5. MAIN PROCESSING FUNCTION
# ==========================================================


def process_clinical_trial_data(dm, ae, cm, ex, mh, ds, rs, pfs, dor, vs, lbcen, pe, ph, osf, verbose=True):
    """
    Main function to process all clinical trial data into baseline, visits, and outcomes.
    """

    if verbose:
        print("\n" + "=" * 80)
        print("CLINICAL TRIAL DATA PROCESSING - EFFICIENT VERSION")
        print("=" * 80 + "\n")

    digital_twin_ids = identify_digital_twin_cohort(dm, ds)

    dm_f = restrict_to_cohort(dm, digital_twin_ids, "DM")
    ae_f = restrict_to_cohort(ae, digital_twin_ids, "AE")
    cm_f = restrict_to_cohort(cm, digital_twin_ids, "CM")
    ex_f = restrict_to_cohort(ex, digital_twin_ids, "EX")
    mh_f = restrict_to_cohort(mh, digital_twin_ids, "MH")
    ds_f = restrict_to_cohort(ds, digital_twin_ids, "DS")
    rs_f = restrict_to_cohort(rs, digital_twin_ids, "RS")
    pfs_f = restrict_to_cohort(pfs, digital_twin_ids, "PFS")
    dor_f = restrict_to_cohort(dor, digital_twin_ids, "DOR")
    vs_f = restrict_to_cohort(vs, digital_twin_ids, "VS")
    lbcen_f = restrict_to_cohort(lbcen, digital_twin_ids, "LBCEN")
    pe_f = restrict_to_cohort(pe, digital_twin_ids, "PE")
    ph_f = restrict_to_cohort(ph, digital_twin_ids, "PH")
    os_f = restrict_to_cohort(osf, digital_twin_ids, "OS")

    visit_ref = build_visit_timeline(ex_f, vs_f, mh_f, cm_f)

    baseline = extract_baseline_features(dm_f, ds_f, mh_f, ph_f, digital_twin_ids)

    visits = extract_visit_features_efficient(
        visit_ref, ex_f, vs_f, lbcen_f, ae_f, cm_f, rs_f, pe_f
    )

    outcomes = extract_outcomes(dm_f, pfs_f, os_f, dor_f, rs_f, digital_twin_ids)

    if verbose:
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"\n✓ Baseline: {baseline.shape}")
        print(f"  Columns: {list(baseline.columns)}")
        print(f"\n✓ Visits: {visits.shape}")
        print(f"  Columns: {list(visits.columns)}")
        print(f"\n✓ Outcomes: {outcomes.shape}")
        print(f"  Columns: {list(outcomes.columns)}")

        print("\nData Quality:")
        print(f"  Patients in baseline: {baseline['USUBJID'].nunique()}")
        print(f"  Patients in visits: {visits['USUBJID'].nunique()}")
        print(f"  Avg visits per patient: {len(visits) / visits['USUBJID'].nunique():.1f}")
        print(f"  Visit features: {len([c for c in visits.columns if c.startswith(('VS_', 'LAB_', 'NUM_', 'MAX_', 'HAS_'))])}")

        print("\n" + "=" * 80 + "\n")

    return baseline, visits, outcomes


# ==========================================================
# 6. DATA LOADING + EXAMPLE USAGE
# ==========================================================


def load_domain(file_name: str, data_dir: Path) -> pd.DataFrame:
    """Helper to read a SAS dataset if it exists."""
    path = data_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Expected domain file not found: {path}")
    df, _ = pyreadstat.read_sas7bdat(str(path))
    return df


def locate_data_dir() -> Path:
    """Locate the lung cancer dataset directory, handling local paths."""
    candidates = [
        Path("/data/lung_cancer"),
        Path(__file__).resolve().parent / "data " / "lung_cancer",
        Path(__file__).resolve().parent / "data" / "lung_cancer",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate lung cancer data directory")


def load_all_domains(data_dir: Path) -> Tuple[pd.DataFrame, ...]:
    """Load all SDTM domains needed for processing."""
    dm = load_domain("dm.sas7bdat", data_dir)
    ae = load_domain("ae.sas7bdat", data_dir)
    cm = load_domain("cm.sas7bdat", data_dir)
    ex = load_domain("ex.sas7bdat", data_dir)
    mh = load_domain("mh.sas7bdat", data_dir)
    ds = load_domain("ds_t.sas7bdat", data_dir)
    rs = load_domain("rs.sas7bdat", data_dir)
    pfs = load_domain("pfs.sas7bdat", data_dir)
    dor = load_domain("dor.sas7bdat", data_dir)
    tr = load_domain("tr.sas7bdat", data_dir)
    vs = load_domain("vs.sas7bdat", data_dir)
    lbcen = load_domain("lbcen.sas7bdat", data_dir) if (data_dir / "lbcen.sas7bdat").exists() else pd.DataFrame()
    pe = load_domain("pe.sas7bdat", data_dir)
    ph = load_domain("ph.sas7bdat", data_dir)
    osf = load_domain("os.sas7bdat", data_dir)
    return dm, ae, cm, ex, mh, ds, rs, pfs, dor, tr, vs, lbcen, pe, ph, osf


if __name__ == "__main__":
    DATA_DIR = locate_data_dir()
    (
        dm,
        ae,
        cm,
        ex,
        mh,
        ds,
        rs,
        pfs,
        dor,
        tr,
        vs,
        lbcen,
        pe,
        ph,
        osf,
    ) = load_all_domains(DATA_DIR)

    baseline, visits, outcomes = process_clinical_trial_data(
        dm,
        ae,
        cm,
        ex,
        mh,
        ds,
        rs,
        pfs,
        dor,
        vs,
        lbcen,
        pe,
        ph,
        osf,
    )

    # Placeholders to prevent unused variable warnings
    _ = (baseline, visits, outcomes)
