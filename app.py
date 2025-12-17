import streamlit as st
import numpy as np
import pandas as pd
import base64

# ----------------------------
# Helpers / placeholders
# ----------------------------

TIERS = {
    1: "Normal",
    2: "Uncommon",
    3: "Rare",
    4: "Epic",
    5: "Legendary",
}
tier_name_to_value = {v: k for k, v in TIERS.items()}

QUALITY_IMPROVEMENTS = {
    1: 1,
    2: 1.3,
    3: 1.6,
    4: 1.9,
    5: 2.5
}

PROD_MODULE_PROD = {
    1: 0.04,
    2: 0.06,
    3: 0.1
}

QUALITY_MODULE_QUALITY = {
    1: 0.01,
    2: 0.02,
    3: 0.025
}

PROD_MODULES = ["Prod 1", "Prod 2", "Prod 3"]
QUAL_MODULES = ["Quality 1", "Quality 2", "Quality 3"]  # naming is UI-only; map however you like


def upgrade_probs_factorio(q: float, max_k: int) -> np.ndarray:
    """
    Factorio-style 'exactly k upgrades' probabilities for k=0..max_k, with tail folded
    into the max_k bucket.

    Rule:
      P(K=0) = 1 - q
      P(K=k) = 9 q / 10^k for k >= 1
    and we fold any probability mass for K > max_k into K=max_k.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0,1], got {q}")

    probs = np.zeros(max_k + 1, dtype=float)
    probs[0] = 1.0 - q
    for k in range(1, max_k + 1):
        probs[k] = 9.0 * q / (10.0 ** k)

    # fold tail beyond max_k
    s = probs.sum()
    if s < 1.0:
        probs[-1] += 1.0 - s
    return probs


def build_quality_matrix(q: float, n_tiers: int) -> np.ndarray:
    """
    Build a row-stochastic, upper-triangular quality transition matrix T (n x n) such that:
      T[i, j] = P(output quality j | input quality i)
    using Factorio-style upgrade probabilities with truncation at the top tier.

    The top tier (legendary) is absorbing: T[last, last] = 1.
    """
    if n_tiers < 2:
        raise ValueError("n_tiers must be >= 2")

    T = np.zeros((n_tiers, n_tiers), dtype=float)
    last = n_tiers - 1

    for i in range(n_tiers):
        if i == last:
            T[i, i] = 1.0
            continue

        max_k = last - i  # maximum possible upgrade steps before capping
        probs = upgrade_probs_factorio(q, max_k)

        for k, pk in enumerate(probs):
            T[i, i + k] += pk

    # sanity check rows sum to 1
    if not np.allclose(T.sum(axis=1), 1.0):
        raise RuntimeError("Quality matrix rows do not sum to 1; check implementation.")
    return T

def projection_nonlegendary(n_tiers: int) -> np.ndarray:
    """
    Projection Pi that zeroes out the legendary component (last tier).
    """
    Pi = np.eye(n_tiers, dtype=float)
    Pi[-1, -1] = 0.0
    return Pi

def steady_state_circulation(
    s_inflow: np.ndarray,
    A: np.ndarray,
    L: np.ndarray,
    p: float,
):
    s_inflow = np.asarray(s_inflow, dtype=float).reshape(-1)
    n = s_inflow.shape[0]
    if A.shape != (n, n) or R.shape != (n, n):
        raise ValueError("Dimension mismatch among s_inflow, A, R")

    I = np.eye(n, dtype=float)
    x_star = np.linalg.solve(I - L, L @ s_inflow)

    # recycled ingredients
    m_star = 0.25 * (R.T @ (x_star + s_inflow))
    c_star =p * (A.T @ m_star)

    return x_star, m_star, c_star


def parse_module_choice(label: str) -> tuple[int, int]:
    """
    label format: 'Prod 3 (Legendary)' or 'Quality 2 (Rare)'
    Returns: (module_level, quality_tier)
    """
    # module_level is 1/2/3; quality_tier is 1..5
    # This parser assumes the label is produced by the app itself.
    left, right = label.split(" (")
    tier_name = right[:-1]  # drop trailing ')'
    level = int(left.split()[-1])  # 'Prod 3' -> 3, 'Quality 2' -> 2
    tier = {v: k for k, v in TIERS.items()}[tier_name]
    return level, tier


def get_module_rates(best_prod_label, best_quality_label):
    prod_level, prod_tier = parse_module_choice(best_prod_label)
    qual_level, qual_tier = parse_module_choice(best_quality_label)

    base_prod_module = PROD_MODULE_PROD[prod_level]
    total_module_prod = QUALITY_IMPROVEMENTS[prod_tier] * base_prod_module

    base_quality_module =   QUALITY_MODULE_QUALITY[qual_level]
    total_module_quality = QUALITY_IMPROVEMENTS[qual_tier] * base_quality_module

    total_productivity = base_productivity  + total_module_prod * n_prod_modules_used
    q_a = total_module_quality * (n_modules_assembly - n_prod_modules_used)
    q_r = total_module_quality * 4

    return total_productivity, q_a, q_r

def compute_loop_rate(
    craft_rate: float,
    recycle_rate: float,
):
    seconds_per_loop = 1/recycle_rate + 1/craft_rate

    return seconds_per_loop

def display_pdf(path: str, height: int = 800):
    with open(path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="{height}"
        type="application/pdf">
    </iframe>
    """

    st.markdown(pdf_display, unsafe_allow_html=True)

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Factorio Quality Loop Calculator", layout="centered")
st.title("Factorio Quality Loop Calculator")

with st.sidebar:
    st.header("Inputs")

    craft_rate = st.number_input("Craft rate (items/sec)", min_value=0.0, value=10.0, step=0.1)
    recycle_rate = st.number_input("Recycle rate (items/sec)", min_value=0.0, value=10.0, step=0.1)

    n_modules_assembly = st.number_input("Number of module slots in assembly", min_value=0, value=4, step=1)
    n_prod_modules_used = st.number_input("Number of prod modules used", min_value=0, value=0, step=1)


    # Build the 15 options: level 1-3 × tier 1-5
    prod_options = [f"Prod {lvl} ({TIERS[tier]})" for lvl in (1, 2, 3) for tier in (1, 2, 3, 4, 5)]
    qual_options = [f"Quality {lvl} ({TIERS[tier]})" for lvl in (1, 2, 3) for tier in (1, 2, 3, 4, 5)]

    best_prod_label = st.selectbox("Best prod module available", prod_options, index=prod_options.index("Prod 3 (Legendary)"))
    best_quality_label = st.selectbox("Best quality module available", qual_options, index=qual_options.index("Quality 3 (Legendary)"))

    base_productivity = st.number_input("Base productivity of craft", min_value=0.0, value=0.0, step=0.01)

    max_quality_available_name = st.selectbox("Best quality tier available", TIERS.values(), index=4)
    items_added_per_sec = st.number_input("Items added to circulation per second (items/sec)", min_value=0.0, value=1.0, step=0.1)

    if n_prod_modules_used > n_modules_assembly:
        st.error("Error: Using more prod modules than available slots in assembly")
        st.stop()

# ----------------------------
# Internal Calculations
# ----------------------------
# Processing Inputs
num_qualitier_tiers = tier_name_to_value[max_quality_available_name]
seconds_per_loop = compute_loop_rate(craft_rate, recycle_rate)
loops_per_second = 1/seconds_per_loop
bonus_productivity, q_a, q_r = get_module_rates(best_prod_label, best_quality_label)
p = min(4, 1+bonus_productivity)

# Matrices
A = build_quality_matrix(q_a, num_qualitier_tiers)
R = build_quality_matrix(q_r, num_qualitier_tiers)
Pi = projection_nonlegendary(num_qualitier_tiers)
L = 0.25 * p * (Pi @ A.T @ R.T)

# Flow
items_added_per_loop = items_added_per_sec * seconds_per_loop
s = np.zeros(num_qualitier_tiers)
s[0] = items_added_per_loop

# st.subheader("Debug Values")
# st.write(
#     {
#         'A': A,
#         'R': R,
#         'L': L,
#         's': s,
#         'productivity': p
#     }
# )
# ----------------------------
# Compute Outputs
# ----------------------------
x_star, m_star, c_star = steady_state_circulation(s, A, L, p)
max_quality_per_loop = c_star[-1]
max_quality_per_second = max_quality_per_loop*loops_per_second
max_quality_per_item_added = max_quality_per_second / items_added_per_sec



# ----------------------------
# Display Outputs
# ----------------------------
tab_calc, tab_theory = st.tabs(["🔧 Calculator", "📄 Theory & Derivation"])


with tab_calc:
    st.markdown("**Steady-state item distribution**")
    tiers = list(TIERS.values())[: len(c_star)]
    df_c_star = pd.DataFrame(
        {
            "Quality tier": tiers,
            "Number of items in circulation": c_star,
        }
    )
    st.dataframe(
        df_c_star,
        hide_index=True,
       width="stretch",
    )


    col1, col2 = st.columns(2)

    col1.metric(
        "Max quality items produced per item added",
        f"{max_quality_per_item_added:.6g}",
    )

    col2.metric(
        "Max quality items produced per second",
        f"{max_quality_per_second:.6g}",
    )

with tab_theory:
    st.header("Theory and Derivation")

    with open("out/quality_analytics.pdf", "rb") as f:
        st.download_button(
            "Download full write-up (PDF)",
            f,
            file_name="quality_analytics.pdf",
        )

    display_pdf("out/quality_analytics.pdf")
