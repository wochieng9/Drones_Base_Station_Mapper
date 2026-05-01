import streamlit as st
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from itertools import combinations
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.graph_objects as go
import io
from distance_export import create_distance_spreadsheet, create_distance_spreadsheet_parallel


# ── Data loading ──────────────────────────────────────────────────────────────

HOSPITAL_KEYWORDS = [
    "hospital", "hôpital", "hopital", "hospitais", "hospitalier",
    "hospitaliero", "hospitalier universitaire", "hospitalier régional",
    "hospital medical center", "hospital medical centre",
    "federal medical centre", "university teaching hospital",
]

@st.cache_data
def load_facilities(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["latitude", "longitude", "country", "facility_n"])
    facility_lower = df["facility_n"].str.lower().str.strip()
    df["is_hospital"] = facility_lower.apply(
        lambda f: any(kw in f for kw in HOSPITAL_KEYWORDS)
    )
    return df


# ── Optimizer ─────────────────────────────────────────────────────────────────

class DroneHubOptimizer:
    """
    P-median optimization to select which health facilities should host
    drone base stations, maximising coverage of all facilities in the network.
    """

    def __init__(self, facilities, operational_radius=80, facility_names=None,
                 candidate_coords=None, candidate_names=None):
        self.facilities       = np.array(facilities)
        self.operational_radius = operational_radius
        self.facility_names   = facility_names or [f"Facility {i}" for i in range(len(facilities))]
        self.candidates       = np.array(candidate_coords) if candidate_coords else self.facilities
        self.candidate_names  = candidate_names or self.facility_names

        # Rectangular: candidates × all facilities
        self.distance_matrix  = self._calculate_distances()
        self.coverage_matrix  = (self.distance_matrix <= operational_radius).astype(int)

    def _calculate_distances(self):
        return cdist(self.candidates, self.facilities,
                     metric=lambda u, v: self._haversine(u, v))

    def _haversine(self, coord1, coord2):
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(a))

    def update_radius(self, new_radius):
        self.operational_radius = new_radius
        self.coverage_matrix = (self.distance_matrix <= new_radius).astype(int)

    def optimize(self, p, method="greedy"):
        if method == "greedy":
            return self._greedy(p)
        return self._exact(p)

    def _greedy(self, p):
        selected, covered, incremental = [], set(), {}
        for _ in range(min(p, len(self.candidates))):
            best_hub, best_new = None, 0
            for i in range(len(self.candidates)):
                if i in selected:
                    continue
                newly = set(np.where(self.coverage_matrix[i] == 1)[0]) - covered
                if len(newly) > best_new:
                    best_new, best_hub = len(newly), i
            if best_hub is not None:
                newly = set(np.where(self.coverage_matrix[best_hub] == 1)[0]) - covered
                incremental[best_hub] = len(newly)
                selected.append(best_hub)
                covered.update(newly)
        return self._format(selected, covered, incremental)

    def _exact(self, p):
        best_cov, best_combo = 0, None
        for combo in combinations(range(len(self.candidates)), p):
            covered = set()
            for idx in combo:
                covered.update(np.where(self.coverage_matrix[idx] == 1)[0])
            if len(covered) > best_cov:
                best_cov, best_combo = len(covered), combo
        covered, incremental = set(), {}
        for idx in sorted(best_combo,
                          key=lambda i: len(np.where(self.coverage_matrix[i] == 1)[0]),
                          reverse=True):
            newly = set(np.where(self.coverage_matrix[idx] == 1)[0]) - covered
            incremental[idx] = len(newly)
            covered.update(newly)
        return self._format(list(best_combo), covered, incremental)

    def _format(self, selected_hubs, covered_facilities, incremental_coverage):
        return {
            "selected_hubs":        selected_hubs,
            "covered_facilities":   list(covered_facilities),
            "coverage_count":       len(covered_facilities),
            "coverage_rate":        len(covered_facilities) / len(self.facilities),
            "incremental_coverage": incremental_coverage,
        }

    def coverage_curve(self, max_p, method="greedy"):
        """Return coverage rates for 1..max_p hubs."""
        rates = []
        for p in range(1, max_p + 1):
            result = self.optimize(p, method=method)
            rates.append(result["coverage_rate"])
        return rates

    def create_folium_map(self, p, method="greedy"):
        result        = self.optimize(p, method=method)
        selected_hubs = result["selected_hubs"]
        covered       = set(result["covered_facilities"])

        all_coords = np.vstack([self.candidates, self.facilities])
        center = all_coords.mean(axis=0)
        m = folium.Map(location=center.tolist(), zoom_start=6, tiles="OpenStreetMap")
        folium.TileLayer("CartoDB positron").add_to(m)

        grp_hubs      = folium.FeatureGroup(name="Drone Hub Facilities")
        grp_covered   = folium.FeatureGroup(name="Covered Facilities")
        grp_uncovered = folium.FeatureGroup(name="Uncovered Facilities")
        grp_circles   = folium.FeatureGroup(name=f"Coverage Radius ({self.operational_radius} km)")

        for idx in selected_hubs:
            lat, lon = self.candidates[idx]
            name = self.candidate_names[idx]
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(
                    f"<b>{name}</b><br><i>Drone Hub</i><br>Radius: {self.operational_radius} km",
                    max_width=220),
                icon=folium.Icon(color="red", icon="bullseye", prefix="fa"),
                tooltip=f"🛸 Hub: {name}",
            ).add_to(grp_hubs)
            folium.Circle(
                [lat, lon],
                radius=self.operational_radius * 1000,
                color="red", fill=True, fillColor="red",
                fillOpacity=0.08, opacity=0.45,
            ).add_to(grp_circles)

        for idx in covered:
            lat, lon = self.facilities[idx]
            folium.CircleMarker(
                [lat, lon], radius=4,
                popup=f"<b>{self.facility_names[idx]}</b><br>✅ Covered",
                color="green", fillColor="green", fillOpacity=0.7,
                tooltip=f"{self.facility_names[idx]} (Covered)",
            ).add_to(grp_covered)

        for idx in set(range(len(self.facilities))) - covered:
            lat, lon = self.facilities[idx]
            folium.CircleMarker(
                [lat, lon], radius=4,
                popup=f"<b>{self.facility_names[idx]}</b><br>❌ Not Covered",
                color="orange", fillColor="orange", fillOpacity=0.7,
                tooltip=f"{self.facility_names[idx]} (Not Covered)",
            ).add_to(grp_uncovered)

        for grp in [grp_circles, grp_hubs, grp_covered, grp_uncovered]:
            grp.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MiniMap().add_to(m)

        return m, result


# ── Streamlit App ─────────────────────────────────────────────────────────────

DATA_PATH = "sub_saharan_health_facilities.xlsx"  


def main():
    st.set_page_config(page_title="Drone Hub Optimizer", page_icon="🛸", layout="wide")
    st.title("🛸 Drone Hub Optimizer — Health Facility Network")
    st.markdown(
        "Select which health facilities should host drone base stations to "
        "maximise commodity delivery coverage across the facility network."
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        df_all = load_facilities(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Could not find `{DATA_PATH}`. Please update `DATA_PATH` at the top of the script.")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Configuration")

    countries = sorted(df_all["country"].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("🌍 Select Country", countries)

    df_country = df_all[df_all["country"] == selected_country].reset_index(drop=True)

    if len(df_country) < 2:
        st.warning(f"Not enough facilities found for **{selected_country}**. Please choose another country.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏥 Hub Location Eligibility")

    hospitals_only = st.sidebar.toggle(
        "Hospitals only as hub candidates",
        value=False,
        help="Restrict drone hub placement to hospital-type facilities to reflect higher setup costs",
    )

    hospitals_coverage_only = st.sidebar.toggle(
        "Hospitals only as coverage targets",
        value=False,
        help="Only count hospitals as facilities that need to be covered",
    )

    # Coverage targets
    df_targets = df_country[df_country["is_hospital"]] if hospitals_coverage_only else df_country
    if len(df_targets) == 0:
        st.warning(f"No hospitals found in **{selected_country}** to use as coverage targets.")
        st.stop()

    facilities     = df_targets[["latitude", "longitude"]].values.tolist()
    facility_names = df_targets["facility_n"].tolist()

    # Hub candidates
    df_candidates = df_country[df_country["is_hospital"]] if hospitals_only else df_country
    if len(df_candidates) == 0:
        st.warning(f"No hospitals found in **{selected_country}**. Try disabling 'Hospitals only'.")
        st.stop()

    candidate_coords = df_candidates[["latitude", "longitude"]].values.tolist()
    candidate_names  = df_candidates["facility_n"].tolist()

    n_hospitals = df_country["is_hospital"].sum()
    st.sidebar.caption(
        f"{n_hospitals} hospitals / {len(df_country)} total facilities in {selected_country} — "
        f"covering {len(df_targets)} target{'s' if len(df_targets) != 1 else ''}"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Optimization Settings")

    num_hubs = st.sidebar.slider(
        "Number of Drone Hubs to Activate",
        min_value=1,
        max_value=min(30, len(df_candidates)),
        value=min(3, len(df_candidates)),
        help="The algorithm will pick the best eligible facilities to host drone hubs",
    )

    operational_radius = st.sidebar.slider(
        "Drone Operational Radius (km)",
        min_value=20, max_value=150, value=80, step=5,
        help="Maximum distance a drone can travel from its hub facility",
    )

    method = st.sidebar.selectbox(
        "Optimization Method",
        options=["greedy", "exact"],
        index=0,
        help="Greedy is fast and near-optimal; Exact guarantees the best solution but is slow for many hubs",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Summary")
    st.sidebar.metric("Health Facilities", len(df_country))
    st.sidebar.metric("Hubs to Select", num_hubs)

    # ── Run optimizer ─────────────────────────────────────────────────────────
    optimizer = DroneHubOptimizer(
        facilities=facilities,
        operational_radius=operational_radius,
        facility_names=facility_names,
        candidate_coords=candidate_coords,
        candidate_names=candidate_names,
    )

    folium_map, result = optimizer.create_folium_map(num_hubs, method=method)

    # ── Metrics ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Drone Hubs Active", num_hubs)
    with col2:
        st.metric("Facilities Covered", f"{result['coverage_count']} / {len(facilities)}")
    with col3:
        st.metric("Coverage Rate", f"{result['coverage_rate']:.1%}")
    with col4:
        uncovered = len(facilities) - result["coverage_count"]
        st.metric("Uncovered Facilities", uncovered,
                  delta=f"-{uncovered}" if uncovered > 0 else "0",
                  delta_color="inverse")



    # ── Map ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗺️ Coverage Map")
    st_folium(folium_map, width=1400, height=600)

    # ── Selected hubs ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📍 Selected Drone Hub Facilities")
    cols = st.columns(min(3, len(result["selected_hubs"])))
    for i, hub_idx in enumerate(result["selected_hubs"]):
        with cols[i % 3]:
            lat, lon = candidate_coords[hub_idx]
            n_incremental = result["incremental_coverage"][hub_idx]
            st.info(
                f"**{candidate_names[hub_idx]}**  \n"
                f"Lat: {lat:.4f}  \nLon: {lon:.4f}  \n"
                f"📦 +**{n_incremental}** new facilit{'y' if n_incremental == 1 else 'ies'} covered"
            )

    # ── Coverage curve ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Coverage Curve")

    with st.spinner("Computing coverage curve…"):
        rates = optimizer.coverage_curve(num_hubs, method=method)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, num_hubs + 1)),
        y=[r * 100 for r in rates],
        mode="lines+markers",
        line=dict(color="#e63946", width=2.5),
        marker=dict(size=8, color="#e63946"),
        hovertemplate="Hubs: %{x}<br>Coverage: %{y:.1f}%<extra></extra>",
    ))
    fig.add_vline(
        x=num_hubs,
        line_dash="dash", line_color="gray", line_width=1.5,
        annotation_text=f"Current: {num_hubs} hub{'s' if num_hubs > 1 else ''}",
        annotation_position="top left",
    )
    fig.update_layout(
        xaxis_title="Number of Drone Hubs",
        yaxis_title="Facilities Covered (%)",
        yaxis=dict(range=[0, 105], ticksuffix="%"),
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        hovermode="x unified",
        plot_bgcolor="white",
        margin=dict(t=30, b=40, l=60, r=30),
        height=350,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)
    # ── Detail expander ───────────────────────────────────────────────────────
    with st.expander("📋 Detailed Coverage Information"):
        st.write(f"**Country:** {selected_country}")
        st.write(f"**Covered:** {result['coverage_count']} / {len(facilities)}")
        st.write(f"**Coverage Rate:** {result['coverage_rate']:.2%}")
        st.write(f"**Hub Facilities:** {[candidate_names[i] for i in result['selected_hubs']]}")
        uncovered_idx = sorted(set(range(len(facilities))) - set(result["covered_facilities"]))
        if uncovered_idx:
            st.write(f"**Uncovered Facilities:** {[facility_names[i] for i in uncovered_idx]}")
        else:
            st.success("🎉 All facilities are covered!")

    # ── Distance export ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Export Distance Data")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Download a spreadsheet with air distances (and optionally road distances) from each hub to all facilities.")
    with col2:
        include_road = st.checkbox("Include road distances", value=False)
        if include_road:
            st.warning("⚠️ Road distance calculation is slow — expect several minutes for large countries.")
            use_parallel = st.toggle("Use parallel processing (faster, more network requests)", value=True)

    if st.button("🔄 Generate Distance Report", type="primary"):
        with st.spinner("Calculating distances…"):
            if not include_road:
                df_dist = create_distance_spreadsheet(
                    optimizer, result["selected_hubs"], candidate_names,
                    facilities, use_road_distance=False)
            elif use_parallel:
                df_dist = create_distance_spreadsheet_parallel(
                    optimizer, result["selected_hubs"], candidate_names, facilities)
            else:
                df_dist = create_distance_spreadsheet(
                    optimizer, result["selected_hubs"], candidate_names,
                    facilities, use_road_distance=True)

            st.success(f"✅ Generated {len(df_dist)} distance records!")
            st.dataframe(df_dist.head(20), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                avg_air = df_dist["Air_Distance_KM"][df_dist["Within_Range"] == "Yes"].mean()
                st.metric("Avg Air Distance (in-range)", f"{avg_air:.2f} km")
            with c2:
                df_dist["Road_Distance_KM"] = pd.to_numeric(df_dist["Road_Distance_KM"], errors="coerce")
                avg_road = df_dist["Road_Distance_KM"][df_dist["Within_Range"] == "Yes"].mean()
                st.metric("Avg Road Distance (in-range)", f"{avg_road:.2f} km")
            with c3:
                within = len(df_dist[df_dist["Within_Range"] == "Yes"])
                st.metric("Routes Within Range", f"{within}/{len(df_dist)}")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_dist.to_excel(writer, sheet_name="Distance Analysis", index=False)
                summary = pd.DataFrame({
                    "Metric": ["Country", "Drone Hubs Active", "Total Facilities",
                               "Avg Air Distance (km)", "Avg Road Distance (km)",
                               "Routes Within Range", "Coverage Rate"],
                    "Value": [
                        selected_country,
                        len(result["selected_hubs"]),
                        len(facilities),
                        f"{df_dist['Air_Distance_KM'].mean():.2f}",
                        f"{pd.to_numeric(df_dist['Road_Distance_KM'], errors='coerce').mean():.2f}"
                        if include_road else "N/A",
                        within,
                        f"{result['coverage_rate']:.1%}",
                    ],
                })
                summary.to_excel(writer, sheet_name="Summary", index=False)
            output.seek(0)

            st.download_button(
                label="📥 Download Excel Report",
                data=output,
                file_name=f"drone_hub_analysis_{selected_country}_{num_hubs}_hubs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ── Technical notes ───────────────────────────────────────────────────────
    with st.expander("📚 Technical Notes: Algorithm Details"):
        st.markdown("""
## Capacitated P-Median Optimization

This app selects which **p** health facilities should host drone base stations
to maximise the number of facilities within delivery range.

### How it works
1. A facility-to-facility distance matrix is computed using the Haversine formula.
2. A binary coverage matrix marks pairs within the operational radius.
3. The algorithm iteratively selects the facility that covers the most
   currently-uncovered facilities (greedy), or exhaustively tests all
   combinations (exact).

### Optimization methods
- **Greedy** — fast, near-optimal (typically 90–98% of best possible), suitable for large networks.
- **Exact** — tests all combinations; guarantees the optimal solution but becomes slow beyond ~10 hubs.

### Distance formula
$$d = 2R \\cdot \\arcsin\\!\\left(\\sqrt{\\sin^2\\!\\frac{\\Delta\\phi}{2} + \\cos\\phi_1\\cos\\phi_2\\sin^2\\!\\frac{\\Delta\\lambda}{2}}\\right)$$

### Limitations
- Binary coverage model (in/out of range) — does not model service quality gradients.
- Static demand — does not account for temporal variation in facility needs.
- No routing optimisation — delivery path sequencing is not considered.
        """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:gray;'>"
        "Built with Streamlit · P-Median Algorithm · Haversine Distance . CDC/GHC/OD/ADS Team"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
