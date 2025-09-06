# =====================================================
# Streamlit App for Lithology Cross-Section
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    log_loss, accuracy_score, precision_score, recall_score, f1_score
)
from bayes_opt import BayesianOptimization

st.set_page_config(page_title="Geotechnical Site Characterization", layout="wide")
st.title("🌍 Lithology Cross-Section with Uncertainty")

# =====================================================
# 1. File Upload
# =====================================================
uploaded_file = st.file_uploader("Upload Borehole Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Borehole Data Preview", df.head())

    required_cols = ["X", "Y", "TE", "BE", "Lithology"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        # Encode lithology
        df['Soil_Label'] = df['Lithology'].astype('category').cat.codes
        soil_label_map = dict(enumerate(df['Lithology'].astype('category').cat.categories))

        features = ['X', 'Y', 'TE', 'BE']
        X = df[features]
        y = df['Soil_Label']

        # =====================================================
        # 2. Train/Test Split
        # =====================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1, stratify=y
        )

        # =====================================================
        # 3. Bayesian Optimization Controls
        # =====================================================
        st.sidebar.subheader("⚙️ Model Settings")
        do_opt = st.sidebar.checkbox("Run Bayesian Optimization", value=False)
        n_init = st.sidebar.number_input("Initial points", min_value=0, max_value=20, value=5)
        n_iter = st.sidebar.number_input("Iterations", min_value=0, max_value=50, value=10)

        default_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }

        def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
            n_estimators = int(round(n_estimators))
            max_depth = int(round(max_depth))
            min_samples_split = int(round(min_samples_split))
            min_samples_leaf = int(round(min_samples_leaf))
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth if max_depth > 0 else None,
                min_samples_split=max(2, min_samples_split),
                min_samples_leaf=max(1, min_samples_leaf),
                max_features='sqrt',
                class_weight='balanced',
                bootstrap=True,
                random_state=1,
                n_jobs=-1
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
            scores = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring='f1_macro', n_jobs=-1)
            return float(np.mean(scores))

        best_params = default_params.copy()
        if do_opt:
            pbounds = {
                'n_estimators': (10, 300),
                'max_depth': (3, 40),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
            }
            optimizer = BayesianOptimization(
                f=rf_cv, pbounds=pbounds, random_state=1, verbose=2
            )
            optimizer.maximize(init_points=n_init, n_iter=n_iter)
            params = optimizer.max['params']
            best_params = {
                'n_estimators': int(round(params['n_estimators'])),
                'max_depth': int(round(params['max_depth'])),
                'min_samples_split': int(round(params['min_samples_split'])),
                'min_samples_leaf': int(round(params['min_samples_leaf']))
            }

        st.write("### Best Parameters", best_params)

        # =====================================================
        # 4. Train Final Model
        # =====================================================
        rf = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'] if best_params['max_depth'] > 0 else None,
            min_samples_split=max(2, best_params['min_samples_split']),
            min_samples_leaf=max(1, best_params['min_samples_leaf']),
            max_features='sqrt',
            class_weight='balanced',
            bootstrap=True,
            random_state=1,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Metrics
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)
        st.write("### Test Metrics")
        st.write(f"- Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.write(f"- Log loss: {log_loss(y_test, y_proba):.4f}")
        st.write(f"- Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        st.write(f"- Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        st.write(f"- F1 Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

        # =====================================================
        # 5. Cross-Section Prediction
        # =====================================================
        st.subheader("🌐 Cross-Section Prediction Settings")
        X_start = st.number_input("X Start", value=float(df["X"].min()))
        Y_start = st.number_input("Y Start", value=float(df["Y"].min()))
        X_end = st.number_input("X End", value=float(df["X"].max()))
        Y_end = st.number_input("Y End", value=float(df["Y"].max()))
        depth_top = st.number_input("Top Depth (TE)", value=float(df["TE"].max()))
        depth_bottom = st.number_input("Bottom Depth (BE)", value=float(df["BE"].min()))
        depth_step = st.number_input("Depth Step", value=0.5)
        num_stations = st.number_input("Number of Stations", value=80)

        if st.button("Generate Cross-Section"):
            Xs = np.linspace(X_start, X_end, num_stations)
            Ys = np.linspace(Y_start, Y_end, num_stations)
            TE = np.arange(depth_top, depth_bottom, -depth_step)
            BE = TE - depth_step

            data_grid = []
            for x, y in zip(Xs, Ys):
                for te, be in zip(TE, BE):
                    data_grid.append([x, y, te, be])
            grid_df = pd.DataFrame(data_grid, columns=['X', 'Y', 'TE', 'BE'])

            # Prediction
            probs = rf.predict_proba(grid_df[features])
            entropy = 1 - np.sum(probs ** 2, axis=1)
            grid_df['entropy'] = entropy
            grid_df['Soil_Label'] = rf.predict(grid_df[features])
            grid_df['Lithology'] = grid_df['Soil_Label'].map(soil_label_map)

            # Distance projection
            dx, dy = X_end - X_start, Y_end - Y_start
            total_distance = np.sqrt(dx**2 + dy**2)

            def project_distance(x, y):
                vec_section = np.array([dx, dy])
                vec_point = np.array([x - X_start, y - Y_start])
                return np.dot(vec_section, vec_point) / total_distance

            grid_df['Distance'] = grid_df.apply(lambda row: project_distance(row['X'], row['Y']), axis=1)
            df['Distance'] = df.apply(lambda row: project_distance(row['X'], row['Y']), axis=1)

            # =====================================================
            # 6. Plot Lithology Cross-Section
            # =====================================================
            unique_soils = list(df['Lithology'].astype('category').cat.categories)
            colors = plt.cm.tab20.colors
            soil_color_map = {soil: colors[i % len(colors)] for i, soil in enumerate(unique_soils)}

            soil_to_code = {soil: i for i, soil in enumerate(unique_soils)}
            grid_df["Lithology_code"] = grid_df["Lithology"].map(soil_to_code)

            litho_pivot = grid_df.pivot_table(
                index="TE", columns="Distance", values="Lithology_code", aggfunc="first"
            )
            X, Y = np.meshgrid(litho_pivot.columns, litho_pivot.index)

            cmap = ListedColormap([soil_color_map[s] for s in unique_soils])

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.pcolormesh(X, Y, litho_pivot.values,
                          cmap=cmap, shading="auto",
                          vmin=0, vmax=len(unique_soils) - 1)

            ax.set_ylim(depth_bottom, depth_top)
            ax.set_xlim(0, total_distance)
            ax.set_ylabel("Elevation (m)")
            ax.set_xlabel("Cross Section Distance (m)")
            ax.set_title("2D Lithology Cross-Section")

            legend_patches = [mpatches.Patch(color=soil_color_map[s], label=s) for s in unique_soils]
            ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")
            st.pyplot(fig)

            # =====================================================
            # 7. Plot Uncertainty
            # =====================================================
            entropy_pivot = grid_df.pivot_table(
                index="TE", columns="Distance", values="entropy", aggfunc="mean"
            )
            X, Y = np.meshgrid(entropy_pivot.columns, entropy_pivot.index)

            fig, ax = plt.subplots(figsize=(12, 6))
            c = ax.pcolormesh(X, Y, entropy_pivot.values, cmap="viridis", shading="auto")

            ax.set_ylim(depth_bottom, depth_top)
            ax.set_xlim(0, total_distance)
            ax.set_ylabel("Elevation (m)")
            ax.set_xlabel("Cross Section Distance (m)")
            ax.set_title("Uncertainty (Entropy) along Cross-Section")
            plt.colorbar(c, ax=ax, label="Entropy")

            st.pyplot(fig)
