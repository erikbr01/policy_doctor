"""Diagnostics tab for clustering analysis.

This module provides diagnostic tools to understand why clustering may not show clear structure.
"""

from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from influence_visualizer.data_loader import InfluenceData
from influence_visualizer.render_clustering import (
    extract_demo_embeddings,
    extract_rollout_embeddings,
)
from influence_visualizer.render_heatmaps import SplitType, get_split_data


def diagnose_embeddings(embeddings: np.ndarray, name: str = "Embeddings"):
    """Run diagnostics on embedding vectors."""
    st.subheader(f"📊 {name} Statistics")

    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Shape", f"{embeddings.shape[0]} × {embeddings.shape[1]}")
    with col2:
        st.metric("Mean", f"{embeddings.mean():.4f}")
    with col3:
        st.metric("Std", f"{embeddings.std():.4f}")
    with col4:
        st.metric("Range", f"[{embeddings.min():.2f}, {embeddings.max():.2f}]")

    # Feature scale analysis
    st.markdown("### 📏 Feature Scale Analysis")
    feature_means = embeddings.mean(axis=0)
    feature_stds = embeddings.std(axis=0)

    std_ratio = feature_stds.max() / (feature_stds.min() + 1e-10)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Feature Means Range",
            f"[{feature_means.min():.4f}, {feature_means.max():.4f}]",
        )
    with col2:
        st.metric(
            "Feature Stds Range",
            f"[{feature_stds.min():.4f}, {feature_stds.max():.4f}]",
        )
    with col3:
        st.metric("Std Ratio (max/min)", f"{std_ratio:.2f}x")

    if std_ratio > 10:
        st.error(f"""
        ⚠️ **WARNING: Features have very different scales!**

        Std ratio: {std_ratio:.2f}x (max/min)

        This will dominate distance calculations in PCA/UMAP/t-SNE.

        **→ SOLUTION: Apply StandardScaler before dimensionality reduction**
        """)
    elif std_ratio > 3:
        st.warning(f"""
        ⚠️ Features have moderately different scales (ratio: {std_ratio:.2f}x).

        Consider using StandardScaler for better results.
        """)
    else:
        st.success(
            f"✅ Feature scales are relatively uniform (ratio: {std_ratio:.2f}x)"
        )

    # Check for outliers
    st.markdown("### 🔍 Outlier Detection")
    z_scores = np.abs((embeddings - embeddings.mean()) / (embeddings.std() + 1e-10))
    outliers = (z_scores > 3).any(axis=1)
    n_outliers = outliers.sum()
    outlier_pct = 100 * n_outliers / len(embeddings)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Samples with |z-score| > 3", f"{n_outliers} ({outlier_pct:.1f}%)")

    if outlier_pct > 5:
        st.warning(f"""
        ⚠️ **{outlier_pct:.1f}% outliers detected** (>{5}% threshold)

        Consider using RobustScaler (robust to outliers) instead of StandardScaler.
        """)
    else:
        st.success(f"✅ Low outlier rate ({outlier_pct:.1f}%)")

    # PCA variance analysis
    st.markdown("### 🎯 PCA Variance Explained")
    n_components = min(50, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)

    cumsum = np.cumsum(pca.explained_variance_ratio_)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("First 2 PCs", f"{100 * cumsum[1]:.1f}%")
    with col2:
        st.metric("First 5 PCs", f"{100 * cumsum[4]:.1f}%")
    with col3:
        st.metric("First 10 PCs", f"{100 * cumsum[9]:.1f}%")
    with col4:
        n_90 = np.argmax(cumsum >= 0.9) + 1
        st.metric("For 90% variance", f"{n_90} PCs")

    if cumsum[1] < 0.2:
        st.warning(f"""
        ⚠️ **First 2 PCs explain <20% variance** ({100 * cumsum[1]:.1f}%)

        Data has high intrinsic dimensionality. May not have clear 2D cluster structure.

        Consider:
        - Using more intermediate dimensions (e.g., PCA with 50 components → t-SNE)
        - UMAP may work better than PCA for high-dimensional data
        """)
    else:
        st.success(f"✅ First 2 PCs capture {100 * cumsum[1]:.1f}% of variance")

    # Plot variance explained
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumsum) + 1)),
            y=100 * cumsum,
            mode="lines+markers",
            name="Cumulative Variance",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
        )
    )
    fig.add_hline(
        y=90, line_dash="dash", line_color="green", annotation_text="90% threshold"
    )
    fig.add_hline(
        y=20, line_dash="dash", line_color="red", annotation_text="20% (2D threshold)"
    )
    fig.update_layout(
        title="PCA Cumulative Variance Explained",
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Variance Explained (%)",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")

    return {
        "shape": embeddings.shape,
        "pca": pca,
        "variance_2d": cumsum[1],
        "variance_10d": cumsum[9],
        "n_components_90": n_90,
        "std_ratio": std_ratio,
        "n_outliers": n_outliers,
        "outlier_pct": outlier_pct,
    }


def compare_scaling_methods(embeddings: np.ndarray, name: str = "Embeddings"):
    """Compare different scaling methods visually."""
    st.subheader(f"🔬 Scaling Method Comparison: {name}")

    st.markdown("""
    Compare how different scaling methods affect the 2D PCA projection.
    """)

    # Apply different scaling methods
    scalers = {
        "None": None,
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
        "MinMaxScaler": MinMaxScaler(),
    }

    results = {}
    for method_name, scaler in scalers.items():
        if scaler is None:
            scaled_embeddings = embeddings
        else:
            scaled_embeddings = scaler.fit_transform(embeddings)

        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(scaled_embeddings)
        variance_explained = pca.explained_variance_ratio_.sum()

        results[method_name] = {
            "embeddings_2d": embeddings_2d,
            "variance": variance_explained,
        }

    # Plot side-by-side comparison
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{name}<br>Variance: {results[name]['variance']:.1%}"
            for name in scalers.keys()
        ],
    )

    for idx, method_name in enumerate(scalers.keys()):
        row = idx // 2 + 1
        col = idx % 2 + 1

        emb_2d = results[method_name]["embeddings_2d"]

        fig.add_trace(
            go.Scatter(
                x=emb_2d[:, 0],
                y=emb_2d[:, 1],
                mode="markers",
                marker=dict(size=5, opacity=0.6),
                name=method_name,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"PCA Projections with Different Scaling Methods",
        height=800,
    )

    st.plotly_chart(fig, width="stretch")

    # Show variance comparison
    st.markdown("### Variance Explained Comparison")
    variance_df = {
        "Scaling Method": list(scalers.keys()),
        "Variance Explained (2 PCs)": [
            f"{100 * results[m]['variance']:.1f}%" for m in scalers.keys()
        ],
    }
    st.dataframe(variance_df, width="stretch")

    # Recommendation
    best_method = max(results.items(), key=lambda x: x[1]["variance"])[0]
    st.success(f"""
    ✅ **Recommendation**: {best_method} captured the most variance ({100 * results[best_method]["variance"]:.1f}%)

    - **StandardScaler**: Best for normally distributed features
    - **RobustScaler**: Best when you have outliers
    - **MinMaxScaler**: Best when you need bounded [0,1] range
    - **None**: Only if features already have comparable scales
    """)


def render_diagnostics_tab(data: InfluenceData, demo_split: SplitType = "train"):
    """Render the diagnostics tab."""
    st.header("🔬 Clustering Diagnostics")

    st.markdown("""
    This tab helps diagnose why your clustering visualizations may not show clear structure.

    **Common Issues:**
    - Missing feature scaling (most common!)
    - High-dimensional data with low variance in 2D
    - Outliers affecting distance calculations
    - Data naturally lacks discrete clusters
    """)

    load_key = f"diagnostics_loaded_{demo_split}"
    with st.expander("Load clustering diagnostics", expanded=False):
        st.caption(
            "Extract demo/rollout embeddings and run PCA, scaling comparison, and recommendations (can take a while)."
        )
        if st.button(
            "Load diagnostics",
            key=f"diagnostics_btn_load_{demo_split}",
        ):
            st.session_state[load_key] = True

    if not st.session_state.get(load_key, False):
        return

    st.divider()

    # Embedding type selector
    embedding_type = st.radio(
        "Select embedding type to diagnose",
        options=["Demo Embeddings", "Rollout Embeddings"],
        horizontal=True,
    )

    st.divider()

    try:
        if embedding_type == "Demo Embeddings":
            st.markdown("### Demo Embeddings")
            st.markdown(
                "Each demo gets an embedding vector representing its influence across all rollout samples."
            )

            with st.spinner("Extracting demo embeddings..."):
                embeddings, metadata = extract_demo_embeddings(data, split=demo_split)

            if len(embeddings) == 0:
                st.error("No demo embeddings found.")
                return

            st.success(f"✅ Extracted {len(embeddings)} demo embeddings")

            # Run diagnostics
            stats = diagnose_embeddings(embeddings, "Demo Embeddings")

            st.divider()

            # Compare scaling methods
            compare_scaling_methods(embeddings, "Demo Embeddings")

        else:  # Rollout Embeddings
            st.markdown("### Rollout Embeddings")
            st.markdown(
                "Each rollout gets an embedding vector representing influences from all demonstration samples."
            )

            with st.spinner("Extracting rollout embeddings..."):
                embeddings, metadata = extract_rollout_embeddings(
                    data, split=demo_split
                )

            if len(embeddings) == 0:
                st.error("No rollout embeddings found.")
                return

            st.success(f"✅ Extracted {len(embeddings)} rollout embeddings")

            # Run diagnostics
            stats = diagnose_embeddings(embeddings, "Rollout Embeddings")

            st.divider()

            # Compare scaling methods
            compare_scaling_methods(embeddings, "Rollout Embeddings")

    except Exception as e:
        st.error(f"Error running diagnostics: {e}")
        import traceback

        st.code(traceback.format_exc())

    st.divider()

    # Summary recommendations
    st.markdown("## 📋 Summary & Recommendations")

    st.markdown("""
    ### ✅ MUST DO:
    1. **Add feature scaling** before PCA/UMAP/t-SNE
       - Use StandardScaler (recommended for most cases)
       - Use RobustScaler if you have >5% outliers
       - Scaling selector is now available in the Clustering tab!

    ### 🎯 SHOULD DO:
    2. **Replace np.mean() aggregation** with richer features
       - Current: `demo_embedding = np.mean(demo_influence, axis=1)`
       - Better: Use min, max, std, percentiles to capture temporal patterns
       - Averaging washes out important temporal dynamics

    3. **Tune parameters** for your dataset
       - t-SNE perplexity: Try values from 5 to 100
       - UMAP n_neighbors: Try 5-50 depending on dataset size
       - UMAP min_dist: Try 0.0 for tighter clusters

    ### 💡 CONSIDER:
    4. **Check if data has natural clusters**
       - Color by known labels (success/quality) in clustering plots
       - If no pattern emerges, data may be naturally continuous
       - TRAK influence scores may have smooth gradients, not discrete groups

    5. **Try advanced methods** if basics don't work
       - Kernel PCA (non-linear without UMAP cost)
       - NMF (interpretable parts-based decomposition)
       - Factor Analysis (handles noisy measurements)
    """)

    st.info("""
    💡 **Tip**: The Clustering tab now has a "Feature scaling" selector at the top.
    Try different scaling methods and see which reveals the best structure!
    """)
