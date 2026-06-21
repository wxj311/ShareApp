import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# =========================
# App configuration
# =========================
st.set_page_config(
    page_title="Concurrent PCI Classification Tool",
    page_icon="🧠",
    layout="wide",
)

# Keep the existing model filename in your current GitHub repository.
# If your final Random Forest model file has a different name, change this line only.
MODEL_FILE = "xgboost_model_260121.pkl"
MODEL_DISPLAY_NAME = "Random Forest"

FEATURE_CONFIG = [
    {"name": "SBP", "label": "SBP (mmHg)", "min": 88, "max": 225, "default": 90, "step": 1},
    {"name": "PLT", "label": "PLT (10³/µL)", "min": 18, "max": 644, "default": 212, "step": 1},
    {"name": "UA", "label": "UA (µmol/L)", "min": 68, "max": 607, "default": 270, "step": 1},
    {"name": "TG", "label": "TG (mg/dL)", "min": 0.46, "max": 7.32, "default": 1.73, "step": 0.01},
    {"name": "BAD", "label": "BAD (mm)", "min": 1.9, "max": 7.0, "default": 4.7, "step": 0.1},
    {"name": "BAL", "label": "BAL (mm)", "min": 0.2, "max": 36.4, "default": 27.0, "step": 0.1},
    {"name": "BAMD", "label": "BAMD (mm)", "min": 0.0, "max": 12.5, "default": 0.0, "step": 0.1},
]

plt.rcParams["axes.unicode_minus"] = False


# =========================
# Model utilities
# =========================
@st.cache_resource
def load_model_and_explainer():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, MODEL_FILE)

    if not os.path.exists(model_path):
        st.error(
            f"Model file not found: {MODEL_FILE}. Please place the model file in the same folder as this app."
        )
        st.stop()

    with open(model_path, "rb") as file:
        loaded_model = pickle.load(file)

    loaded_explainer = shap.TreeExplainer(loaded_model)
    return loaded_model, loaded_explainer


def get_classification_probability(loaded_model, data: pd.DataFrame) -> float:
    if not hasattr(loaded_model, "predict_proba"):
        st.error("The loaded model does not support probability output through predict_proba().")
        st.stop()

    return float(loaded_model.predict_proba(data)[0, 1])


def build_shap_explanation(loaded_explainer, data: pd.DataFrame) -> shap.Explanation:
    """Create a single-case SHAP explanation for the concurrent PCI class."""
    try:
        explanation = loaded_explainer(data)
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values).ravel()

        if values.ndim == 3:
            shap_values_for_case = values[0, :, 1]
            base_value_for_class = float(base_values[1] if base_values.size > 1 else base_values[0])
        elif values.ndim == 2:
            shap_values_for_case = values[0]
            base_value_for_class = float(base_values[0])
        else:
            shap_values_for_case = values
            base_value_for_class = float(base_values[0])

    except Exception:
        raw_values = loaded_explainer.shap_values(data)
        expected_value = loaded_explainer.expected_value

        if isinstance(raw_values, list):
            shap_values_for_case = np.asarray(raw_values[1])[0]
            expected_array = np.asarray(expected_value).ravel()
            base_value_for_class = float(expected_array[1] if expected_array.size > 1 else expected_array[0])
        else:
            raw_array = np.asarray(raw_values)
            if raw_array.ndim == 3:
                shap_values_for_case = raw_array[0, :, 1]
            elif raw_array.ndim == 2:
                shap_values_for_case = raw_array[0]
            else:
                shap_values_for_case = raw_array

            expected_array = np.asarray(expected_value).ravel()
            base_value_for_class = float(expected_array[1] if expected_array.size > 1 else expected_array[0])

    return shap.Explanation(
        values=shap_values_for_case,
        base_values=base_value_for_class,
        data=data.iloc[0].to_numpy(),
        feature_names=data.columns.tolist(),
    )


# =========================
# User interface
# =========================
def collect_input_data() -> pd.DataFrame:
    st.sidebar.header("Selection Panel")

    values = {}
    for feature in FEATURE_CONFIG:
        values[feature["name"]] = st.sidebar.slider(
            feature["label"],
            min_value=feature["min"],
            max_value=feature["max"],
            value=feature["default"],
            step=feature["step"],
        )

    return pd.DataFrame({feature["name"]: [values[feature["name"]]] for feature in FEATURE_CONFIG})


def render_shap_plot(loaded_explainer, data: pd.DataFrame):
    st.subheader("SHAP Feature Contribution to Classification Output")

    shap_explanation = build_shap_explanation(loaded_explainer, data)

    plot_col, blank_col = st.columns([0.58, 0.42])
    with plot_col:
        fig = plt.figure(figsize=(7.2, 4.2), dpi=120)
        shap.plots.waterfall(shap_explanation, max_display=7, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7.2, 4.2)

        for ax in fig.axes:
            ax.tick_params(axis="both", labelsize=8)
            ax.title.set_size(10)
            ax.xaxis.label.set_size(9)
            ax.yaxis.label.set_size(9)
            for text in ax.texts:
                text.set_fontsize(8)

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        plt.close(fig)


def main():
    model, explainer = load_model_and_explainer()
    input_data = collect_input_data()

    st.title(f"Classification of concurrent PCI based on {MODEL_DISPLAY_NAME} model")

    if st.button("Run classification"):
        classification_probability = get_classification_probability(model, input_data)
        st.markdown(
            f"### Classification probability for concurrent PCI: {classification_probability * 100:.2f}%"
        )
        render_shap_plot(explainer, input_data)


if __name__ == "__main__":
    main()
