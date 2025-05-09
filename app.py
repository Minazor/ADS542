# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import warnings
import streamlit.components.v1 as components
import shap
warnings.filterwarnings("ignore")

# 🧾 Page configuration
st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    page_icon="💼",
    layout="wide"
)

# 🚦 Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Data Preview", "Feature Engineering", "Model Performance", "Try Prediction"]
)

# 📥 Load data
@st.cache_data
def load_data():
    df = pd.read_csv("bank+marketing/bank-additional/bank-additional.csv", sep=";")
    df['y_binary'] = df['y'].map({'no': 0, 'yes': 1})
    return df

df = load_data()

# 📃 Page 1: Overview
if page == "Overview":
    st.title("📊 Bank Marketing Term Deposit Classifier")

    st.markdown("""
    This app uses the UCI Bank Marketing dataset to predict whether a client will subscribe to a term deposit.
    [Dataset source](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

    ### 📦 Dataset Summary
    - **File Used**: `bank-additional.csv` (10% random sample of full dataset)
    - **Rows**: 4,119 clients
    - **Features**: 20 input variables + 1 output variable (`y`)
    - **Target Variable**: `y` — Whether the client subscribed to a term deposit (`yes`/`no`)
    - **Source**: Direct marketing campaigns of a Portuguese bank (2008–2010)

    ### 🏷️ Feature Descriptions
    - **age**: Client's age (numeric)
    - **job**: Type of job (e.g., admin., technician, retired)
    - **marital**: Marital status (married, single, divorced)
    - **education**: Education level (basic, high.school, university.degree, etc.)
    - **default**: Has credit in default?
    - **housing**: Has a housing loan?
    - **loan**: Has a personal loan?
    - **contact**: Contact communication type (cellular, telephone)
    - **month**: Last contact month (jan–dec)
    - **day_of_week**: Last contact day (mon–fri)
    - **duration**: Duration of last contact in seconds — *excluded from modeling due to leakage*
    - **campaign**: Number of contacts during the current campaign
    - **pdays**: Days since the client was last contacted (999 means never)
    - **previous**: Number of contacts before this campaign
    - **poutcome**: Outcome of the previous campaign
    - **emp.var.rate**: Employment variation rate (quarterly)
    - **cons.price.idx**: Consumer price index (monthly)
    - **cons.conf.idx**: Consumer confidence index (monthly)
    - **euribor3m**: 3-month Euribor rate (daily)
    - **nr.employed**: Number of employees (quarterly)
    """)

    st.markdown("---")
    st.subheader("🔎 Example Rows")
    st.dataframe(df.head())

    st.markdown("---")
    st.subheader("📊 Target Variable Distribution")
    target_counts = df['y'].value_counts().reset_index()
    target_counts.columns = ['Response', 'Count']

    import plotly.express as px
    fig = px.bar(
        target_counts,
        x='Response',
        y='Count',
        color='Response',
        text='Count',
        title="Term Deposit Subscription ('y') Class Distribution",
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(fig, use_container_width=True)


# 📊 Page 2: Data Preview
elif page == "Data Preview":
    st.title("🔍 Data Overview")

    # Ayır: sayısal ve kategorik sütunlar
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    tab1, tab2 = st.tabs(["Numerical Columns", "Categorical Columns"])

    with tab1:
        st.subheader("📊 Numerical Feature Summary")
        st.write(df[num_cols].describe().T)

        st.subheader("📈 Interactive Distributions")

        import plotly.express as px

        numerical_columns = [
            'age', 'duration', 'campaign', 'pdays', 'previous',
            'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
            'euribor3m', 'nr.employed'
        ]

        for col in numerical_columns:
            st.markdown(f"**{col}**")
            fig = px.histogram(df, x=col, marginal="box", nbins=30, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🧮 Categorical Feature Distributions")
        for col in cat_cols:
            st.markdown(f"**{col}**")
            vc = df[col].value_counts().reset_index()
            vc.columns = [col, 'count']
            fig = px.bar(vc, x=col, y='count',
                        labels={col: col, 'count': 'Count'},
                        title=f"Count of {col}")
            st.plotly_chart(fig, use_container_width=True)



# 🛠️ Page 3: Feature Engineering
elif page == "Feature Engineering":
    st.title("🛠️ Feature Engineering")

    st.markdown("""
    In this section, we derive new features to improve model performance.
    Feature engineering allows us to capture hidden patterns or relationships within the data.
    """)

    # Yeni özellikler
    df['multiple_contacts'] = (df['campaign'] > 1).astype(int)
    df['log_campaign'] = np.log1p(df['campaign'])
    df['log_previous'] = np.log1p(df['previous'])
    df['was_previously_contacted'] = (df['pdays'] != 999).astype(int)
    df['is_summer'] = df['month'].isin(['jun', 'jul', 'aug']).astype(int)
    df['is_winter'] = df['month'].isin(['dec', 'jan', 'feb']).astype(int)
    df['previous_category'] = df['previous'].apply(lambda x: 'none' if x == 0 else 'once' if x == 1 else 'multiple')

    # Açıklama tablosu
    st.subheader("🧾 Derived Features")
    feature_expl = {
        'multiple_contacts': "1 if client was contacted more than once in current campaign",
        'log_campaign': "Log-transformed number of contacts (handles skew)",
        'log_previous': "Log-transformed number of previous contacts",
        'was_previously_contacted': "1 if client was contacted in a previous campaign",
        'is_summer': "1 if contact month was June, July, or August",
        'is_winter': "1 if contact month was December, January, or February",
        'previous_category': "Categorical encoding of previous contact count: none / once / multiple"
    }
    st.table(pd.DataFrame.from_dict(feature_expl, orient='index', columns=['Description']))

    # Örnek veri
    st.subheader("📋 Sample with New Features")
    st.dataframe(df[[
        'campaign', 'log_campaign', 'multiple_contacts',
        'previous', 'log_previous', 'previous_category',
        'pdays', 'was_previously_contacted',
        'month', 'is_summer', 'is_winter'
    ]].head())

    # Görselleştirme
    import plotly.express as px
    st.subheader("📈 Feature Distributions")

    # log_campaign
    fig1 = px.histogram(df, x='log_campaign', nbins=30, title="Distribution of log_campaign")
    st.plotly_chart(fig1, use_container_width=True)

    # log_previous
    fig2 = px.histogram(df, x='log_previous', nbins=30, title="Distribution of log_previous")
    st.plotly_chart(fig2, use_container_width=True)

    # multiple_contacts
    vc1 = df['multiple_contacts'].value_counts().reset_index()
    vc1.columns = ['multiple_contacts', 'count']
    fig3 = px.bar(vc1, x='multiple_contacts', y='count', title="Counts of Multiple Contacts")
    st.plotly_chart(fig3, use_container_width=True)

    # previous_category
    vc2 = df['previous_category'].value_counts().reset_index()
    vc2.columns = ['previous_category', 'count']
    fig4 = px.bar(vc2, x='previous_category', y='count', title="Counts of Previous Contact Category")
    st.plotly_chart(fig4, use_container_width=True)

    # was_previously_contacted
    vc3 = df['was_previously_contacted'].value_counts().reset_index()
    vc3.columns = ['was_previously_contacted', 'count']
    fig5 = px.bar(vc3, x='was_previously_contacted', y='count', title="Counts of Was Previously Contacted")
    st.plotly_chart(fig5, use_container_width=True)

    # is_summer
    vc4 = df['is_summer'].value_counts().reset_index()
    vc4.columns = ['is_summer', 'count']
    fig6 = px.bar(vc4, x='is_summer', y='count', title="Counts of Summer Contacts")
    st.plotly_chart(fig6, use_container_width=True)

    # is_winter
    vc5 = df['is_winter'].value_counts().reset_index()
    vc5.columns = ['is_winter', 'count']
    fig7 = px.bar(vc5, x='is_winter', y='count', title="Counts of Winter Contacts")
    st.plotly_chart(fig7, use_container_width=True)


# 📈 Page 4: Model Performance
elif page == "Model Performance":
    st.title("📈 Model Performance")

    tab_labels = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"]
    model_files = ["logistic_model.pkl", "rf_model.pkl", "xgb_model.pkl", "cat_model.pkl"]

    # Load test set
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")

    tabs = st.tabs(tab_labels)
    for i, tab in enumerate(tabs):
        with tab:
            model_name = tab_labels[i]
            st.subheader(f"🔍 {model_name} Evaluation")

            try:
                # --- Load pipeline
                pipeline     = joblib.load(model_files[i])
                preprocessor = pipeline.named_steps["preprocessor"]
                clf          = pipeline.named_steps["classifier"]
                feature_names = preprocessor.get_feature_names_out()

                # --- Preprocess & predict
                X_transformed = preprocessor.transform(X_test)
                y_pred        = clf.predict(X_transformed)
                y_proba       = clf.predict_proba(X_transformed)[:, 1]

                # --- Metrics
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
                st.metric("ROC AUC",  f"{roc_auc_score(y_test, y_proba):.3f}")

                rpt_df = pd.DataFrame(
                    classification_report(
                        y_test, y_pred, output_dict=True, zero_division=0
                    )
                ).transpose()
                st.subheader("📋 Classification Report")
                st.dataframe(rpt_df)

                # --- Random Forest için interaktif feature importance
                if model_name == "Random Forest":
                    st.subheader("🔍 Random Forest Feature Importances")

                    importances = clf.feature_importances_
                    feature_names = preprocessor.get_feature_names_out()

                    # DataFrame oluşturup sıralıyoruz
                    feat_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=True)

                    # Plotly ile yatay bar chart
                    import plotly.express as px
                    fig = px.bar(
                        feat_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Feature Importances",
                        labels={'importance': 'Importance', 'feature': 'Feature'},
                        height=600
                    )
                    fig.update_layout(margin=dict(l=200, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)


                # --- Diğer modeller için SHAP
                else:
                    st.subheader("🔍 SHAP Feature Importance")

                    if model_name == "Logistic Regression":
                        explainer   = shap.LinearExplainer(clf, X_transformed, feature_names=feature_names)
                        shap_vals   = explainer.shap_values(X_transformed)
                        base_value  = explainer.expected_value
                    else:
                        explainer  = shap.TreeExplainer(clf)
                        shap_all   = explainer.shap_values(X_transformed)
                        shap_vals  = shap_all[1] if isinstance(shap_all, list) else shap_all
                        ev = explainer.expected_value
                        base_value = ev[1] if isinstance(ev, (list, np.ndarray)) else ev

                    # SHAP summary
                    fig, _ = plt.subplots()
                    shap.summary_plot(shap_vals, X_transformed,
                                      plot_type="bar",
                                      feature_names=feature_names,
                                      show=False)
                    st.pyplot(fig)

                    # SHAP waterfall (örnek 0)
                    st.subheader("🔹 SHAP Waterfall (Sample 1)")
                    expl = shap.Explanation(
                        values       = shap_vals[0],
                        base_values  = base_value,
                        data         = X_transformed[0],
                        feature_names=feature_names
                    )
                    fig2, _ = plt.subplots()
                    shap.plots.waterfall(expl, show=False)
                    st.pyplot(fig2)

                # --- Hiperparametreler (bilgi amaçlı)
                st.subheader("⚙️ Best Hyperparameters")
                params_map = {
                    "Logistic Regression": "C=0.01, penalty='l2', solver='saga', class_weight='balanced'",
                    "Random Forest":       "n_estimators=100, max_depth=None, min_samples_split=2, class_weight='balanced'",
                    "XGBoost":             "learning_rate=0.05, max_depth=6, n_estimators=200",
                    "CatBoost":            "iterations=200, depth=8, learning_rate=0.05, class_weights=[1, 8]",
                }
                st.code(params_map[model_name])

            except Exception as e:
                st.error(f"Could not load or evaluate {model_name}: {e}")




# 🤖 Page 5: Try Prediction
elif page == "Try Prediction":
    st.title("🤖 Make a Prediction")

    model = joblib.load("final_pipeline.pkl")

    st.subheader("Enter Client Information")

    age = st.slider("Age", 18, 95, 35)
    campaign = st.slider("Number of Contacts (campaign)", 1, 50, 2)
    pdays = st.slider("Days Since Last Contact (pdays)", 0, 999, 999)
    previous = st.slider("Previous Contacts", 0, 30, 0)
    emp_var_rate = st.slider("Employment Variation Rate", -3.5, 1.5, 0.0)
    cons_price_idx = st.slider("Consumer Price Index", 92.0, 95.0, 93.2)
    cons_conf_idx = st.slider("Consumer Confidence Index", -50.0, 0.0, -40.0)
    euribor3m = st.slider("Euribor 3 Month Rate", 0.5, 5.0, 2.0)

    job = st.selectbox("Job", df['job'].unique())
    marital = st.selectbox("Marital Status", df['marital'].unique())
    education = st.selectbox("Education", df['education'].unique())
    default = st.selectbox("Credit in Default?", df['default'].unique())
    housing = st.selectbox("Has Housing Loan?", df['housing'].unique())
    loan = st.selectbox("Has Personal Loan?", df['loan'].unique())
    contact = st.selectbox("Contact Type", df['contact'].unique())
    month = st.selectbox("Last Contact Month", df['month'].unique())
    day_of_week = st.selectbox("Day of Week", df['day_of_week'].unique())
    poutcome = st.selectbox("Previous Campaign Outcome", df['poutcome'].unique())

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            'age': age,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'emp.var.rate': emp_var_rate,
            'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx,
            'euribor3m': euribor3m,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'day_of_week': day_of_week,
            'poutcome': poutcome
        }])
        
        input_df['log_campaign'] = np.log1p(input_df['campaign'])
        input_df['log_previous'] = np.log1p(input_df['previous'])
        input_df['multiple_contacts'] = (input_df['campaign'] > 1).astype(int)
        input_df['was_previously_contacted'] = (input_df['pdays'] != 999).astype(int)
        input_df['previous_category'] = input_df['previous'].apply(lambda x: 'none' if x == 0 else 'once' if x == 1 else 'multiple')
        input_df['is_summer'] = input_df['month'].isin(['jun', 'jul', 'aug']).astype(int)
        input_df['is_winter'] = input_df['month'].isin(['dec', 'jan', 'feb']).astype(int)

        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.success(f"Prediction: {'Yes ✅' if prediction == 1 else 'No ❌'}")
        st.info(f"Probability of Subscription: {probability:.2%}")
