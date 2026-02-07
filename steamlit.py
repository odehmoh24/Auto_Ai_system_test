import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title("Auto AI system")


df = None
file_name = ""
model_choise_AI = ""
issupervise = ""
type_of_task = ""
type_Oftask_ai = ""
suggested_model = ""
potential_targets = []
list_of_unique_ratio = []
num_classes = 0
num_numeric = 0
num_categorical = 0
non_linear = False
target = None
outlier_report = {}

Data = st.file_uploader("Upload your data")
if Data:
    file_name = Data.name
    if file_name.endswith(".csv"):
        df = pd.read_csv(Data)
        st.dataframe(df.head(5))
    elif file_name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(Data)
        st.dataframe(df.head(5))
    elif file_name.endswith(".json"):
        df = pd.read_json(Data)
        st.dataframe(df.head(5))
    elif file_name.endswith(".txt"):
        content = Data.read().decode("utf-8")
        st.text(content)

    if df is not None:
        num_numeric = df.select_dtypes(include='number').shape[1]
        num_categorical = df.select_dtypes(include='object').shape[1]

analysis_button=st.button("analyz my Data")


with st.expander("Show Data Analysis"):
 if analysis_button :
#Missing_value
    st.markdown(" Missing Values")    
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
            st.success("No missing values ")
    else:
            st.write(missing)
            st.warning("Ther is missing value")

    
 #Outlier           
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        outliers_sum = outliers[col].sum
        if not outliers.empty:
                outlier_report[col] = len(outliers)
    st.markdown("Outlier")
    
    if not outlier_report:
         st.success("No significant outliers detected ")
    else:
         st.dataframe(pd.DataFrame.from_dict(outlier_report, orient="index", columns=["Outlier Count"]))
         st.warning("Recommendation: Scaling / Clipping / Log Transform")
         
      
 #imbalance
    st.markdown("Categorical Imbalance")

    cat_cols = df.select_dtypes(include="object").columns

    if len(cat_cols) == 0:
            st.info("No categorical columns detected")
    else:
            for col in cat_cols:
                counts = df[col].value_counts(normalize=True)
                max_ratio = counts.max()

                st.write(f" {col}")
                st.dataframe(counts)

                if max_ratio > 0.75:
                    st.error("Imbalanced Column")
                    st.warning(" Recommendation: Use SMOTE, class weights, or undersampling")
                else:
                    st.success("Balanced ✔")
  
#dist num

   
    st.markdown(" Boxplot: Numeric Columns Distribution")
    num_cols = df.select_dtypes(include="number").columns

    plt.figure(figsize=(10,6))
    plt.boxplot([df[col].dropna() for col in num_cols], labels=num_cols)
    plt.title("Boxplot of All Numeric Columns")
    plt.ylabel("Value")
    st.pyplot(plt)





model_choise = st.selectbox("Choose your AI model", [" ", "Machine", "Deep learning", "NLP", "let the AI choise"])


if model_choise == "let the AI choise" and df is not None:
    if file_name.endswith(".csv") or df.shape[0] < 100000:
        model_choise_AI = "ML"
        st.write("The AI recommends Machine Learning (ML)")


if (model_choise == "Machine" or model_choise_AI == "ML") and df is not None:

    superviseML = st.selectbox("Is the data supervised?", [" ", "supervise", "unsupervise", "let the AI choise"])

    if superviseML in ["supervise", "unsupervise"]:
        issupervise = superviseML
        if issupervise == "supervise":
            target = st.selectbox("Choose your label column", df.columns)
            num_classes = df[target].nunique()
    elif superviseML == "let the AI choise":
        potential_targets = []
        list_of_unique_ratio = []
      
        for col in df.columns:
            nunique_ratio = df[col].nunique() / df.shape[0]
            if nunique_ratio < 0.1 and df[col].dtype != "object":
                potential_targets.append(col)
                list_of_unique_ratio.append(nunique_ratio)

        if potential_targets:
            target_index = list_of_unique_ratio.index(min(list_of_unique_ratio))
            target = potential_targets[target_index]
            st.write(f"Suggested label column: {target}")
            num_classes = df[target].nunique()
            missing_ratio = df[target].isna().mean()
            if missing_ratio < 0.25:
                issupervise = "Supervised Learning"
            else:
                issupervise = "UnSupervised Learning"
        else:
            issupervise = "UnSupervised Learning"

    st.write(f"Learning type: {issupervise}")

    if issupervise:
        type_of_task = st.selectbox("Choose your type of task", ["", "let the AI choise", "Classification", "Regression"])

    if type_of_task == "let the AI choise" and target is not None:
        
        if num_classes <= 15 and df[target].dtype == "object":
            type_Oftask_ai = "Classification"
        elif num_classes <= 10 and df[target].dtype in ["int64", "float64"]:
            
            type_Oftask_ai = "Classification"
        else:
            type_Oftask_ai = "Regression"

    if type_of_task in ["Classification"] or type_Oftask_ai == "Classification":
       
        classification_model_name = st.selectbox(
            "Choose your classification model",
            ["", "Logistic Regression", "SVM", "Decision Tree",
             "Random Forest", "Gradient Boosting", "LightGBM",
             "all them", "let AI choise"]
        )

        if classification_model_name == "let AI choise":
            
            non_linear = True if num_numeric > 1 else False
            if df.shape[0] < 10000:
                if num_classes <= 2:
                    if num_categorical == 0:
                        suggested_model = "Logistic Regression"
                    elif non_linear:
                        suggested_model = "SVM"
                    else:
                        suggested_model = "Decision Tree"
                elif num_classes <= 5:
                    if num_categorical > 0 and num_numeric > 0:
                        suggested_model = "Decision Tree"
                    else:
                        suggested_model = "SVM"
                else:
                    suggested_model = "Decision Tree"
            elif df.shape[0] < 500000:
                suggested_model = "Random Forest" if non_linear else "Gradient Boosting"
            else:
                suggested_model = "LightGBM"

            st.write(f"Suggested model: {suggested_model}")

                   
  









