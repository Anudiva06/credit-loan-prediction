import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Credit Loan Prediction", page_icon="💰")

st.title("💰 Credit Loan Prediction")
st.write("Enter applicant details to predict loan approval.")

# -----------------------------
# Load original dataset
# -----------------------------
raw_df = pd.read_csv("loanapprove.csv")
df = raw_df.copy()

# -----------------------------
# Missing value handling
# Same approach as notebook
# -----------------------------
num_col = df.select_dtypes(include=["number"]).columns
cat_col = df.select_dtypes(include=["object"]).columns

numimp = SimpleImputer(strategy="mean")
df[num_col] = numimp.fit_transform(df[num_col])

catimp = SimpleImputer(strategy="most_frequent")
df[cat_col] = catimp.fit_transform(df[cat_col])

# -----------------------------
# Encoding
# Same approach as notebook
# -----------------------------
le_education = LabelEncoder()
df["Education_Level"] = le_education.fit_transform(df["Education_Level"])

le_target = LabelEncoder()
df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])

cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
]

ohe = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(cols),
    index=df.index
)

df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

# -----------------------------
# Train/test split + scaling
# -----------------------------
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -----------------------------
# Train Naive Bayes
# -----------------------------
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# -----------------------------
# User input
# -----------------------------
st.subheader("Applicant Details")

input_data = {}

# Numeric features
numeric_features = [
    col for col in X.columns
    if col not in encoded_df.columns and col != "Education_Level"
]

for col in numeric_features:
    input_data[col] = st.number_input(
        col.replace("_", " "),
        value=float(raw_df[col].median())
    )

# Education
education_options = sorted(raw_df["Education_Level"].dropna().unique())
input_data["Education_Level"] = st.selectbox(
    "Education Level",
    education_options
)

# Categorical features
for col in cols:
    options = sorted(raw_df[col].dropna().unique())
    input_data[col] = st.selectbox(
        col.replace("_", " "),
        options
    )

if st.button("Predict Loan Approval"):

    input_df = pd.DataFrame([input_data])

    # Impute numeric values
    numeric_input_cols = [
        col for col in num_col if col != "Loan_Approved"
    ]

    input_df[numeric_input_cols] = numimp.transform(
        input_df[numeric_input_cols]
    )

    # Encode education
    input_df["Education_Level"] = le_education.transform(
        input_df["Education_Level"]
    )

    # One-hot encode categorical values
    encoded_input = ohe.transform(input_df[cols])

    encoded_input_df = pd.DataFrame(
        encoded_input,
        columns=ohe.get_feature_names_out(cols),
        index=input_df.index
    )

    input_df = pd.concat(
        [input_df.drop(columns=cols), encoded_input_df],
        axis=1
    )

    # Keep exactly the same feature order as training
    input_df = input_df[X.columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    predicted_label = le_target.inverse_transform([prediction])[0]

    if str(predicted_label).lower() == "yes":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.write("Prediction made using the Naive Bayes model.")
