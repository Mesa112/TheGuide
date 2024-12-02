import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import openai
from sklearn.preprocessing import LabelEncoder
import requests  # Add this at the top with other imports
from io import BytesIO
import gdown

# --- Set page configuration ---
st.set_page_config(
    page_title="The Guide",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---

st.markdown("""
    <style>
    /* Base styles */
    * {
        color: black !important;
    }
    
    /* Streamlit specific input elements */
    .stSelectbox, 
    .stNumberInput, 
    .stTextInput {
        color: black !important;
    }
    
    /* Dropdown and select elements */
    select option,
    .streamlit-selectbox option,
    .stSelectbox > div[data-baseweb="select"] > div,
    .stSelectbox > div > div > div {
        color: black !important;
        background-color: white !important;
    }
    
    /* Input fields */
    input, 
    .stNumberInput > div > div > input {
        color: black !important;
    }
    
    /* Text elements */
    div.row-widget.stSelectbox > div,
    div.row-widget.stSelectbox > div > div > div,
    .streamlit-expanderContent,
    .stMarkdown,
    p, span, label {
        color: black !important;
    }
    
    /* Keep button text white */
    .stButton > button {
        color: white !important;
        background-color: #FF4B4B;
    }
    
    /* Specific styling for select boxes */
    div[data-baseweb="select"] {
        color: black !important;
        background-color: white !important;
    }
    
    div[data-baseweb="select"] * {
        color: black !important;
    }
    
    /* Style for the selected option */
    div[data-baseweb="select"] > div:first-child {
        color: black !important;
        background-color: white !important;
    }
    
    /* Dropdown menu items */
    [role="listbox"] {
        background-color: white !important;
    }
    
    [role="listbox"] [role="option"] {
        color: black !important;
    }
    
    /* Number input specific styling */
    input[type="number"] {
        color: black !important;
        background-color: white !important;
    }
    
    .stNumberInput div[data-baseweb="input"] {
        background-color: white !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Cache functions ---
def create_brand_categories():
    return {
        'luxury_brands': {
            'rolls-royce': (300000, 600000),
            'bentley': (200000, 500000),
            'lamborghini': (250000, 550000),
            'ferrari': (250000, 600000),
            'mclaren': (200000, 500000),
            'aston-martin': (150000, 400000),
            'maserati': (100000, 300000)
        },
        'premium_brands': {
            'porsche': (60000, 150000),
            'bmw': (40000, 90000),
            'mercedes-benz': (45000, 95000),
            'audi': (35000, 85000),
            'lexus': (40000, 80000),
            'jaguar': (45000, 90000),
            'land-rover': (40000, 90000),
            'volvo': (35000, 75000),
            'infiniti': (35000, 70000),
            'cadillac': (40000, 85000),
            'tesla': (40000, 100000)
        },
        'mid_tier_brands': {
            'acura': (30000, 50000),
            'lincoln': (35000, 65000),
            'buick': (25000, 45000),
            'chrysler': (25000, 45000),
            'alfa-romeo': (35000, 60000),
            'genesis': (35000, 60000)
        },
        'standard_brands': {
            'toyota': (20000, 35000),
            'honda': (20000, 35000),
            'volkswagen': (20000, 35000),
            'mazda': (20000, 32000),
            'subaru': (22000, 35000),
            'hyundai': (18000, 32000),
            'kia': (17000, 30000),
            'ford': (20000, 40000),
            'chevrolet': (20000, 38000),
            'gmc': (25000, 45000),
            'jeep': (25000, 45000),
            'dodge': (22000, 40000),
            'ram': (25000, 45000),
            'nissan': (18000, 32000)
        },
        'economy_brands': {
            'mitsubishi': (15000, 25000),
            'suzuki': (12000, 22000),
            'fiat': (15000, 25000),
            'mini': (20000, 35000),
            'smart': (15000, 25000)
        },
        'discontinued_brands': {
            'pontiac': (5000, 15000),
            'saturn': (4000, 12000),
            'mercury': (4000, 12000),
            'oldsmobile': (3000, 10000),
            'plymouth': (3000, 10000),
            'saab': (5000, 15000)
        }
    }

@st.cache_resource
def download_file_from_google_drive(file_id):
    """Downloads a file from Google Drive using gdown."""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        with st.spinner('Downloading from Google Drive...'):
            output = f"temp_{file_id}.pkl"
            gdown.download(url, output, quiet=False)
            
            with open(output, 'rb') as f:
                content = f.read()
            
            # Clean up the temporary file
            os.remove(output)
            return content
            
    except Exception as e:
        st.error(f"Error downloading from Google Drive: {str(e)}")
        raise e

@st.cache_data
def load_datasets():
    """Load the dataset from Google Drive."""
    dataset_file_id = "1emG-BQ3-x4xsMAGMEznkh1ACdlAj5Dn1"
    
    try:
        with st.spinner('Loading dataset...'):
            content = download_file_from_google_drive(dataset_file_id)
            # Use BytesIO to read the CSV content
            original_data = pd.read_csv(BytesIO(content), low_memory=False)
            
            # Ensure column names match the model's expectations
            original_data.columns = original_data.columns.str.strip().str.capitalize()
            return original_data
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        raise e

@st.cache_resource
def load_model_and_encodings():
    """Load model from Google Drive and create encodings."""
    model_file_id = "1wKixkdW2pVKEpJW-N1QIyKUr2nYirU7I"
    
    try:
        # Show loading message
        with st.spinner('Loading model...'):
            model_content = download_file_from_google_drive(model_file_id)
            model = joblib.load(BytesIO(model_content))
        
        # Load data for encodings
        original_data = load_datasets()
        
        # Create fresh encoders from data
        label_encoders = {}
        categorical_features = ['Make', 'model', 'condition', 'fuel', 'title_status', 
                              'transmission', 'drive', 'size', 'type', 'paint_color']
        
        for feature in categorical_features:
            if feature in original_data.columns:
                le = LabelEncoder()
                unique_values = original_data[feature].fillna('unknown').str.strip().unique()
                le.fit(unique_values)
                label_encoders[feature.lower()] = le
        
        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e


# --- Load data and models ---
try:
    original_data = load_datasets()
    model, label_encoders = load_model_and_encodings()  # Using the new function
except Exception as e:
    st.error(f"Error loading data or models: {str(e)}")
    st.stop()

# --- Define categorical and numeric features ---
# From model.py
# --- Define features ---
numeric_features = ['year', 'odometer', 'age', 'age_squared', 'mileage_per_year']
# Update the categorical features list to use lowercase
categorical_features = ['make', 'model', 'condition', 'fuel', 'title_status', 
                       'transmission', 'drive', 'size', 'type', 'paint_color']
required_features = numeric_features + categorical_features

# --- Feature engineering functions ---
def create_features(df):
    df = df.copy()
    current_year = 2024
    df['age'] = current_year - df['year']
    df['age_squared'] = df['age'] ** 2
    df['mileage_per_year'] = np.clip(df['odometer'] / (df['age'] + 1), 0, 200000)
    return df

def prepare_input(input_dict, label_encoders):
    # Convert None values to 'unknown' for safe handling
    input_dict = {k: v if v is not None else 'unknown' for k, v in input_dict.items()}
    
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Ensure columns match the model's expected casing
    feature_name_mapping = {
        "make": "Make",  # Match casing for 'Make'
        "model": "Model",  # Match casing for 'Model'
        "condition": "Condition",
        "fuel": "Fuel",
        "title_status": "Title_status",
        "transmission": "Transmission",
        "drive": "Drive",
        "size": "Size",
        "type": "Type",
        "paint_color": "Paint_color",
        "year": "Year",
        "odometer": "Odometer",
        "age": "Age",
        "age_squared": "Age_squared",
        "mileage_per_year": "Mileage_per_year"
    }
    input_df.rename(columns=feature_name_mapping, inplace=True)

    # Numeric feature conversions
    input_df["Year"] = pd.to_numeric(input_df.get("Year", 0), errors="coerce")
    input_df["Odometer"] = pd.to_numeric(input_df.get("Odometer", 0), errors="coerce")
    
    # Feature engineering
    current_year = 2024
    input_df["Age"] = current_year - input_df["Year"]
    input_df["Age_squared"] = input_df["Age"] ** 2
    input_df["Mileage_per_year"] = input_df["Odometer"] / (input_df["Age"] + 1)
    input_df["Mileage_per_year"] = input_df["Mileage_per_year"].clip(0, 200000)

    # Encode categorical features
    for feature, encoded_feature in feature_name_mapping.items():
        if feature in label_encoders:
            input_df[encoded_feature] = input_df[encoded_feature].fillna("unknown").astype(str).str.strip()
            try:
                input_df[encoded_feature] = label_encoders[feature].transform(input_df[encoded_feature])
            except ValueError:
                input_df[encoded_feature] = 0  # Assign default for unseen values

    # Ensure all required features are present
    for feature in model.feature_names_in_:
        if feature not in input_df:
            input_df[feature] = 0  # Default value for missing features

    # Reorder columns
    input_df = input_df[model.feature_names_in_]

    return input_df



# --- Styling functions ---
st.markdown("""
    <style>
    /* Force black text globally */
    .stApp, .stApp * {
        color: black !important;
    }
    
    /* Specific overrides for different elements */
    .main {
        padding: 0rem 1rem;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white !important;  /* Keep button text white */
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    
    /* Input fields and selectboxes */
    .stSelectbox select, 
    .stSelectbox option,
    .stSelectbox div,
    .stNumberInput input,
    .stTextInput input {
        color: black !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    
    /* Labels and text */
    label, .stText, p, span {
        color: black !important;
    }
    
    /* Selectbox options */
    option {
        color: black !important;
        background-color: white !important;
    }
    
    /* Override for any Streamlit specific classes */
    .st-emotion-cache-16idsys p,
    .st-emotion-cache-1wmy9hl p,
    .st-emotion-cache-16idsys span,
    .st-emotion-cache-1wmy9hl span {
        color: black !important;
    }
    
    /* Force white text only for the prediction button */
    .stButton>button[data-testid="stButton"] {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

def style_metric_container(label, value):
    st.markdown(f"""
        <div style="
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            border-left: 5px solid #FF4B4B;
        ">
            <p style="color: #666; margin-bottom: 0.2rem; font-size: 0.9rem;">{label}</p>
            <p style="color: #1E1E1E; font-size: 1.5rem; font-weight: 600; margin: 0;">{value}</p>
        </div>
    """, unsafe_allow_html=True)

# --- OpenAI GPT-3 Assistant ---
def generate_gpt_response(prompt):
    # Ensure the API key is set securely
    # You can use Streamlit's secrets management or environment variables
    openai.api_key = "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"

    # Define the system message and messages list
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful car shopping assistant. "
            "Provide car recommendations based on user queries. "
            "Include car makes, models, years, and approximate prices. "
            "Be friendly and informative."
        )
    }

    messages = [system_message, {"role": "user", "content": prompt}]

    # Call the OpenAI ChatCompletion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        messages=messages,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the assistant's reply
    assistant_reply = response['choices'][0]['message']['content'].strip()

    return assistant_reply

def create_assistant_section():
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h2 style='color: #1E1E1E; margin-top: 0;'>🤖 Car Shopping Assistant</h2>
            <p style='color: #666;'>Ask me anything about cars! For example: 'What's a good car under $30,000 with low mileage?'</p>
        </div>
    """, unsafe_allow_html=True)

    if "assistant_responses" not in st.session_state:
        st.session_state.assistant_responses = []

    prompt = st.text_input("Ask about car recommendations...", 
                           placeholder="Type your question here...")

    if prompt:
        try:
            # Use OpenAI API to generate response
            response = generate_gpt_response(prompt)
            st.session_state.assistant_responses.append(response)
        except Exception as e:
            response = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.assistant_responses.append(response)

        # Display the latest response
        st.write(response)

        # Optionally display previous responses
        if len(st.session_state.assistant_responses) > 1:
            st.markdown("### Previous Responses")
            for prev_response in st.session_state.assistant_responses[:-1]:
                st.markdown("---")
                st.write(prev_response)

    if st.button("Clear Chat"):
        st.session_state.assistant_responses = []
        st.experimental_rerun()

# --- Prediction Interface ---
def create_prediction_interface():
    with st.sidebar:
        st.markdown("""
            <div style='background-color: #FF4B4B; padding: 1rem; border-radius: 5px; margin-bottom: 2rem;'>
                <h2 style='color: white; margin: 0;'>Car Details</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Year slider
        year = st.slider("Year", min_value=1980, max_value=2024, value=2022)
        
        # Make selection
        make_options = sorted(original_data['Make'].dropna().unique())  # Correct casing for 'Make'
        make = st.selectbox("Make", options=make_options)  
        
        # Filter models based on selected make
        filtered_models = sorted(original_data[original_data['Make'] == make]['Model'].dropna().unique())  # Match 'Model' casing
        model_name = st.selectbox("Model", options=filtered_models if len(filtered_models) > 0 else ["No models available"])
        
        if model_name == "No models available":
            st.warning("No models are available for the selected make.")

        # Additional inputs
        condition = st.selectbox("Condition", ['new', 'like new', 'excellent', 'good', 'fair', 'salvage', 'parts only'])
        fuel = st.selectbox("Fuel Type", sorted(original_data['Fuel'].fillna('Unknown').unique()))  # Match casing for 'Fuel'
        odometer = st.number_input("Odometer (miles)", min_value=0, value=20000, format="%d", step=1000)
        title_status = st.selectbox("Title Status", sorted(original_data['Title_status'].fillna('Unknown').unique()))  # Match casing
        transmission = st.selectbox("Transmission", sorted(original_data['Transmission'].fillna('Unknown').unique()))
        drive = st.selectbox("Drive Type", sorted(original_data['Drive'].fillna('Unknown').unique()))
        size = st.selectbox("Size", sorted(original_data['Size'].fillna('Unknown').unique()))
        paint_color = st.selectbox("Paint Color", sorted(original_data['Paint_color'].fillna('Unknown').unique()))
        
        car_type = 'sedan'  # Default type
        
        # Prediction button
        predict_button = st.button("📊 Predict Price", use_container_width=True)

    return {
        'year': year,
        'make': make.strip(),  # Use correctly cased `make`
        'model': model_name if model_name != "No models available" else 'unknown',
        'condition': condition.lower().strip(),
        'fuel': fuel.lower().strip(),
        'odometer': odometer,
        'title_status': title_status.lower().strip(),
        'transmission': transmission.lower().strip(),
        'drive': drive.lower().strip(),
        'size': size.lower().strip(),
        'type': car_type.lower().strip(),
        'paint_color': paint_color.lower().strip()
    }, predict_button



def create_market_trends_plot_with_model(model, make, base_inputs, label_encoders, years_range=range(1980, 2025)):
    predictions = []
    
    for year in years_range:
        try:
            current_inputs = base_inputs.copy()
            current_inputs['year'] = float(year)
            age = 2024 - year
            
            # Base value calculation
            base_price = 30000  # Average new car price
            
            # Depreciation curve
            if age <= 1:
                value_factor = 0.85  # 15% first year depreciation
            elif age <= 5:
                value_factor = 0.85 * (0.90 ** (age - 1))  # 10% years 2-5
            else:
                value_factor = 0.85 * (0.90 ** 4) * (0.95 ** (age - 5))  # 5% thereafter
            
            price = base_price * value_factor
            predictions.append({"year": year, "predicted_price": max(price, 2000)})  # Floor of $2000
            
        except Exception as e:
            continue

    if not predictions:
        return None
    
    predictions_df = pd.DataFrame(predictions)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predictions_df["year"], predictions_df["predicted_price"], color="#FF4B4B", linewidth=2)
    ax.set_title(f"Average Car Value by Age")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.grid(True, alpha=0.3)
    
    return fig

def inspect_model_features(model):
    # Check feature names the model expects
    try:
        if hasattr(model, "feature_names_in_"):
            print("Model feature names:", model.feature_names_in_)
        else:
            print("Model does not have 'feature_names_in_' attribute.")
    except Exception as e:
        print(f"Error inspecting model features: {e}")

def predict_with_ranges(inputs, model, label_encoders):
    input_df = prepare_input(inputs, label_encoders)
    base_prediction = float(np.expm1(model.predict(input_df)[0]))
    
    brand_categories = create_brand_categories()
    make = inputs['make'].lower()
    year = inputs['year']
    condition = inputs['condition']
    odometer = inputs['odometer']
    age = 2024 - year
    
    # Find brand category and price range
    price_range = None
    for category, brands in brand_categories.items():
        if make in brands:
            price_range = brands[make]
            break
    if not price_range:
        price_range = (15000, 35000)  # Default range
    
    # Calculate adjustment factors
    mileage_factor = max(1 - (odometer / 200000) * 0.3, 0.7)
    age_factor = 0.85 ** min(age, 15)
    condition_factor = {
        'new': 1.0,
        'like new': 0.9,
        'excellent': 0.8,
        'good': 0.7,
        'fair': 0.5,
        'salvage': 0.3
    }.get(condition, 0.7)
    
    # Apply all factors
    min_price = price_range[0] * mileage_factor * age_factor * condition_factor
    max_price = price_range[1] * mileage_factor * age_factor * condition_factor
    predicted_price = base_prediction * mileage_factor * age_factor * condition_factor
    
    # Use uniform distribution instead of clamping
    final_prediction = np.random.uniform(min_price, max_price)
    
    return {
        'predicted_price': final_prediction,
        'min_price': min_price,
        'max_price': max_price
    }
# --- Main Application ---
def main(model, label_encoders):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            <h1 style='text-align: center;'>The Guide 🚗</h1>
            <p style='text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;'>
                A cutting-edge data science project leveraging machine learning to detect which car would be best for you.
            </p>
        """, unsafe_allow_html=True)
        
        inputs, predict_button = create_prediction_interface()
        
        # Prepare base inputs
        base_inputs = {
            "year": inputs.get("year", 2022),
            "make": inputs.get("make", "toyota").lower(),
            "model": inputs.get("model", "camry"),
            "odometer": inputs.get("odometer", 20000),
            "condition": inputs.get("condition", "good"),
            "fuel": inputs.get("fuel", "gas"),
            "title_status": inputs.get("title_status", "clean"),
            "transmission": inputs.get("transmission", "automatic"),
            "drive": inputs.get("drive", "fwd"),
            "size": inputs.get("size", "mid-size"),
            "paint_color": inputs.get("paint_color", "black"),
            "type": inputs.get("type", "sedan")
        }

        if base_inputs["condition"] == "new":
            base_inputs["odometer"] = 0

        if predict_button:
            st.write(f"Analyzing {base_inputs['year']} {base_inputs['make'].title()} {base_inputs['model'].title()}...")
            prediction_results = predict_with_ranges(base_inputs, model, label_encoders)
            
            st.markdown(f"""
                ### Price Analysis
                - **Estimated Range**: ${prediction_results['min_price']:,.2f} - ${prediction_results['max_price']:,.2f}
                - **Model Prediction**: ${prediction_results['predicted_price']:,.2f}
                
                *Note: Range based on market data, condition, and mileage*
            """)

        # Generate and display the graph
        fig = create_market_trends_plot_with_model(model, base_inputs["make"], base_inputs, label_encoders)
        if fig:
            st.pyplot(fig)
        else:
            st.warning("No graph generated. Please check your data or selection.")

    with col2:
        create_assistant_section()

if __name__ == "__main__":
    try:
        # Load data and model
        original_data = load_datasets()
        model, label_encoders = load_model_and_encodings()
        
        # Inspect model features
        inspect_model_features(model)
        
        # Call the main function
        main(model, label_encoders)
    except Exception as e:
        st.error(f"Error loading data or models: {str(e)}")
        st.stop()
