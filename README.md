# The Guide: Car Price Prediction 🚗

An advanced machine learning application that helps users find the perfect car within their budget using Random Forest regression and real-time market analysis. The system provides price predictions and personalized recommendations through an interactive web interface.

## 🔗 Live Demo
Try the application live: [The Guide on Hugging Face](https://huggingface.co/spaces/Cipher29/TheGuide)

## Features

### 🤖 Machine Learning Core
- **Random Forest Model**: Implemented for accurate car price predictions
- **Multi-factor Analysis**: Considers vehicle age, mileage, condition, and market trends
- **Price Range Estimation**: Provides minimum and maximum price estimates
- **Real-time Market Adjustment**: Adapts predictions based on current market conditions

### 💻 Interactive Web Interface
- **Streamlit Dashboard**: User-friendly interface for input and visualization
- **Dynamic Filtering**: Real-time model and make selection
- **Price Trend Visualization**: Interactive graphs showing historical price trends
- **AI Assistant**: Integrated GPT-powered chatbot for personalized recommendations

### 📊 Data Analysis
- **Market Trends**: Visual representation of depreciation curves
- **Brand Categories**: Hierarchical classification of car brands
- **Comprehensive Metrics**: Including mileage analysis and condition impact

## Technology Stack

- **Python 3.8+**
- **Machine Learning**: scikit-learn (Random Forest)
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **AI Integration**: OpenAI GPT-3.5
  
## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mesa112/TheGuide.git
cd TheGuide
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage

1. **Access the Live Application**:
   - Visit [https://huggingface.co/spaces/Cipher29/TheGuide](https://huggingface.co/spaces/Cipher29/TheGuide)
   - No installation required for the web version

2. **Local Installation** (if you want to run it locally):
   ```bash
   git clone https://github.com/Mesa112/TheGuide.git
   cd TheGuide
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. Enter vehicle details:
   - Select make and model
   - Input year and condition
   - Specify mileage and other features

4. Get predictions and recommendations:
   - View estimated price range
   - Explore market trends
   - Chat with AI assistant for personalized advice

## Project Structure
```
├── app.py                # Main Streamlit application
├── model/
│   ├── random_forest.pkl # Trained Random Forest model
│   └── encoders.pkl     # Label encoders for categorical variables
├── data/
│   └── car_data.csv     # Training dataset
├── utils/
│   ├── preprocessing.py # Data preprocessing functions
│   └── visualization.py # Plotting and visualization tools
└── requirements.txt     # Project dependencies
```
## Model Details

### Random Forest Implementation
- **Features**: Year, Make, Model, Condition, Mileage, etc.
- **Target Variable**: Car Price
- **Feature Engineering**:
  - Age calculation
  - Mileage per year
  - Brand category encoding
  - Condition impact factors

### Brand Categories Implementation
Implemented price boundaries for different brand categories to ensure realistic predictions:

| Category | Example Brands | Price Range |
|----------|---------------|-------------|
| Luxury | Rolls-Royce, Bentley | $200,000 - $600,000 |
| Premium | BMW, Mercedes | $35,000 - $150,000 |
| Mid-tier | Acura, Lincoln | $25,000 - $65,000 |
| Standard | Toyota, Honda | $17,000 - $45,000 |
| Economy | Mitsubishi | $12,000 - $35,000 |

### Price Prediction Process
1. Data preprocessing and normalization
2. Feature extraction and engineering
3. Random Forest prediction
4. Market adjustment factors application
5. Price range calculation with brand-specific boundaries

## Performance Metrics

### Model Accuracy
- **RMSE (Root Mean Square Error)**: 0.83
- **Average Prediction Variance**: ±$4,000
- **Response Time**: Real-time predictions (<2s)

### Validation Results
- Predictions consistently align with market values
- Real-time adjustments based on:
  - Vehicle condition
  - Mileage impact
  - Market trends
  - Seasonal variations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Contact

Esteban Mesa - [estebanmesa57@gmail.com](mailto:estebanmesa57@gmail.com)

Project Link: [https://github.com/Mesa112/TheGuide](https://github.com/Mesa112/TheGuide)
