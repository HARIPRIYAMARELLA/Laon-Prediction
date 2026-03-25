A Streamlit-powered Machine Learning web application designed to predict loan-related outcomes based on financial attributes. The application supports dataset upload, data cleaning, feature engineering, model training, and performance evaluation, providing users with an interactive and user-friendly interface.
🚀 Live Demo
👉 
📌 Features
📂 Upload custom CSV datasets

🧹 Automatic data preprocessing

Missing value handling
Outlier treatment
🔄 Data transformation

Encoding categorical features
Feature scaling
🤖 Model training with multiple algorithms:

K-Nearest Neighbors (KNN)
Linear Regression
Decision Tree Regressor
📊 Model evaluation:

Mean Squared Error (MSE)
R² Score
📉 Visualization:

Boxplots
Elbow method (for KNN tuning)
🛠️ Tech Stack
Python 🐍
Streamlit
Pandas & NumPy
Scikit-learn
Matplotlib & Seaborn
📂 Project Structure
├── app.py                # Main Streamlit application
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
⚙️ Installation & Run Locally
Clone the repository:
git clone https://github.com/Saikoushik14/Loan-Prediction-ML-App
cd Loan-Prediction-ML-App
Create virtual environment (recommended):
python -m venv venv
venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app.py
📊 How It Works
Upload your dataset (CSV format)

The app automatically:

Cleans missing values
Handles outliers
Encodes categorical data
Scales numerical features
Select a machine learning model

Train and evaluate performance instantly

🎯 Use Cases
Loan risk prediction
Financial data analysis
Machine learning model comparison
Educational/demo ML projects
📈 Future Improvements
🔮 Real-time prediction form
📊 Feature importance visualization
💾 Model download option
🌐 Advanced UI enhancements
👨‍💻 Author
Sai Koushik Kasula B.Tech Data Science Student Anurag University

⭐ If you like this project
Give it a ⭐ on GitHub and share it!

