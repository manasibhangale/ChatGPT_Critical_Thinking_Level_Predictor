# ChatGPT Critical Thinking Level Predictor

This project predicts a user's **critical thinking aptitude level** based on their academic and behavioral traits using a machine learning model. It classifies users into **Low**, **Medium**, or **High Aptitude Buckets**, provides personalized feedback, and generates a detailed PDF report with graphs.

🚀 Features

- Collects user inputs via a user-friendly Streamlit web interface.
- Predicts aptitude bucket using a trained machine learning model.
- Displays prediction probabilities with a bar chart.
- Generates a downloadable, styled PDF report including:
  - User details
  - Prediction results
  - Bar graph of prediction probabilities
  - Personalized tips for improvement

🧠 Input Features

| Feature                        | Type     | Range / Options                                      |
|-------------------------------|----------|------------------------------------------------------|
| `CGPA`                        | Float    | 0.0 – 10.0                                           |
| `ChatGPT Usage Frequency`     | Integer  | 0 – 100 times per week                               |
| `Average Session Duration`    | Integer  | 0 – 180 minutes                                      |
| `Department`                  | Categorical | CSE / ECE / Mechanical / Civil / Other            |
| `Reason for Using ChatGPT`    | Categorical | Learning / Homework Help / Coding Assistance / Casual Chat / Other |

 📄 PDF Report Includes

- User's Name and Predicted Aptitude Level
- Prediction Probabilities (in %)
- Colorful bar graph (matplotlib)
- Tailored tips for the user's level

 🛠️ Technologies Used

- **Python**
- **Pandas** for data handling
- **Streamlit** for UI
- **Scikit-learn** for ML model
- **Joblib** for model loading
- **Matplotlib & Seaborn** for visualizations
- **ReportLab** for PDF generation

▶️ How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

3. Fill in the details and click **Predict Aptitude Bucket**.

4. View results and download the PDF report.
5. ## 🚀 Live Demo

🔗 **Try the App Now**:  
[ChatGPT Critical Thinking Level Predictor](https://chatgptcriticalthinkinglevelpredictor-hpnd9rz6gjafrxqxedzzze.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatgptcriticalthinkinglevelpredictor-hpnd9rz6gjafrxqxedzzze.streamlit.app/)



