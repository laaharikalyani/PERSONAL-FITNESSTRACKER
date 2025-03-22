import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time

# --- Page Configuration ---
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

# --- Helper Functions ---
def calculate_bmi(height, weight):
    return weight / (height / 100) ** 2 if height > 0 else 0

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

def save_data(file_name, data):
    data.to_csv(file_name, index=False)

def load_data(file_name, default_columns):
    try:
        return pd.read_csv(file_name)
    except FileNotFoundError:
        return pd.DataFrame(columns=default_columns)

@st.cache_resource
def prepare_model():
    # Load datasets
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    data = exercise.merge(calories, on="User_ID")
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    
    # Handle Gender encoding
    data = pd.get_dummies(data, columns=["Gender"], drop_first=False)
    
    # Ensure the Gender_Male column exists
    if "Gender_Male" not in data.columns:
        data["Gender_Male"] = 0

    # Features and Labels
    X = data[["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Gender_Male"]]
    y = data["Calories"]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Train Model
    model = RandomForestRegressor(n_estimators=1000, max_depth=6, random_state=1)
    model.fit(X_train, y_train)
    return model, X.columns

# --- Initialize Model ---
model, feature_columns = prepare_model()

# --- Tabs ---
st.title("🏃‍♂️ Personal Fitness Tracker")
profile_tab, tracking_tab, visualize_tab = st.tabs(["📋 Profile", "📝 Daily Tracking", "📊 Progress"])

# --- Profile Tab ---
with profile_tab:
    st.header("User Profile")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col2:
        height = st.number_input("Height (cm)", min_value=100, max_value=250)
    with col3:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200)
    with col4:
        age = st.number_input("Age", min_value=15, max_value=100)
    
    if st.button("Update Profile"):
        bmi = calculate_bmi(height, weight)
        st.session_state.profile = {"gender": gender, "height": height, "weight": weight, "age": age, "bmi": bmi}
        st.success("Profile Updated!")
    
    if "profile" in st.session_state:
        profile = st.session_state.profile
        st.metric("Current BMI", f"{profile['bmi']:.2f}")
        st.info(f"Category: {get_bmi_category(profile['bmi'])}")

# --- Daily Tracking Tab ---
with tracking_tab:
    if "profile" in st.session_state:
        st.subheader("Daily Tracking Features")
        track_tabs = st.tabs(["🍎 Calorie Prediction", "🏋️ Exercise Tracking", "💧 Water Intake", "💤 Sleep Monitoring"])
        
        # Calorie Prediction
        with track_tabs[0]:
            st.subheader("Predict Calories Burned")
            
            duration = st.selectbox("Select Duration (in minutes):", [10, 20, 30, 40, 50, 60])
            heart_rate = st.selectbox("Select Heart Rate (bpm):", [60, 90, 120, 150, 180])
            body_temp = st.selectbox("Select Body Temperature (°C):", [36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0])
            
            user_data = pd.DataFrame([{
                "Age": st.session_state.profile["age"],
                "BMI": st.session_state.profile["bmi"],
                "Duration": duration,
                "Heart_Rate": heart_rate,
                "Body_Temp": body_temp,
                "Gender_Male": 1 if st.session_state.profile['gender'] == "Male" else 0
            }], columns=feature_columns)
            
            prediction = model.predict(user_data)[0]
            st.metric("Predicted Calories Burned", f"{prediction:.2f} kcal")
            
            # Save predicted data
            if st.button("Log Calories Burned"):
                calorie_file = "calorie_data.csv"
                calorie_columns = ['date', 'calories_consumed', 'calories_burned']
                
                calorie_data = load_data(calorie_file, calorie_columns)
                
                # Add new entry
                new_row = {"date": pd.Timestamp.now(), "calories_consumed": 0, "calories_burned": prediction}
                calorie_data = pd.concat([calorie_data, pd.DataFrame([new_row])], ignore_index=True)
                save_data(calorie_file, calorie_data)
                st.success("Calories burned logged successfully!")
        # Exercise Tracking
        with track_tabs[1]:
            st.subheader("Log Exercise Data")
            exercise_file = "exercise_data.csv"
            exercise_columns = ['date', 'exercise_type', 'duration']
            exercise_data = load_data(exercise_file, exercise_columns)
            
            with st.form("exercise_form"):
                exercise_date = st.date_input("Date", value=pd.Timestamp.now())
                exercise_type = st.selectbox("Exercise Type", ["Running", "Walking", "Cycling", "Swimming", "Weight Training"])
                duration = st.number_input("Duration (minutes)", min_value=0)
                submit_exercise = st.form_submit_button("Log Exercise")
            
            if submit_exercise:
                new_row = {'date': exercise_date, 'exercise_type': exercise_type, 'duration': duration}
                exercise_data = pd.concat([exercise_data, pd.DataFrame([new_row])], ignore_index=True)
                save_data(exercise_file, exercise_data)
                st.success("Exercise Data Logged Successfully!")
        
        # Water Intake Tracking
        with track_tabs[2]:
            st.subheader("Log Water Intake")
            water_file = "water_data.csv"
            water_columns = ['date', 'glasses']
            water_data = load_data(water_file, water_columns)
            
            with st.form("water_form"):
                water_date = st.date_input("Date", value=pd.Timestamp.now())
                glasses = st.number_input("Glasses of Water (250ml each)", min_value=0)
                submit_water = st.form_submit_button("Log Water Intake")
            
            if submit_water:
                new_row = {'date': water_date, 'glasses': glasses}
                water_data = pd.concat([water_data, pd.DataFrame([new_row])], ignore_index=True)
                save_data(water_file, water_data)
                st.success("Water Intake Logged Successfully!")

        
        # Sleep Monitoring
        with track_tabs[3]:
            st.subheader("Log Sleep Data")
            sleep_file = "sleep_data.csv"
            sleep_columns = ["date", "hours_slept", "sleep_quality"]
            sleep_data = load_data(sleep_file, sleep_columns)
            
            with st.form("sleep_form"):
                sleep_date = st.date_input("Date", value=pd.Timestamp.now())
                sleep_duration = st.number_input("Hours Slept", min_value=0.0, max_value=24.0, step=0.5)
                sleep_quality = st.radio("Sleep Quality (1 = Poor, 5 = Excellent):", [1, 2, 3, 4, 5])
                submit_sleep = st.form_submit_button("Log Sleep Data")
            
            if submit_sleep:
                new_row = {"date": sleep_date, "hours_slept": sleep_duration, "sleep_quality": sleep_quality}
                sleep_data = pd.concat([sleep_data, pd.DataFrame([new_row])], ignore_index=True)
                save_data(sleep_file, sleep_data)
                st.success("Sleep Data Logged Successfully!")

# --- Progress Tab ---
with visualize_tab:
    st.header("Progress Over Time")
    
    # Calorie Visualization
    calorie_file = "calorie_data.csv"
    calorie_columns = ['date', 'calories_consumed', 'calories_burned']
    calorie_data = load_data(calorie_file, calorie_columns)
    
    if not calorie_data.empty:
        calorie_data['date'] = pd.to_datetime(calorie_data['date'])
        fig = px.line(calorie_data, x='date', y=['calories_consumed', 'calories_burned'], 
                      title="Calorie Tracking Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No calorie data available. Please log your activity in Daily Tracking.")
    # Exercise Visualization
    exercise_file = "exercise_data.csv"
    exercise_columns = ['date', 'exercise_type', 'duration']
    exercise_data = load_data(exercise_file, exercise_columns)
    
    if not exercise_data.empty:
        exercise_data['date'] = pd.to_datetime(exercise_data['date'])
        fig = px.bar(exercise_data, x='date', y='duration', color='exercise_type', 
                     title="Exercise Duration Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Water Intake Visualization
    water_file = "water_data.csv"
    water_columns = ['date', 'glasses']
    water_data = load_data(water_file, water_columns)
    
    if not water_data.empty:
        water_data['date'] = pd.to_datetime(water_data['date'])
        fig = px.line(water_data, x='date', y='glasses', 
                      title="Water Intake Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No water intake data available. Please log your water intake.")
    
    # Sleep Monitoring Visualization
    sleep_file = "sleep_data.csv"
    sleep_columns = ["date", "hours_slept", "sleep_quality"]
    sleep_data = load_data(sleep_file, sleep_columns)
    
    if not sleep_data.empty:
        sleep_data['date'] = pd.to_datetime(sleep_data['date'])
        # Hours Slept Visualization
        fig_hours = px.line(sleep_data, x='date', y='hours_slept', 
                            title="Sleep Duration Over Time", markers=True, 
                            labels={"hours_slept": "Hours Slept"})
        st.plotly_chart(fig_hours, use_container_width=True)
        
        # Sleep Quality Visualization
        fig_quality = px.line(sleep_data, x='date', y='sleep_quality', 
                              title="Sleep Quality Over Time", markers=True, 
                              labels={"sleep_quality": "Quality (1-5)"})
        st.plotly_chart(fig_quality, use_container_width=True)
    else:
        st.warning("No sleep data available. Please log your sleep details in Daily Tracking.")
 