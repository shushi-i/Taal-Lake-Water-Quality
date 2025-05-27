import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
from streamlit_option_menu import option_menu
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, TimeDistributed

# Enhanced UI Configuration
st.set_page_config(
    page_title="Taal Lake Water Quality Dashboard",
    page_icon="taal logo.png",  # Use local file as favicon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Improved Container Styling */
    .stApp {
        background-color: #f0f6fc;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Enhancements */
    .css-1aumxhk {
        background-color: white;
        border-right: 1px solid #e0e4e8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Card-like Containers */
    .stContainer, .stDataFrame {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Colorful Metric Boxes */
    .metric-container {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Buttons and Inputs */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }

    /* Plotly Chart Improvements */
    .plotly-chart {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_excel("Water quality.xlsx")

df = load_data()



# Sidebar Navigation
with st.sidebar:
    st.image("taal logo.png", width=250)
    st.markdown("## 💧 Taal Lake Monitoring")

    selected = option_menu(
        menu_title=None,
        options=["Overview", "Time Series", "Predictions", "Developer Info"],
        icons=["house-fill", "graph-up", "magic", "info-circle"],
        default_index=0,
        styles={
            "container": {
                "padding": "10px",
                "background-color": "transparent"
            },
            "icon": {"color": "#2563EB", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "transition": "all 0.3s",
                "border-radius": "10px"
            },
            "nav-link-selected": {
                "background-color": "#2563EB",
                "color": "white"
            },
        }
    )

    # Date Range Filter
if not df.empty:
    date_col = next((col for col in df.columns if "date" in col.lower() or "time" in col.lower()), None)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()

        with st.sidebar:
            st.markdown("### 📅 Date Range")
            use_full_range = st.checkbox("Use Full Date Range", value=True)

            if not use_full_range:
                date_range = st.date_input(
                    "Select Date Range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date

            # Filter the dataframe
            mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))
            df = df.loc[mask]

# Overview Page
if selected == "Overview":
    # Display the banner image (above the title)
    st.image("taal_banner.png", use_container_width=True)

    # Title
    st.title("📊 Water Quality Overview")

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>Total Records</h3>
            <h1>{}</h1>
        </div>
        """.format(len(df)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>Parameters</h3>
            <h1>{}</h1>
        </div>
        """.format(len(df.columns) - 1), unsafe_allow_html=True)

    with col3:
        if date_col:
            start_date = df[date_col].min().strftime('%Y-%m-%d')
        else:
            start_date = "N/A"

        st.markdown("""
        <div class="metric-container">
            <h3>Start Date</h3>
            <h1>{}</h1>
        </div>
        """.format(start_date), unsafe_allow_html=True)

    with col4:
        if date_col:
            end_date = df[date_col].max().strftime('%Y-%m-%d')
        else:
            end_date = "N/A"

        st.markdown("""
        <div class="metric-container">
            <h3>End Date</h3>
            <h1>{}</h1>
        </div>
        """.format(end_date), unsafe_allow_html=True)

    # Raw Data Display with options
    st.subheader("📋 Complete Raw Data")

    data_view_options = st.radio(
        "Data Display Options:",
        ["View All Data", "Filter by Column", "Search Records"],
        horizontal=True
    )

    if data_view_options == "View All Data":
        st.dataframe(df, use_container_width=True, height=500)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Data as CSV",
            data=csv,
            file_name="taal_lake_water_quality_data.csv",
            mime="text/csv",
        )

    elif data_view_options == "Filter by Column":
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select columns to display:", all_columns, default=all_columns[:5])

        if selected_columns:
            st.dataframe(df[selected_columns], use_container_width=True, height=500)
        else:
            st.info("Please select at least one column to display data.")

    elif data_view_options == "Search Records":
        search_col = st.selectbox("Select column to search:", df.columns.tolist())

        if df[search_col].dtype == 'object':
            search_term = st.text_input("Enter search term:")
            if search_term:
                filtered_df = df[df[search_col].astype(str).str.contains(search_term, case=False)]
                st.dataframe(filtered_df, use_container_width=True, height=500)
                st.write(f"Found {len(filtered_df)} matching records")
        else:
            min_val = float(df[search_col].min())
            max_val = float(df[search_col].max())
            range_val = st.slider(f"Select range for {search_col}:",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val))
            filtered_df = df[(df[search_col] >= range_val[0]) & (df[search_col] <= range_val[1])]
            st.dataframe(filtered_df, use_container_width=True, height=500)
            st.write(f"Found {len(filtered_df)} matching records")


# Time Series
elif selected == "Time Series":
    st.image("taal_banner.png", use_container_width=True)
    st.title("📈 Time Series Analysis")

    if not df.empty:
        time_col = next((col for col in df.columns if "date" in col.lower() or "time" in col.lower()), None)

        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])

            # Extract year and month from time_col for filtering
            df['Year'] = df[time_col].dt.year
            df['Month'] = df[time_col].dt.month

            # Apply filters based on the sidebar date selection
            df_filtered = df.copy()  # Keep the filtered data separately

            # Prepare numeric columns excluding Year and Month
            numeric_cols = df_filtered.select_dtypes(include='number').columns.tolist()
            if 'Year' in numeric_cols:
                numeric_cols.remove('Year')
            if 'Month' in numeric_cols:
                numeric_cols.remove('Month')

            # Initialize session state
            if "time_series_selected_params" not in st.session_state:
                st.session_state.time_series_selected_params = []

            if "select_all_clicked" not in st.session_state:
                st.session_state.select_all_clicked = False

            st.markdown("#### Select Parameters to Plot")

            # Select All checkbox
            select_all = st.checkbox("Select All Parameters")

            # If Select All is clicked, update session state once
            if select_all and not st.session_state.select_all_clicked:
                st.session_state.time_series_selected_params = numeric_cols
                st.session_state.select_all_clicked = True
            elif not select_all and st.session_state.select_all_clicked:
                st.session_state.time_series_selected_params = []
                st.session_state.select_all_clicked = False

            # Multiselect (will not override if you're selecting manually)
            selected_params = st.multiselect(
                "Choose parameters:",
                options=numeric_cols,
                default=st.session_state.time_series_selected_params
            )

            # Always sync selected parameters
            st.session_state.time_series_selected_params = selected_params

            # Plotting
            if selected_params and not df_filtered.empty:
                fig = px.line(
                    df_filtered,
                    x=time_col,
                    y=selected_params,
                    labels={"value": "Measurement", "variable": "Parameter"},
                    template="simple_white"
                )

                fig.update_traces(mode="lines", line=dict(width=2))
                fig.update_layout(
                    title="Water Quality Trends Over Time",
                    title_font=dict(size=18),
                    legend_title="Parameters",
                    height=450,
                    margin=dict(t=40, l=20, r=20, b=40),
                    xaxis_title="Date",
                    yaxis_title="",
                    font=dict(family="Segoe UI, sans-serif", size=13),
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Visualization Meters for Selected Parameters
                st.markdown("#### Visualization Meters")
                for param in selected_params:
                    if param in df_filtered.columns:
                        avg_value = df_filtered[param].mean()
                        st.metric(label=param, value=f"{avg_value:.2f}", delta=f"{df_filtered[param].max() - df_filtered[param].min():.2f}")

            else:
                st.info("Please select at least one parameter and ensure the selected filters have data to generate the plot.")
        else:
            st.warning("⚠️ No date/time column detected in the dataset.")


# Predictions Page
elif selected == "Predictions":
    st.image("taal_banner.png", use_container_width=True)
    st.title("🔮 Water Quality Predictions")
# Define water parameters and external factors
    water_params = ['pH', 'Water Temperature', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen']
    external_factors = ['Sulfide', 'Carbon Dioxide', 'Air Temperature', 'Wind Direction', 'Weather Condition']

    # Dropdown to select target parameter to predict
    target_col = st.selectbox("🎯 Select Water Parameter to Predict:", water_params)

    # Option to include external factors or not
    include_external = st.radio(
        "Include External Factors in Prediction?",
        ("Water Parameters Only", "Water + External Factors"),
        index=0
    )
    include_external = (include_external == "Water + External Factors")

    # Dropdown to select prediction frequency
    prediction_time = st.selectbox("📅 Select Prediction Time:", ["Weekly", "Monthly", "Yearly"])

    # Prepare features based on selection
    if include_external:
        selected_columns = water_params + external_factors
    else:
        selected_columns = water_params

    # Check if all selected columns and target exist in dataframe
    missing_cols = [col for col in selected_columns + [target_col] if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in data: {missing_cols}")
    else:
        data = df[selected_columns + [target_col]].dropna()

        if len(data) < 50:
            st.warning("Not enough data for training.")
        else:
            X = data[selected_columns].values
            y = data[target_col].values

            # Reshape X for RNN (CNN and LSTM)
            X_rnn = X.reshape((X.shape[0], X.shape[1], 1))

            # Pad X to 16 features for CNN-LSTM hybrid model (4x4x1)
            X_padded = np.pad(X, ((0, 0), (0, max(0, 16 - X.shape[1]))), 'constant')
            X_cnn_lstm = X_padded.reshape((X_padded.shape[0], 4, 4, 1))

            # Train/test/val split for rnn models
            X_temp, X_test, y_temp, y_test = train_test_split(X_rnn, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

            # Train/test/val split for hybrid model
            X_temp_h, X_test_h, y_temp_h, y_test_h = train_test_split(X_cnn_lstm, y, test_size=0.2, random_state=42)
            X_train_h, X_val_h, y_train_h, y_val_h = train_test_split(X_temp_h, y_temp_h, test_size=0.25, random_state=42)

            # Define and train CNN model
            cnn_model = Sequential([
                Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_rnn.shape[1], 1)),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(50, activation='relu'),
                Dense(1)
            ])
            cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            cnn_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=0)

            # Define and train LSTM model
            lstm_model = Sequential([
                LSTM(64, input_shape=(X_rnn.shape[1], 1)),
                Dense(50, activation='relu'),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            lstm_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=0)

            # Define and train CNN-LSTM model
            cnn_lstm_model = Sequential([
                TimeDistributed(Conv1D(32, kernel_size=2, activation='relu'), input_shape=(4, 4, 1)),
                TimeDistributed(MaxPooling1D(pool_size=2)),
                TimeDistributed(Flatten()),
                LSTM(64),
                Dense(50, activation='relu'),
                Dense(1)
            ])
            cnn_lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            cnn_lstm_model.fit(X_train_h, y_train_h, epochs=50, batch_size=16, validation_data=(X_val_h, y_val_h), verbose=0)

            # Forecasting function for external factors averages
            def forecast_external(df, cols, steps=12, window=3):
                if 'Date' in df.columns:
                    df_sorted = df.sort_values('Date').copy()
                elif 'date' in df.columns:
                    df_sorted = df.sort_values('date').copy()
                else:
                    df_sorted = df.copy()
                forecast = []
                for _ in range(steps):
                    avg = df_sorted[cols].tail(window).mean()
                    forecast.append(avg)
                    df_sorted = pd.concat([df_sorted, pd.DataFrame([avg])], ignore_index=True)
                return pd.DataFrame(forecast)

            # Map prediction times to steps and freq
            steps_map = {'weekly': 52, 'monthly': 12, 'yearly': 10}
            freq_map = {'weekly': 'W', 'monthly': 'MS', 'yearly': 'YS'}

            steps = steps_map[prediction_time.lower()]
            date_freq = freq_map[prediction_time.lower()]

            # Prepare forecast data for external factors or padding zero if not included
            if include_external:
                future_external = forecast_external(df, external_factors, steps=steps, window=3)
            else:
                # Fill zeros if external factors not included
                future_external = pd.DataFrame(np.zeros((steps, len(external_factors))), columns=external_factors)

            last_water = data[water_params].iloc[-1].values

            preds_cnn, preds_lstm, preds_hybrid = [], [], []

            for i in range(steps):
                ext = future_external.iloc[i].values
                if include_external:
                    full_input = np.concatenate([last_water, ext])
                else:
                    full_input = last_water

                # Use actual length of full_input to reshape
                input_rnn = full_input.reshape((1, full_input.shape[0], 1))

                # Pad input for CNN-LSTM hybrid model
                padded_input = np.pad(full_input, (0, max(0, 16 - full_input.shape[0])), 'constant')
                input_cnn_lstm = padded_input.reshape((1, 4, 4, 1))

                pred1 = cnn_model.predict(input_rnn, verbose=0)[0][0]
                pred2 = lstm_model.predict(input_rnn, verbose=0)[0][0]
                pred3 = cnn_lstm_model.predict(input_cnn_lstm, verbose=0)[0][0]

                preds_cnn.append(pred1)
                preds_lstm.append(pred2)
                preds_hybrid.append(pred3)


            if 'Date' in df.columns:
                last_date = pd.to_datetime(df['Date'].max())
            elif 'date' in df.columns:
                last_date = pd.to_datetime(df['date'].max())
            else:
                last_date = pd.Timestamp.today()

            future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=steps, freq=date_freq)
            result_df = pd.DataFrame({
                'Date': future_dates,
                'CNN': preds_cnn,
                'LSTM': preds_lstm,
                'CNN-LSTM': preds_hybrid
            })

            # Show result dataframe
            st.subheader(f"📋 Predicted {target_col} Levels")
            st.dataframe(result_df)

            # Plot CNN predictions
            st.subheader("🧠 CNN Predictions")
            fig_cnn, ax_cnn = plt.subplots(figsize=(10, 4))
            ax_cnn.plot(result_df['Date'], result_df['CNN'], label='CNN', color='blue', marker='o')
            ax_cnn.set_title(f'CNN - Predicted {target_col} ({prediction_time})')
            ax_cnn.set_xlabel('Date')
            ax_cnn.set_ylabel(f'{target_col}')
            ax_cnn.grid(True)
            ax_cnn.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig_cnn)

            # Plot LSTM predictions
            st.subheader("🔁 LSTM Predictions")
            fig_lstm, ax_lstm = plt.subplots(figsize=(10, 4))
            ax_lstm.plot(result_df['Date'], result_df['LSTM'], label='LSTM', color='green', marker='s')
            ax_lstm.set_title(f'LSTM - Predicted {target_col} ({prediction_time})')
            ax_lstm.set_xlabel('Date')
            ax_lstm.set_ylabel(f'{target_col}')
            ax_lstm.grid(True)
            ax_lstm.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig_lstm)

            # Plot CNN-LSTM predictions
            st.subheader("🔀 Hybrid CNN-LSTM Predictions")
            fig_hybrid, ax_hybrid = plt.subplots(figsize=(10, 4))
            ax_hybrid.plot(result_df['Date'], result_df['CNN-LSTM'], label='Hybrid CNN-LSTM', color='purple', marker='^')
            ax_hybrid.set_title(f'CNN-LSTM - Predicted {target_col} ({prediction_time})')
            ax_hybrid.set_xlabel('Date')
            ax_hybrid.set_ylabel(f'{target_col}')
            ax_hybrid.grid(True)
            ax_hybrid.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig_hybrid)

            # Recommendations based on average prediction
            st.subheader("📌 Recommendations")

            avg_pred = {
                'CNN': np.mean(preds_cnn),
                'LSTM': np.mean(preds_lstm),
                'CNN-LSTM': np.mean(preds_hybrid)
            }

            def generate_recommendation(param, value):
                if param == 'Dissolved Oxygen':
                    if value < 5:
                        return "⚠️ Low DO can harm aquatic life. Consider aeration or reducing organic load."
                    else:
                        return "✅ DO levels are healthy."
                elif param == 'Ammonia':
                    if value > 0.1:
                        return "⚠️ High ammonia levels. Improve filtration or reduce waste sources."
                    else:
                        return "✅ Ammonia levels are safe."
                elif param == 'Nitrate':
                    if value > 10:
                        return "⚠️ Elevated nitrate. Control runoff and review agricultural practices."
                    else:
                        return "✅ Nitrate levels are within acceptable limits."
                elif param == 'Phosphate':
                    if value > 0.1:
                        return "⚠️ High phosphate may cause algal blooms. Reduce detergent/agri runoff."
                    else:
                        return "✅ Phosphate levels are safe."
                elif param == 'pH':
                    if value < 6.5:
                        return "⚠️ Water is too acidic. Consider buffering with limestone or similar agents."
                    elif value > 8.5:
                        return "⚠️ Water is too basic. Investigate sources of alkalinity."
                    else:
                        return "✅ pH is within the safe range."
                elif param == 'Water Temperature':
                    if value > 30:
                        return "⚠️ Water too warm. It can reduce DO. Increase shade or flow."
                    elif value < 20:
                        return "⚠️ Water is quite cold. Monitor biological activity."
                    else:
                        return "✅ Temperature is optimal."
                else:
                    return "ℹ️ No specific recommendation for this parameter."

            # Create a container for recommendations
            for model in ['CNN', 'LSTM', 'CNN-LSTM']:
                avg = avg_pred[model]
                rec = generate_recommendation(target_col, avg)

                # Create a box for each model's recommendation
                with st.container():
                    st.markdown(f"### {model} Recommendation")
                    st.markdown(f"<div style='padding: 10px; border: 1px solid #2563EB; border-radius: 8px; background-color: #f0f6fc;'>"
                                f"<strong>{rec}</strong> (Avg Predicted: <code>{avg:.2f}</code>)</div>", unsafe_allow_html=True)


# Developer Info
elif selected == "Developer Info":
    st.image("taal_banner.png", use_container_width=True)
    st.title("👨‍💻 Developer Information")
    st.markdown("""
This interactive dashboard helps visualize and analyze **Taal Lake's water quality** data.

**🔧 Technologies Used:**
- Python
- Streamlit
- Pandas, Matplotlib, Plotly

**📍 Purpose:**
- Aid researchers and environmental engineers in tracking water quality trends.
- Support evidence-based environmental decisions.

**📁 Data Source:**
- Collected from Taal Lake monitoring stations
""")

    # Developer data
    devs = [

    {
        "name": "Lebron James G. Larido",
        "role": "Project Lead / Full-Stack Developer",
        "desc": "Oversaw the entire development process, handled both frontend and backend tasks, and ensured seamless integration across all components.",
        "image": "dev2.jpg"
    },
    {
        "name": "Clark Patrick G. Agravante",
        "role": "Dashboard Developer / UI/UX Designer",
        "desc": "Designed the user interface and experience, and developed the main dashboard layout and core components of the application.",
        "image": "dev1.jpg"
    },
    {
        "name": "Nel Johnceen Pulido",
        "role": "Data Analyst",
        "desc": "Handled data processing, analysis, and transformation to ensure accurate and meaningful insights were integrated into the system.",
        "image": "dev3.jpg"
    },
    {
        "name": "Johndel M. Orosco",
        "role": "Backend Developer",
        "desc": "Contributed to server-side development, database integration, and implementation of backend functionalities.",
        "image": "dev4.jpg"
    },
    {
         "name": "Precious Erica G. Sueño",
        "role": "UI Support & Deployment Coordinator",
        "desc": "Managed project deployment and documentation, and provided minor enhancements to the user interface during development.",
        "image": "dev6.png"

    },
    {
        "name": "Carl Louise Sembrano",
        "role": "Documentation Assistant",
        "desc": "Assisted in preparing and organizing the project documentation to ensure clarity, consistency, and completeness.",
        "image": "dev5.jpg"
    }

]


    # Display 3 developers per row
    for i in range(0, len(devs), 3):
        cols = st.columns(3)
        for j, dev in enumerate(devs[i:i+3]):
            with cols[j]:
                if dev["image"]:
                    st.image(dev["image"], width=150)
                st.markdown(f"**{dev['name']}**  \n*{dev['role']}*")
                st.caption(dev["desc"])

# Enhanced Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>© 2025 Taal Lake Water Quality Dashboard | Sustainable Monitoring Initiative</p>",
    unsafe_allow_html=True
)