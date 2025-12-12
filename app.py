from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI(title="🚕 The Future Of Taxis")

# Load pre-trained model
model_data = joblib.load("demand_forecasting_best_model.pkl")
best_model = model_data['model']
scaler = model_data['scaler']

FEATURES = ['pickup_hour', 'day_of_week', 'is_weekend']

# CSS for Taxi-themed UI
CSS_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

body { 
    font-family: 'Poppins', sans-serif; 
    background: #ffffff;
    color: #000;
    line-height: 1.6;
    min-height: 100vh;
    padding-bottom: 50px;
}

/* HEADER GOLD */
nav { 
    background: #FFD700;
    padding: 15px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    position: sticky;
    top: 0;
    z-index: 1000;
}

nav .nav-content {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
}

nav .logo {
    font-size: 24px;
    font-weight: 700;
    color: #000;
    text-decoration: none;
}

nav a { 
    color: #000; 
    text-decoration: none; 
    font-weight: 600;
    font-size: 16px;
    padding: 6px 16px;
    border-radius: 20px;
    transition: all 0.3s ease;
}

nav a:hover { 
    background: #fff;
    color: #000; 
}

.container { 
    max-width: 1000px; 
    margin: 50px auto; 
    padding: 0 15px;
}

.hero-card {
    background: #f2f2f2;
    padding: 50px 30px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 40px;
    color: #000;
    box-shadow: 0 5px 25px rgba(0,0,0,0.05);
}

.hero-card h1 { 
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 10px;
}

.hero-card .subtitle {
    font-size: 18px;
    font-weight: 500;
}

.card { 
    background: #f2f2f2;
    padding: 35px 30px;
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
}

h2 { 
    color: #000000; 
    font-size: 26px;
    margin-bottom: 20px;
    font-weight: 700;
}

/* FEATURE CARDS - WHITE ONLY */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.feature-item {
    background: #ffffff;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    transition: all 0.3s ease;
    border: 2px solid transparent;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
}

.feature-item:hover {
    border-color: #FFD700;
    transform: translateY(-3px);
}

.feature-icon {
    font-size: 36px;
    margin-bottom: 10px;
}

.form-group {
    width: 100%;
    max-width: 400px;
}

.form-group label {
    display: block;
    font-weight: 600;
    margin-bottom: 8px;
    font-size: 16px;
    color: #000;
}

input[type="date"] {
    width: 100%;
    padding: 12px 16px;
    border-radius: 15px;
    border: 2px solid #ddd;
    background: #fff;
    color: #000;
}

button { 
    padding: 12px 35px;
    border: none;
    border-radius: 50px;
    background: #FFD700;
    color: #000;
    font-weight: 700;
    font-size: 16px;
    cursor: pointer;
}

#results { 
    background: #f2f2f2;
    padding: 25px;
    border-radius: 15px;
    font-size: 18px;
    text-align: center;
}

/* STATS CARDS (TEXT ABOVE, NUMBER BELOW) */
.stats-row {
    display: flex;
    gap: 20px;
    margin-top: 20px;
    flex-wrap: wrap;
}

.stat-box {
    flex: 1;
    min-width: 180px;
    background: #ffffff;
    color: #000;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 2px solid #f2f2f2;
}

.stat-label {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
}

.stat-value {
    font-size: 28px;
    font-weight: 800;
    margin-top: 5px;
}

.chart-container img {
    display: block;
    margin: 20px auto;
    max-width: 100%;
    border-radius: 12px;
}
"""


# Base HTML template
BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} | 🚕 The Future Of Taxis</title>
<style>{style}</style>
</head>
<body>
<nav>
    <div class="nav-content">
        <a href="/" class="logo">🚕 The Future Of Taxis</a>
        <div class="nav-links">
            <a href="/">Home</a>
        </div>
    </div>
</nav>
<div class="container">
{content}
</div>
</body>
</html>
"""


# Home page
@app.get("/", response_class=HTMLResponse)
async def home():
    content = """
    <div class="hero-card">
        <h1>🚕 Welcome to the Future of Taxi</h1>
    </div>
    
    <div class="card">
        <h2>The Features</h2>
        <div class="features-grid">
            <div class="feature-item">
                <a href="/dashboard" style="text-decoration: none; color: inherit;">
                    <span class="feature-icon">📊</span>
                    <h3>Interactive Dashboard</h3>
                </a>
            </div>
            <div class="feature-item">
                <a href="/predict" style="text-decoration: none; color: inherit;">
                    <span class="feature-icon">⏰</span>
                    <h3>Predict Taxi Demand</h3>
                </a>
            </div>
        </div>
    </div>
    """
    return BASE_HTML.format(title="Home", style=CSS_STYLE, content=content)


# GET Predict
@app.get("/predict", response_class=HTMLResponse)
async def predict_get():
    content = """
    <div class="hero-card"><h1>📅 Predict Taxi Demand</h1></div>
    <div class="card">
        <form method="post">
            <div class="form-group">
                <label for="date">📆 Choose a Date</label>
                <input type="date" id="date" name="date" required>
            </div>
            <button type="submit">🔮 Generate Prediction</button>
        </form>
    </div>
    """
    return BASE_HTML.format(title="Predict", style=CSS_STYLE, content=content)


# POST Predict
@app.post("/predict", response_class=HTMLResponse)
async def predict_post(date: str = Form(...)):
    try:
        selected_date = pd.to_datetime(date)

        df_pred = pd.DataFrame({
            "pickup_hour": list(range(24)),
            "day_of_week": [selected_date.dayofweek] * 24,
            "is_weekend": [1 if selected_date.dayofweek in [5, 6] else 0] * 24
        })

        X_scaled = scaler.transform(df_pred[FEATURES].astype(float))
        preds = np.maximum(best_model.predict(X_scaled), 0)

        max_pred = np.max(preds)
        avg_pred = np.mean(preds)
        min_pred = np.min(preds)

        peak_hours = [i for i, v in enumerate(preds) if v == max_pred]
        peak_hours_text = ", ".join(f"{h}:00-{h}:59" for h in peak_hours)

        day_type = "Weekend" if selected_date.dayofweek in [5, 6] else "Weekday"

        # BAR COLORS (Peak = Yellow, Others = Light Gray)
        colors = ["#D9D9D9"] * 24
        for h in peak_hours:
            colors[h] = "#FFD700"

        # === PLOT ===
        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")

        bars = ax.bar(df_pred["pickup_hour"], preds, color=colors, edgecolor="#333", width=0.7)

        # BAR LABELS
        for bar, pred in zip(bars, preds):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(preds)*0.01,
                f"{int(pred)}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold"
            )

        # AXES
        ax.set_xlabel("Hour of Day", fontsize=14, fontweight="bold")
        ax.set_ylabel("Predicted Demand", fontsize=14, fontweight="bold")
        ax.set_title("Taxi Demand Forecast", fontsize=24, fontweight="bold")

        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, facecolor="#ffffff")
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # PAGE HTML
        content = f"""
        <div class="hero-card"><h1>📅 Predict Taxi Demand</h1></div>

        <div class="card">
            <form method="post">
                <div class="form-group">
                    <label for="date">📆 Choose Another Date</label>
                    <input type="date" id="date" name="date" required value="{selected_date.date()}">
                </div>
                <button type="submit">🔮 Generate Prediction</button>
            </form>
        </div>

        <div id="results">
            <p>{selected_date.strftime('%A, %B %d, %Y')}</p>
            Peak Demand Hours: {peak_hours_text}
        </div>

        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-label">Peak Demand</div>
                <div class="stat-value">{int(max_pred)}</div>
            </div>

            <div class="stat-box">
                <div class="stat-label">Average Demand</div>
                <div class="stat-value">{int(avg_pred)}</div>
            </div>

            <div class="stat-box">
                <div class="stat-label">Lowest Demand</div>
                <div class="stat-value">{int(min_pred)}</div>
            </div>

            <div class="stat-box">
                <div class="stat-label">Day Type</div>
                <div class="stat-value">{day_type}</div>
            </div>
        </div>

        <div class="chart-container">
            <img src="data:image/png;base64,{img_base64}" alt="Predicted Demand Chart">
        </div>
        """

        return BASE_HTML.format(title="Results", style=CSS_STYLE, content=content)

    except Exception as e:
        return BASE_HTML.format(
            title="Error",
            style=CSS_STYLE,
            content=f"""
            <div class="card">
                <h2>⚠️ Error</h2>
                <p style="color:red;">{str(e)}</p>
            </div>
            """
        )
