# dashboard/app.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests

st.set_page_config(page_title="CineIQ", page_icon="🎬", layout="wide")
st.title("🎬 CineIQ — Your Personal Movie Intelligence")

API_URL = "http://localhost:8000"

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("Your Preferences")
user_id    = st.sidebar.number_input("User ID", min_value=1, value=1)
seed_movie = st.sidebar.text_input("Seed Movie", value="The Dark Knight")
n_recs     = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# ── Recommendations ───────────────────────────────────────
if st.sidebar.button("Get Recommendations"):
    with st.spinner("Analyzing your taste..."):
        resp = requests.post(f"{API_URL}/recommend", json={
            "user_id": user_id,
            "seed_movie": seed_movie,
            "n": n_recs
        })
        if resp.status_code == 200:
            data = resp.json()
            recs = pd.DataFrame(data['recommendations'])
            
            st.subheader(f"Top {n_recs} Picks for You")
            for _, row in recs.iterrows():
                with st.expander(f"🎥 Movie ID {row['title']} — Score: {row['score']:.2f}"):
                    st.write(f"**Why recommended:** {row.get('explanation', 'N/A')}")
        else:
            st.error("API error — make sure FastAPI is running")

# ── User Taste Dashboard ──────────────────────────────────
st.subheader("Your Taste Profile")
col1, col2 = st.columns(2)

with col1:
    # Genre radar chart
    genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Sci-Fi', 'Romance']
    scores = [0.8, 0.6, 0.4, 0.75, 0.9, 0.3]   # replace with real user data
    
    fig = go.Figure(go.Scatterpolar(
        r=scores, theta=genres, fill='toself', name='Genre Affinity'
    ))
    fig.update_layout(title="Genre Radar", polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Decade preference bar chart
    decades = ['1980s', '1990s', '2000s', '2010s', '2020s']
    counts  = [12, 28, 35, 45, 20]   # replace with real user data
    
    fig2 = px.bar(x=decades, y=counts, title="Movies Watched by Decade",
                  color=counts, color_continuous_scale='viridis')
    st.plotly_chart(fig2, use_container_width=True)

# Run with: streamlit run dashboard/app.py