import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(
    page_title="NFL Team Performance Explorer (2003-2023)",
    page_icon="🏈",
    layout='wide',
    initial_sidebar_state='expanded'
)

team_colors = {
    'Arizona Cardinals': '#97233F',
    'Atlanta Falcons': '#A71930',
    'Baltimore Ravens': '#241773',
    'Buffalo Bills': '#00338D',
    'Carolina Panthers': '#0085CA',
    'Chicago Bears': '#0B162A',
    'Cincinnati Bengals': '#FB4F14',
    'Cleveland Browns': '#311D00',
    'Dallas Cowboys': '#041E42',
    'Denver Broncos': '#002244',
    'Detroit Lions': '#0076B6',
    'Green Bay Packers': '#203731',
    'Houston Texans': '#03202F',
    'Indianapolis Colts': '#002C5F',
    'Jacksonville Jaguars': '#006778',
    'Kansas City Chiefs': '#E31837',
    'Las Vegas Raiders': '#000000',
    'Los Angeles Chargers': '#0080C6',
    'Los Angeles Rams': '#003594',
    'Miami Dolphins': '#008E97',
    'Minnesota Vikings': '#4F2683',
    'New England Patriots': '#002244',
    'New Orleans Saints': '#D3BC8D',
    'New York Giants': '#0B2265',
    'New York Jets': '#125740',
    'Philadelphia Eagles': '#004C54',
    'Pittsburgh Steelers': '#FFB612',
    'San Francisco 49ers': '#AA0000',
    'Seattle Seahawks': '#002244',
    'Tampa Bay Buccaneers': '#D50A0A',
    'Tennessee Titans': '#4B92DB',
    'Washington Commanders': '#5A1414'
}

@st.cache_data
def load_data():
    try:
        url = "https://github.com/carter-oleary/JHU/blob/main/EN685-662Data_Patterns_And_Representations/Module_3/team_stats_2003_2023.csv?raw=true"
        df = pd.read_csv(url)
        return df
    except FileNotFoundError:
        # Create sample data structure for demonstration
        st.error("CSV file not found. Using sample data structure.")
        years = list(range(2003, 2024))
        teams = ['Patriots', 'Steelers', 'Cowboys', '49ers', 'Packers', 'Ravens', 'Colts', 'Saints']
        
        sample_data = []
        for year in years:
            for team in teams:
                wins = np.random.randint(4, 17)
                sample_data.append({
                    'year': year,
                    'team': team,
                    'wins': wins,
                    'losses': 17-wins,
                    'win_loss_pct': float(wins)/17,
                    'points': np.random.randint(250, 550),
                    'points_opp': np.random.randint(250, 450),
                    'total_yards': np.random.randint(4500, 6500),
                    'pass_yds': np.random.randint(3000, 5000),
                    'rush_yds': np.random.randint(1500, 2500),
                    'turnovers': np.random.randint(15, 35)
                })
        
        return pd.DataFrame(sample_data)

df = load_data()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'year' in num_cols:
    num_cols.remove('year')
    
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# -------------------------
# 1. Introduction
# -------------------------
st.markdown("<h1 style='font-size: 48px; color: #FC0307; text-align: center;'>🏈 NFL Team Performance Explorer (2003-2023)</h1>", unsafe_allow_html=True)

st.markdown("""
---
### 📊 Dataset Overview
Welcome to Carter's Module 3 Assignment!  
This interactive application allows you to explore NFL team statistics from 2003-2023.  

Use the controls below to filter and analyze team statistics.

**Dataset Source:** [NFL Team Data 2003-2023 on Kaggle](https://www.kaggle.com/datasets/nickcantalupa/nfl-team-data-2003-2023)
""", unsafe_allow_html=True)

# -------------------------
# 2. Feature Selection
# -------------------------
st.markdown('---')
st.markdown('## 📝 Feature Selection')
st.markdown('Select the features (columns) you want to analyze:')

sel_cols = []
checkbox_cols = st.columns(4)
for i, col in enumerate(num_cols):
    index = i % 4
    with checkbox_cols[index]:
        if st.checkbox(col, value=(col in num_cols[:3]), key=f"checkbox_{col}"):
            sel_cols.append(col)

# Base columns always included
base_cols = [c for c in ['team', 'year'] if c in df.columns]
selected_columns = base_cols + sel_cols  
filtered_df = df[selected_columns].copy()

# -------------------------
# 3. Filtering Controls
# -------------------------
st.markdown('---')
st.markdown('## 🔍 Data Filtering')

team_col, year_col = st.columns(2)
with team_col:            
    if 'team' in df.columns:
        st.markdown('#### Filter by Teams')
        avail_teams = sorted(df['team'].unique())
        sel_teams = st.multiselect(
            'Select one or more teams:',
            options=avail_teams,
            default=avail_teams[0],
            help='Choose specific teams to focus analysis on'
        )
with year_col:
    st.markdown('#### Select Year Range')
    min_year = df['year'].min()
    max_year = df['year'].max()
    year_range = st.slider(
        'Year Range:',
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
        help='Filter data by selecting a year range'
    )

st.markdown('#### Select and Filter a Primary Feature')    
pri_feature_cols = st.columns(2)
if sel_cols:  # Only show if features selected
    with pri_feature_cols[0]:
        pri_feat = st.selectbox(
            'Select a primary feature:',
            options=sel_cols,
            index=0
        )
    with pri_feature_cols[1]:
        min_val = df[pri_feat].min()
        max_val = df[pri_feat].max()
        val_range = st.slider(
            f'{pri_feat} Range',
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            help='Filter data for your primary feature'
        )
else:
    st.info("👉 Select at least one feature above to enable primary feature filtering.")
    pri_feat, val_range = None, None

# Apply filters
if sel_teams:
    filtered_df = filtered_df[filtered_df['team'].isin(sel_teams)]
if year_range:
    filtered_df = filtered_df[
        (filtered_df['year'] >= year_range[0]) &
        (filtered_df['year'] <= year_range[1])
    ]
if pri_feat and val_range:
    filtered_df = filtered_df[
        (filtered_df[pri_feat] >= val_range[0]) &
        (filtered_df[pri_feat] <= val_range[1])
    ]

# -------------------------
# 4. Results
# -------------------------
st.markdown('---')
st.markdown('## 📈 Results')

st.dataframe(filtered_df, hide_index=True)

# -------------------------
# Visualization: Line Chart
# -------------------------
if pri_feat and 'year' in filtered_df.columns and 'team' in filtered_df.columns:
    st.markdown(f"### 📊 {pri_feat.title()} Trends Over Time")

    # Restrict color mapping to only selected teams
    selected_teams = filtered_df['team'].unique().tolist()
    selected_colors = [team_colors.get(team, "#999999") for team in selected_teams]

    line_chart = (
        alt.Chart(filtered_df)
        .mark_line(point=True)
        .encode(
            x=alt.X('year:O', title='Year'),
            y=alt.Y(f'{pri_feat}:Q', title=pri_feat.replace("_", " ").title()),
            color=alt.Color(
                'team:N',
                scale=alt.Scale(domain=selected_teams,
                                range=selected_colors),
                legend=alt.Legend(title="Team")
            ),
            tooltip=['year', 'team', pri_feat]
        )
        .properties(width=800, height=400)
        .interactive()
    )

    st.altair_chart(line_chart, use_container_width=True)

st.markdown("### 🧾 Summary Insights")
st.write(f"Number of records: {len(filtered_df)}")
if sel_cols:
    st.write(filtered_df[sel_cols].describe())




