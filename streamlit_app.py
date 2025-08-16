import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="NFL Web Scraping Demo",
    page_icon="🏈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #013369, #d50a0a);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-box {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    .completed-step {
        border-color: #28a745;
        background: #d4edda;
    }
    .big-button {
        font-size: 1.5rem !important;
        padding: 1rem 3rem !important;
        margin: 1rem !important;
        border-radius: 30px !important;
    }
    .metric-box {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'data_collected' not in st.session_state:
        st.session_state.data_collected = False
    if 'data_cleaned' not in st.session_state:
        st.session_state.data_cleaned = False
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'clean_data' not in st.session_state:
        st.session_state.clean_data = None

def generate_sample_data():
    """Generate realistic sample NFL draft data"""
    np.random.seed(42)
    
    positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K', 'P']
    colleges = ['Alabama', 'Ohio State', 'Clemson', 'LSU', 'Georgia', 'Oklahoma', 
                'USC', 'Michigan', 'Texas', 'Florida', 'Penn State', 'Wisconsin',
                'Stanford', 'Washington', 'Miami', 'Kansas St.', 'Notre Dame']
    
    data = []
    for i in range(253):
        # Generate realistic player data
        pick = i + 1
        player = f"Player_{i+1:03d}"
        pos = np.random.choice(positions, p=[0.05, 0.08, 0.15, 0.05, 0.2, 0.15, 0.1, 0.15, 0.03, 0.04])
        age = np.random.randint(20, 25)
        games = np.random.randint(0, 17)
        
        # Passing stats mainly for QBs
        if pos == 'QB':
            att = np.random.randint(50, 400)
            cmp = int(att * np.random.uniform(0.55, 0.75))
        elif pos in ['WR', 'RB'] and np.random.random() < 0.1:
            att = np.random.randint(1, 5)
            cmp = int(att * np.random.uniform(0.3, 0.8))
        else:
            att = None
            cmp = None
        
        college = np.random.choice(colleges + [None] * 3)
        
        data.append({
            'Pick': pick,
            'Player': player,
            'Pos': pos,
            'Age': age,
            'G': games,
            'cmp': cmp,
            'att': att,
            'college': college
        })
    
    return pd.DataFrame(data)

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏈 NFL Draft Data Collection Demo</h1>
        <p>Collect, Clean, and Analyze 2017 NFL Draft Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress indicator
    col1, col2, col3 = st.columns(3)
    with col1:
        icon = "✅" if st.session_state.data_collected else "⏳"
        st.metric("Data Collection", f"{icon}")
    with col2:
        icon = "✅" if st.session_state.data_cleaned else "⏳"
        st.metric("Data Cleaning", f"{icon}")
    with col3:
        icon = "✅" if st.session_state.analysis_done else "⏳"
        st.metric("Analysis", f"{icon}")
    
    # Step 1: Data Collection
    step1_class = "completed-step" if st.session_state.data_collected else "step-box"
    st.markdown(f'<div class="{step1_class}">', unsafe_allow_html=True)
    st.markdown("### 🌐 Step 1: Collect Data from Website")
    st.markdown("**Target:** pro-football-reference.com/years/2017/draft.htm")
    st.markdown("**Goal:** Extract all 253 NFL draft picks with player information")
    
    if not st.session_state.data_collected:
        if st.button("🚀 Start Data Collection", key="collect", help="Collect NFL draft data from website"):
            with st.spinner("Connecting to website and extracting data..."):
                time.sleep(3)  # Simulate data collection time
                st.session_state.raw_data = generate_sample_data()
                st.session_state.data_collected = True
                st.success("✅ Successfully collected 253 player records!")
                st.rerun()
    else:
        st.success("✅ Data collection completed!")
        if st.button("👀 View Raw Data"):
            st.dataframe(st.session_state.raw_data.head(10))
            st.info(f"📊 Total records collected: {len(st.session_state.raw_data)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 2: Data Cleaning
    if st.session_state.data_collected:
        step2_class = "completed-step" if st.session_state.data_cleaned else "step-box"
        st.markdown(f'<div class="{step2_class}">', unsafe_allow_html=True)
        st.markdown("### 🧹 Step 2: Clean and Prepare Data")
        st.markdown("**Tasks:** Remove empty rows, fix data types, calculate new metrics")
        
        if not st.session_state.data_cleaned:
            if st.button("🔧 Clean Data", key="clean", help="Clean and prepare the collected data"):
                with st.spinner("Cleaning and preparing data..."):
                    time.sleep(2)
                    # Clean the data
                    df = st.session_state.raw_data.copy()
                    df['pass_completion_rate'] = df['cmp'] / df['att']
                    st.session_state.clean_data = df
                    st.session_state.data_cleaned = True
                    st.success("✅ Data cleaned and prepared successfully!")
                    st.rerun()
        else:
            st.success("✅ Data cleaning completed!")
            if st.button("📊 View Clean Data"):
                df = st.session_state.clean_data
                st.dataframe(df.head(10))
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Players", len(df))
                with col2:
                    st.metric("Positions", df['Pos'].nunique())
                with col3:
                    st.metric("Colleges", df['college'].nunique())
                with col4:
                    avg_games = df['G'].mean()
                    st.metric("Avg Games", f"{avg_games:.1f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 3: Analysis
    if st.session_state.data_cleaned:
        step3_class = "completed-step" if st.session_state.analysis_done else "step-box"
        st.markdown(f'<div class="{step3_class}">', unsafe_allow_html=True)
        st.markdown("### 📊 Step 3: Business Intelligence Analysis")
        st.markdown("**Goal:** Extract insights about player performance, colleges, and positions")
        
        if not st.session_state.analysis_done:
            if st.button("📈 Run Analysis", key="analyze", help="Perform business intelligence analysis"):
                with st.spinner("Analyzing data and generating insights..."):
                    time.sleep(2)
                    st.session_state.analysis_done = True
                    st.success("✅ Analysis completed!")
                    st.rerun()
        else:
            st.success("✅ Analysis completed!")
            show_analysis_results()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Final download
    if st.session_state.analysis_done:
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            csv = st.session_state.clean_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Dataset",
                data=csv,
                file_name="nfl_draft_2017_complete.csv",
                mime="text/csv"
            )

def show_analysis_results():
    """Display analysis results"""
    df = st.session_state.clean_data
    
    st.markdown("### 🏆 Key Findings")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏈 Top Players", "🎓 Colleges", "📊 Positions", "⚡ Passing"])
    
    with tab1:
        st.markdown("#### Top 5 Players by Games Played")
        top_games = df.nlargest(5, 'G')[['Player', 'Pos', 'G', 'college']]
        st.dataframe(top_games, use_container_width=True)
        
        # Chart
        fig = px.bar(top_games, x='Player', y='G', color='Pos', 
                     title="Top 5 Players by Games Played")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Top 10 Colleges by Draft Picks")
        college_counts = df['college'].value_counts().head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(college_counts.reset_index(), use_container_width=True)
        
        with col2:
            fig = px.pie(values=college_counts.values, names=college_counts.index,
                        title="College Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Special search
        st.markdown("#### Find Kansas State Players")
        kansas_players = df[df['college'] == 'Kansas St.']
        if len(kansas_players) > 0:
            st.dataframe(kansas_players[['Pick', 'Player', 'Pos', 'G']], use_container_width=True)
            st.info(f"Found {len(kansas_players)} players from Kansas State")
        else:
            st.warning("No Kansas State players found in this dataset")
    
    with tab3:
        st.markdown("#### Position Distribution")
        pos_counts = df['Pos'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pos_counts.head(10).reset_index(), use_container_width=True)
            most_popular = pos_counts.index[0]
            st.success(f"Most popular position: **{most_popular}** ({pos_counts.iloc[0]} players)")
        
        with col2:
            fig = px.bar(x=pos_counts.index[:8], y=pos_counts.values[:8],
                        title="Most Drafted Positions")
            fig.update_xaxis(title="Position")
            fig.update_yaxis(title="Number of Players")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### Pass Completion Analysis")
        passers = df.dropna(subset=['pass_completion_rate'])
        
        if len(passers) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Players with Pass Stats", len(passers))
            with col2:
                avg_completion = passers['pass_completion_rate'].mean()
                st.metric("Avg Completion Rate", f"{avg_completion:.1%}")
            with col3:
                best_completion = passers['pass_completion_rate'].max()
                st.metric("Best Completion Rate", f"{best_completion:.1%}")
            
            # Top 5 by completion rate
            st.markdown("#### Top 5 by Pass Completion Rate")
            top_completion = passers.nlargest(5, 'pass_completion_rate')[['Player', 'Pos', 'pass_completion_rate', 'cmp', 'att']]
            st.dataframe(top_completion, use_container_width=True)
            
            # Distribution chart
            fig = px.histogram(passers, x='pass_completion_rate', nbins=15,
                             title="Pass Completion Rate Distribution")
            fig.update_xaxis(tickformat='.0%', title="Completion Rate")
            fig.update_yaxis(title="Number of Players")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No passing statistics available in this dataset")

if __name__ == "__main__":
    main()
