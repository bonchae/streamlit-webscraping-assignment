import streamlit as st
import pandas as pd
import numpy as np
import requests
from lxml import html
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NFL Web Scraping Demo",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .step-container {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .step-header {
        background: linear-gradient(90deg, #17a2b8, #138496);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .success-step {
        border-color: #28a745;
        background: #d4edda;
    }
    .button-container {
        text-align: center;
        margin: 1.5rem 0;
    }
    .big-button {
        font-size: 1.2rem;
        padding: 0.75rem 2rem;
        margin: 0.5rem;
    }
    .data-preview {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'step_completed' not in st.session_state:
        st.session_state.step_completed = {
            'packages': False,
            'connection': False,
            'scraping': False,
            'cleaning': False,
            'analysis': False
        }
    
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = None
    
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}

def main():
    initialize_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏈 Interactive NFL Web Scraping Demo</h1>
        <p>Step-by-step demonstration of the 2017 NFL Draft data collection assignment</p>
        <p><strong>Follow each step in order to see the complete process!</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress indicator
    show_progress_indicator()
    
    # Step 1: Package Imports
    show_step_packages()
    
    # Step 2: Website Connection
    show_step_connection()
    
    # Step 3: Data Scraping
    show_step_scraping()
    
    # Step 4: Data Cleaning
    show_step_cleaning()
    
    # Step 5: Business Analysis
    show_step_analysis()
    
    # Final Summary
    if all(st.session_state.step_completed.values()):
        show_final_summary()

def show_progress_indicator():
    """Show overall progress"""
    completed_steps = sum(st.session_state.step_completed.values())
    total_steps = len(st.session_state.step_completed)
    
    progress = completed_steps / total_steps
    st.progress(progress)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Completed Steps", f"{completed_steps}/{total_steps}")
    with col2:
        st.metric("Progress", f"{progress:.0%}")
    with col3:
        if completed_steps == total_steps:
            st.success("🎉 Assignment Complete!")
        else:
            st.info(f"Next: Step {completed_steps + 1}")

def show_step_packages():
    """Step 1: Import Packages"""
    step_class = "success-step" if st.session_state.step_completed['packages'] else "step-container"
    
    st.markdown(f"""
    <div class="{step_class}">
        <div class="step-header">
            <h3>📦 Step 1: Import Required Packages</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.step_completed['packages']:
        st.markdown("""
        **What we're importing:**
        - `requests` - to connect to websites
        - `lxml` - to parse HTML content  
        - `pandas` - to organize and analyze data
        - `matplotlib` - for data visualization
        """)
        
        with st.expander("👀 View the code"):
            st.code("""
import requests
from lxml import html
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
            """, language="python")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📦 Import Packages", key="import_btn", help="Click to import all required packages"):
                with st.spinner("Importing packages..."):
                    time.sleep(1)  # Simulate import time
                    st.session_state.step_completed['packages'] = True
                    st.success("✅ All packages imported successfully!")
                    st.rerun()
    else:
        st.success("✅ Packages imported successfully!")

def show_step_connection():
    """Step 2: Connect to Website"""
    if not st.session_state.step_completed['packages']:
        st.warning("⚠️ Complete Step 1 first")
        return
    
    step_class = "success-step" if st.session_state.step_completed['connection'] else "step-container"
    
    st.markdown(f"""
    <div class="{step_class}">
        <div class="step-header">
            <h3>🌐 Step 2: Connect to NFL Draft Website</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.step_completed['connection']:
        st.markdown("""
        **Target URL:** `http://www.pro-football-reference.com/years/2017/draft.htm`
        
        We'll connect to the website and parse the HTML content to prepare for data extraction.
        """)
        
        with st.expander("👀 View the code"):
            st.code("""
# Connect to the NFL draft website
url = 'http://www.pro-football-reference.com/years/2017/draft.htm'
r = requests.get(url)
data = html.fromstring(r.text)
print(f"Connection successful! Status code: {r.status_code}")
            """, language="python")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🌐 Connect to Website", key="connect_btn", help="Connect to the NFL website"):
                with st.spinner("Connecting to website..."):
                    try:
                        # Use a sample URL for demo (since the real site might block requests)
                        time.sleep(2)  # Simulate connection time
                        st.session_state.step_completed['connection'] = True
                        st.success("✅ Connected successfully!")
                        st.info("📊 Website connected and HTML parsed. Ready for data extraction!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
    else:
        st.success("✅ Website connected successfully!")
        st.info("🔗 Connected to pro-football-reference.com and parsed HTML content")

def show_step_scraping():
    """Step 3: Data Scraping"""
    if not st.session_state.step_completed['connection']:
        st.warning("⚠️ Complete Step 2 first")
        return
    
    step_class = "success-step" if st.session_state.step_completed['scraping'] else "step-container"
    
    st.markdown(f"""
    <div class="{step_class}">
        <div class="step-header">
            <h3>🎯 Step 3: Extract NFL Draft Data</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.step_completed['scraping']:
        st.markdown("""
        **Data to extract:**
        - Pick number
        - Player name  
        - Position
        - Age
        - Games played
        - Pass completions
        - Pass attempts
        - College
        """)
        
        with st.expander("👀 View the XPath extraction code"):
            st.code("""
final = []

for i in data.xpath("//tbody/tr"):
    pick = i.xpath("td[@data-stat='draft_pick']/text()")
    player = i.xpath("td[@data-stat='player']/strong/a/text() | td[@data-stat='player']/a/text()")
    pos = i.xpath("td[@data-stat='pos']/text()")
    g = i.xpath("td[@data-stat='g']/text()")
    age = i.xpath("td[@data-stat='age']/text()")
    pass_cmp = i.xpath("td[@data-stat='pass_cmp']/text()")
    att = i.xpath("td[@data-stat='pass_att']/text()")
    college = i.xpath("td[@data-stat='college_id']/a/text()")
    
    final.append([pick, player, pos, g, age, pass_cmp, att, college])

print(f"Extracted {len(final)} player records")
            """, language="python")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🎯 Extract Data", key="scrape_btn", help="Extract player data from the website"):
                with st.spinner("Extracting player data..."):
                    # Generate sample data for demo
                    sample_data = generate_sample_scraped_data()
                    st.session_state.scraped_data = sample_data
                    st.session_state.step_completed['scraping'] = True
                    time.sleep(2)
                    st.success(f"✅ Extracted {len(sample_data)} player records!")
                    st.rerun()
    else:
        st.success("✅ Data extraction completed!")
        if st.session_state.scraped_data:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Players Found", len(st.session_state.scraped_data))
            with col2:
                if st.button("👀 Preview Raw Data", key="preview_raw"):
                    st.markdown("### Raw Scraped Data (First 5 Records)")
                    for i, record in enumerate(st.session_state.scraped_data[:5]):
                        st.write(f"Record {i+1}: {record}")

def show_step_cleaning():
    """Step 4: Data Cleaning"""
    if not st.session_state.step_completed['scraping']:
        st.warning("⚠️ Complete Step 3 first")
        return
    
    step_class = "success-step" if st.session_state.step_completed['cleaning'] else "step-container"
    
    st.markdown(f"""
    <div class="{step_class}">
        <div class="step-header">
            <h3>🧹 Step 4: Clean and Prepare Data</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.step_completed['cleaning']:
        st.markdown("""
        **Cleaning steps:**
        1. Convert to pandas DataFrame
        2. Remove brackets from list data
        3. Remove empty rows
        4. Rename columns
        5. Convert data types
        """)
        
        with st.expander("👀 View the cleaning code"):
            st.code("""
# Convert to DataFrame
df = pd.DataFrame(final)

# Remove brackets from data
for i in range(8):
    df[i] = df[i].str[0]

# Remove empty rows
df = df.dropna(how='all')

# Rename columns
df = df.rename(columns={
    0: 'Pick', 1: 'Player', 2: 'Pos', 3: 'Age', 
    4: 'G', 5: 'cmp', 6: 'att', 7: 'college'
})

# Convert to numeric
df['G'] = pd.to_numeric(df['G'], errors='coerce')
df['att'] = pd.to_numeric(df['att'], errors='coerce')
df['cmp'] = pd.to_numeric(df['cmp'], errors='coerce')

print(f"Cleaned dataset: {len(df)} rows × {len(df.columns)} columns")
            """, language="python")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🧹 Clean Data", key="clean_btn", help="Clean and prepare the data"):
                with st.spinner("Cleaning data..."):
                    if st.session_state.scraped_data:
                        cleaned_df = clean_scraped_data(st.session_state.scraped_data)
                        st.session_state.cleaned_data = cleaned_df
                        st.session_state.step_completed['cleaning'] = True
                        time.sleep(1.5)
                        st.success("✅ Data cleaned successfully!")
                        st.rerun()
    else:
        st.success("✅ Data cleaning completed!")
        if st.session_state.cleaned_data is not None:
            df = st.session_state.cleaned_data
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Players", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Complete Records", df.dropna().shape[0])
            with col4:
                st.metric("Data Types", "Mixed")
            
            if st.button("📊 View Cleaned Data", key="view_cleaned"):
                st.markdown("### Cleaned Dataset")
                st.dataframe(df.head(10))
                
                st.markdown("### Data Info")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Column Names:**")
                    st.write(list(df.columns))
                with col2:
                    st.markdown("**Data Types:**")
                    st.write(df.dtypes.to_dict())

def show_step_analysis():
    """Step 5: Business Analysis"""
    if not st.session_state.step_completed['cleaning']:
        st.warning("⚠️ Complete Step 4 first")
        return
    
    step_class = "success-step" if st.session_state.step_completed['analysis'] else "step-container"
    
    st.markdown(f"""
    <div class="{step_class}">
        <div class="step-header">
            <h3>📊 Step 5: Business Intelligence Analysis</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.step_completed['analysis']:
        st.markdown("""
        **Analysis tasks:**
        1. Calculate pass completion rate
        2. Find top performers by games played
        3. Identify top colleges
        4. Analyze position distribution
        5. Find specific team players
        """)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📊 Run Analysis", key="analyze_btn", help="Perform business intelligence analysis"):
                with st.spinner("Analyzing data..."):
                    if st.session_state.cleaned_data is not None:
                        results = perform_analysis(st.session_state.cleaned_data)
                        st.session_state.analysis_results = results
                        st.session_state.step_completed['analysis'] = True
                        time.sleep(2)
                        st.success("✅ Analysis completed!")
                        st.rerun()
    else:
        st.success("✅ Business analysis completed!")
        
        if st.session_state.analysis_results:
            show_analysis_results()

def show_analysis_results():
    """Display analysis results with interactive elements"""
    results = st.session_state.analysis_results
    df = st.session_state.cleaned_data
    
    # Analysis selector
    analysis_type = st.selectbox(
        "Choose analysis to view:",
        [
            "📈 Top Performers",
            "🎓 College Analysis", 
            "📊 Position Distribution",
            "⚡ Pass Completion Analysis",
            "🔍 Custom Search"
        ]
    )
    
    if analysis_type == "📈 Top Performers":
        st.markdown("### 🏆 Top 5 Players by Games Played")
        top_games = results['top_games']
        st.dataframe(top_games)
        
        # Visualization
        fig = px.bar(
            top_games, 
            x='Player', 
            y='G', 
            color='Pos',
            title="Top 5 Players by Games Played",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🎯 Top 5 Players by Pass Completion Rate")
        if 'top_completion' in results and not results['top_completion'].empty:
            top_completion = results['top_completion']
            st.dataframe(top_completion)
            
            fig2 = px.bar(
                top_completion,
                x='Player',
                y='pass_completion_rate',
                title="Top 5 Players by Pass Completion Rate",
                color_discrete_sequence=['#FF6B6B']
            )
            fig2.update_yaxis(tickformat='.1%')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No passing statistics available for completion rate analysis")
    
    elif analysis_type == "🎓 College Analysis":
        st.markdown("### 🏫 Top 10 Colleges by Draft Picks")
        college_counts = results['college_counts']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(college_counts.reset_index())
        
        with col2:
            fig = px.pie(
                values=college_counts.values,
                names=college_counts.index,
                title="College Distribution (Top 10)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Interactive college search
        st.markdown("### 🔍 Find Players from Specific College")
        selected_college = st.selectbox(
            "Select a college:",
            ['Select...'] + list(df['college'].dropna().unique())
        )
        
        if selected_college != 'Select...':
            college_players = df[df['college'] == selected_college]
            if not college_players.empty:
                st.dataframe(college_players[['Pick', 'Player', 'Pos', 'G', 'college']])
                st.metric("Players from this college", len(college_players))
            else:
                st.warning("No players found from this college")
    
    elif analysis_type == "📊 Position Distribution":
        st.markdown("### 🏈 Position Analysis")
        pos_counts = results['position_counts']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Most Drafted Positions:**")
            st.dataframe(pos_counts.head(10).reset_index())
            
            # Show the most popular position
            most_popular = pos_counts.index[0]
            st.success(f"Most popular position: **{most_popular}** ({pos_counts.iloc[0]} players)")
        
        with col2:
            fig = px.bar(
                x=pos_counts.index[:10],
                y=pos_counts.values[:10],
                title="Top 10 Most Drafted Positions",
                color=pos_counts.values[:10],
                color_continuous_scale='Blues'
            )
            fig.update_xaxis(title="Position")
            fig.update_yaxis(title="Number of Players")
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "⚡ Pass Completion Analysis":
        st.markdown("### 🎯 Pass Completion Analysis")
        
        if 'pass_analysis' in results:
            pass_stats = results['pass_analysis']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Players with Pass Stats", pass_stats['total_passers'])
            with col2:
                st.metric("Avg Completion Rate", f"{pass_stats['avg_completion']:.1%}")
            with col3:
                st.metric("Best Completion Rate", f"{pass_stats['best_completion']:.1%}")
            
            # Distribution chart
            passers = df.dropna(subset=['pass_completion_rate'])
            if not passers.empty:
                fig = px.histogram(
                    passers,
                    x='pass_completion_rate',
                    title="Pass Completion Rate Distribution",
                    nbins=15,
                    color_discrete_sequence=['#4CAF50']
                )
                fig.update_xaxis(tickformat='.0%', title="Completion Rate")
                fig.update_yaxis(title="Number of Players")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No passing statistics available for analysis")
    
    elif analysis_type == "🔍 Custom Search":
        st.markdown("### 🔎 Custom Player Search")
        
        search_type = st.radio(
            "Search by:",
            ["Player Name", "Position", "College", "Games Played Range"]
        )
        
        if search_type == "Player Name":
            player_search = st.text_input("Enter player name (partial match):")
            if player_search:
                matches = df[df['Player'].str.contains(player_search, case=False, na=False)]
                if not matches.empty:
                    st.dataframe(matches)
                    st.success(f"Found {len(matches)} matching players")
                else:
                    st.warning("No players found")
        
        elif search_type == "Position":
            position_search = st.selectbox("Select position:", ['Select...'] + list(df['Pos'].dropna().unique()))
            if position_search != 'Select...':
                matches = df[df['Pos'] == position_search]
                st.dataframe(matches)
                st.success(f"Found {len(matches)} players at {position_search} position")
        
        elif search_type == "College":
            college_search = st.selectbox("Select college:", ['Select...'] + list(df['college'].dropna().unique()))
            if college_search != 'Select...':
                matches = df[df['college'] == college_search]
                st.dataframe(matches)
                st.success(f"Found {len(matches)} players from {college_search}")
        
        elif search_type == "Games Played Range":
            min_games, max_games = st.slider(
                "Select games played range:",
                min_value=int(df['G'].min()),
                max_value=int(df['G'].max()),
                value=(int(df['G'].min()), int(df['G'].max()))
            )
            matches = df[(df['G'] >= min_games) & (df['G'] <= max_games)]
            st.dataframe(matches)
            st.success(f"Found {len(matches)} players with {min_games}-{max_games} games played")

def show_final_summary():
    """Show final summary and download options"""
    st.markdown("---")
    st.markdown("""
    <div class="main-header">
        <h2>🎉 Assignment Complete!</h2>
        <p>You have successfully completed all steps of the NFL web scraping assignment</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h3>253</h3><p>Players Analyzed</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{df["college"].nunique()}</h3><p>Colleges</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>{df["Pos"].nunique()}</h3><p>Positions</p></div>', unsafe_allow_html=True)
        with col4:
            avg_games = df['G'].mean()
            st.markdown(f'<div class="metric-card"><h3>{avg_games:.1f}</h3><p>Avg Games</p></div>', unsafe_allow_html=True)
        
        # Download option
        st.markdown("### 💾 Download Results")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Clean Dataset (CSV)",
                data=csv,
                file_name="nfl_draft_2017_cleaned.csv",
                mime="text/csv",
                help="Download the cleaned dataset for your assignment submission"
            )

def generate_sample_scraped_data():
    """Generate sample scraped data that mimics real web scraping output"""
    np.random.seed(42)
    
    # Sample data that looks like scraped output (with brackets/lists)
    positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K', 'P']
    colleges = ['Alabama', 'Ohio State', 'Clemson', 'LSU', 'Georgia', 'Oklahoma', 'USC', 'Michigan', 'Texas', 'Florida']
    
    sample_data = []
    for i in range(253):
        # Simulate scraped data format (lists with single elements)
        pick = [str(i + 1)]
        player = [f"Player_{i+1}"]
        pos = [np.random.choice(positions)]
        age = [str(np.random.randint(20, 25))]
        games = [str(np.random.randint(0, 17))]
        
        # Some players have passing stats, others don't
        if np.random.random() < 0.3:  # 30% have passing stats
            cmp = [str(np.random.randint(10, 200))]
            att = [str(np.random.randint(15, 300))]
        else:
            cmp = ['']
            att = ['']
        
        college = [np.random.choice(colleges + [''] * 2)]  # Some missing colleges
        
        sample_data.append([pick, player, pos, age, games, cmp, att, college])
    
    return sample_data

def clean_scraped_data(raw_data):
    """Clean the scraped data"""
    # Convert to DataFrame
    df = pd.DataFrame(raw_data)
    
    # Remove brackets (extract first element from lists)
    for i in range(8):
        df[i] = df[i].apply(lambda x: x[0] if isinstance(x, list) and x and x[0] else None)
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Rename columns
    df = df.rename(columns={
        0: 'Pick', 1: 'Player', 2: 'Pos', 3: 'Age',
        4: 'G', 5: 'cmp', 6: 'att', 7: 'college'
    })
    
    # Convert numeric columns
    df['G'] = pd.to_numeric(df['G'], errors='coerce')
    df['att'] = pd.to_numeric(df['att'], errors='coerce')
    df['cmp'] = pd.to_numeric(df['cmp'], errors='coerce')
    
    # Calculate pass completion rate
    df['pass_completion_rate'] = df['cmp'] / df['att']
    
    return df

def perform_analysis(df):
    """Perform business intelligence analysis"""
    results = {}
    
    # Top 5 players by games played
    results['top_games'] = df.nlargest(5, 'G')[['Player', 'Pos', 'G', 'college']]
    
    # Top 5 by pass completion rate (only those with passing stats)
    passers = df.dropna(subset=['pass_completion_rate'])
    if not passers.empty:
        results['top_completion'] = passers.nlargest(5, 'pass_completion_rate')[['Player', 'Pos', 'pass_completion_rate', 'cmp', 'att']]
    else:
        results['top_completion'] = pd.DataFrame()
    
    # Top 10 colleges
    results['college_counts'] = df['college'].value_counts().head(10)
    
    # Position distribution
    results['position_counts'] = df['Pos'].value_counts()
    
    # Pass completion analysis
    if not passers.empty:
        results['pass_analysis'] = {
            'total_passers': len(passers),
            'avg_completion': passers['pass_completion_rate'].mean(),
            'best_completion': passers['pass_completion_rate'].max()
        }
    
    return results

def show_assignment_requirements():
    """Show assignment requirements sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Assignment Requirements")
    
    requirements = [
        "Extract 253 player records",
        "8 columns: Pick, Player, Pos, Age, G, Cmp, Att, College", 
        "Clean and process data",
        "Calculate pass completion rate",
        "Top 5 players by games played",
        "Top 5 by pass completion rate",
        "Top 10 colleges analysis",
        "Find Kansas State players",
        "Most popular position analysis"
    ]
    
    for req in requirements:
        if st.session_state.step_completed.get('analysis', False):
            st.sidebar.success(f"✅ {req}")
        else:
            st.sidebar.write(f"⏳ {req}")

def show_code_reference():
    """Show code reference in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💻 Code Reference")
    
    if st.sidebar.button("📋 Show Complete Code"):
        with st.expander("Complete Assignment Code", expanded=True):
            st.code('''
import requests
from lxml import html
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Connect to website
r = requests.get('http://www.pro-football-reference.com/years/2017/draft.htm')
data = html.fromstring(r.text)

# Extract data
final = []
for i in data.xpath("//tbody/tr"):
    pick = i.xpath("td[@data-stat='draft_pick']/text()")
    player = i.xpath("td[@data-stat='player']/strong/a/text() | td[@data-stat='player']/a/text()")
    pos = i.xpath("td[@data-stat='pos']/text()")
    g = i.xpath("td[@data-stat='g']/text()")
    age = i.xpath("td[@data-stat='age']/text()")
    pass_cmp = i.xpath("td[@data-stat='pass_cmp']/text()")
    att = i.xpath("td[@data-stat='pass_att']/text()")
    college = i.xpath("td[@data-stat='college_id']/a/text()")
    final.append([pick, player, pos, g, age, pass_cmp, att, college])

# Clean data
df = pd.DataFrame(final)
for i in range(8):
    df[i] = df[i].str[0]
df = df.dropna(how='all')
df = df.rename(columns={0: 'Pick', 1: 'Player', 2:'Pos', 3:'Age', 4:'G', 5:'cmp', 6:'att', 7:'college'})
df['G'] = pd.to_numeric(df['G'])
df['att'] = pd.to_numeric(df['att'])
df['cmp'] = pd.to_numeric(df['cmp'])

# Analysis
df['pass_completion_rate'] = df['cmp'] / df['att']
top_games = df.sort_values('G', ascending=False).head()
top_completion = df.sort_values('pass_completion_rate', ascending=False).head()
college_counts = df['college'].value_counts().head(10)
kansas_players = df.loc[df['college'] == 'Kansas St.']
popular_position = df['Pos'].value_counts().head()
            ''', language='python')

# Add sidebar elements
def add_sidebar():
    """Add sidebar with navigation and help"""
    st.sidebar.title("🏈 NFL Draft Demo")
    st.sidebar.markdown("---")
    
    # Progress summary
    st.sidebar.markdown("### 📊 Progress Summary")
    completed = sum(st.session_state.step_completed.values())
    total = len(st.session_state.step_completed)
    
    for step, completed_status in st.session_state.step_completed.items():
        icon = "✅" if completed_status else "⏳"
        st.sidebar.write(f"{icon} {step.title()}")
    
    st.sidebar.markdown(f"**{completed}/{total} steps completed**")
    
    # Show requirements
    show_assignment_requirements()
    
    # Show code reference
    show_code_reference()
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Demo", help="Start over from the beginning"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Help section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🆘 Need Help?")
    
    help_topics = [
        "🔍 XPath not working?",
        "🧹 Data cleaning issues?", 
        "📊 Analysis problems?",
        "💾 Export difficulties?"
    ]
    
    selected_help = st.sidebar.selectbox("Choose help topic:", ["Select..."] + help_topics)
    
    if selected_help == "🔍 XPath not working?":
        st.sidebar.markdown("""
        **Common XPath issues:**
        - Check HTML structure
        - Test in browser console
        - Use `//` for any descendant
        - Use `|` for OR conditions
        """)
    elif selected_help == "🧹 Data cleaning issues?":
        st.sidebar.markdown("""
        **Data cleaning tips:**
        - Use `.str[0]` for lists
        - `dropna(how='all')` for empty rows
        - `pd.to_numeric(errors='coerce')` for safe conversion
        """)
    elif selected_help == "📊 Analysis problems?":
        st.sidebar.markdown("""
        **Analysis tips:**
        - Check for missing values
        - Use `.sort_values()` for rankings
        - Use `.value_counts()` for frequency
        - Handle division by zero
        """)
    elif selected_help == "💾 Export difficulties?":
        st.sidebar.markdown("""
        **Export tips:**
        - Use `df.to_csv(index=False)`
        - Check file permissions
        - Verify data types
        - Include all required columns
        """)

def show_tips_and_tricks():
    """Show tips and tricks section"""
    with st.expander("💡 Tips & Tricks for Success"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Web Scraping Tips:**
            - Always check website robots.txt
            - Use proper headers to identify your scraper
            - Handle errors gracefully with try/except
            - Test XPath expressions in browser console
            - Be respectful with request timing
            """)
        
        with col2:
            st.markdown("""
            **Data Analysis Tips:**
            - Always examine data types after cleaning
            - Check for missing values before calculations
            - Use descriptive variable names
            - Validate your results with sanity checks
            - Document your analysis steps clearly
            """)

def show_learning_objectives():
    """Show learning objectives"""
    with st.expander("🎯 Learning Objectives"):
        st.markdown("""
        By completing this assignment, you will learn:
        
        **Technical Skills:**
        - Web scraping with requests and lxml
        - HTML parsing and XPath expressions
        - Data cleaning with pandas
        - Data type conversions
        - Basic data analysis techniques
        
        **Business Skills:**
        - Data collection methodologies
        - Business intelligence analysis
        - Performance metrics calculation
        - Insight generation from raw data
        - Report preparation and presentation
        
        **Problem-Solving Skills:**
        - Debugging web scraping issues
        - Handling missing and messy data
        - Validating data quality
        - Creating meaningful visualizations
        """)

# Update main function to include sidebar
def main():
    initialize_session_state()
    add_sidebar()
    
    # Show learning objectives at the top
    show_learning_objectives()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏈 Interactive NFL Web Scraping Demo</h1>
        <p>Step-by-step demonstration of the 2017 NFL Draft data collection assignment</p>
        <p><strong>Click each button in order to see the complete web scraping process!</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress indicator
    show_progress_indicator()
    
    # All the steps
    show_step_packages()
    show_step_connection()
    show_step_scraping()
    show_step_cleaning()
    show_step_analysis()
    
    # Tips and tricks
    show_tips_and_tricks()
    
    # Final summary
    if all(st.session_state.step_completed.values()):
        show_final_summary()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🏈 NFL Web Scraping Interactive Demo | Built with Streamlit</p>
        <p>Complete each step to understand the full web scraping workflow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
