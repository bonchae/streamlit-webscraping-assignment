import streamlit as st
import pandas as pd
import numpy as np
import requests
from lxml import html
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NFL Web Scraping Assignment",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #013369, #d50a0a);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        background: linear-gradient(90deg, #f39c12, #e67e22);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .code-block {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏈 NFL Web Scraping & Business Intelligence Assignment</h1>
        <p>Interactive Guide to Capture Data from the Web & Extract Business Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📚 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        [
            "📋 Assignment Overview",
            "🌐 Understanding Web Scraping", 
            "💻 Code Walkthrough",
            "📊 Sample Data Analysis",
            "🎯 Practice Exercises",
            "✅ Requirements Checklist",
            "🆘 Help & Resources"
        ]
    )
    
    if page == "📋 Assignment Overview":
        show_assignment_overview()
    elif page == "🌐 Understanding Web Scraping":
        show_web_scraping_concepts()
    elif page == "💻 Code Walkthrough":
        show_code_walkthrough()
    elif page == "📊 Sample Data Analysis":
        show_sample_analysis()
    elif page == "🎯 Practice Exercises":
        show_practice_exercises()
    elif page == "✅ Requirements Checklist":
        show_requirements_checklist()
    elif page == "🆘 Help & Resources":
        show_help_resources()

def show_assignment_overview():
    st.markdown('<div class="section-header"><h2>📋 Assignment Overview</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Goals
        **Data Collection & Business Intelligence**
        
        This project involves:
        1. **Web Scraping**: Collecting NFL draft data from websites
        2. **Data Cleaning**: Processing and preparing scraped data
        3. **Business Intelligence**: Extracting meaningful insights
        
        ### 📊 Expected Dataset
        - **253 rows** of player data
        - **8 columns**: Pick, Player, Position, Age, Games, Completions, Attempts, College
        - **Source**: pro-football-reference.com (2017 NFL Draft)
        """)
        
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Important Note:</strong><br>
        Some players may not have complete information for Age, Pass_cmp, Pass_att, and College fields. This is normal and expected in real-world data.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://via.placeholder.com/300x200/013369/FFFFFF?text=NFL+Draft+Data", caption="Sample NFL Draft Data Visualization")
        
        st.markdown("""
        ### 🏆 Learning Outcomes
        - Web scraping techniques
        - Data cleaning methods
        - Business intelligence analysis
        - Python libraries usage
        - Data visualization skills
        """)

def show_web_scraping_concepts():
    st.markdown('<div class="section-header"><h2>🌐 Understanding Web Scraping</h2></div>', unsafe_allow_html=True)
    
    # Interactive web scraping concept
    st.markdown("### What is Web Scraping?")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Concept", "🛠️ Tools", "⚡ Process"])
    
    with tab1:
        st.markdown("""
        **Web scraping** is the process of automatically extracting data from websites. It's like having a robot that can:
        
        - 🌐 Visit web pages
        - 👀 Read the HTML content
        - 🎯 Find specific data elements
        - 📦 Extract and organize the data
        - 💾 Save it for analysis
        
        ### Why Web Scraping?
        - **Traditional methods**: Surveys, interviews, IT requests
        - **Modern methods**: APIs, web scraping, real-time data
        - **Benefits**: Automated, scalable, real-time data collection
        """)
        
        # Interactive demo
        if st.button("🔍 Try Mini Web Scraping Demo"):
            try:
                # Simple example with a public API
                import json
                demo_data = {
                    "player": "Sample Player",
                    "position": "QB", 
                    "college": "Sample University",
                    "pick": "1"
                }
                st.json(demo_data)
                st.success("✅ This is what scraped data might look like!")
            except:
                st.error("Demo unavailable - network issues")
    
    with tab2:
        st.markdown("""
        ### 🛠️ Python Libraries for Web Scraping
        
        | Library | Purpose | Example Use |
        |---------|---------|-------------|
        | `requests` | HTTP requests | Connect to websites |
        | `lxml` | HTML/XML parsing | Parse web page structure |
        | `BeautifulSoup` | HTML parsing | Alternative to lxml |
        | `selenium` | Browser automation | Dynamic content scraping |
        | `pandas` | Data manipulation | Organize scraped data |
        
        ### 📋 Your Assignment Uses:
        """)
        
        st.code("""
import requests      # Connect to NFL website
from lxml import html  # Parse HTML content
import pandas as pd    # Organize data into DataFrame
        """, language="python")
    
    with tab3:
        st.markdown("""
        ### ⚡ Web Scraping Process
        
        **Step 1: Connect**
        ```python
        r = requests.get('http://website.com')
        ```
        
        **Step 2: Parse**
        ```python
        data = html.fromstring(r.text)
        ```
        
        **Step 3: Extract**
        ```python
        player_names = data.xpath("//td[@data-stat='player']/text()")
        ```
        
        **Step 4: Organize**
        ```python
        df = pd.DataFrame(extracted_data)
        ```
        
        **Step 5: Analyze**
        ```python
        top_players = df.sort_values('games_played', ascending=False)
        ```
        """)

def show_code_walkthrough():
    st.markdown('<div class="section-header"><h2>💻 Code Walkthrough</h2></div>', unsafe_allow_html=True)
    
    # Code sections
    section = st.selectbox(
        "Select code section to explore:",
        [
            "1. Package Imports",
            "2. Website Connection", 
            "3. Data Extraction",
            "4. Data Cleaning",
            "5. Business Analysis"
        ]
    )
    
    if section == "1. Package Imports":
        st.markdown("### 📦 Required Package Imports")
        st.code("""
# Python packages for web scraping
import requests
from lxml import html

import pandas as pd

# python package for data visualization
import matplotlib.pyplot as plt
%matplotlib inline

# show 1000 rows
pd.set_option('display.max_rows', 1000)

# ignore warning messages
import warnings
warnings.filterwarnings('ignore')
        """, language="python")
        
        st.markdown("""
        **Explanation:**
        - `requests`: Makes HTTP requests to websites
        - `lxml.html`: Parses HTML content 
        - `pandas`: Data manipulation and analysis
        - `matplotlib`: Data visualization
        - Settings for better display and fewer warnings
        """)
    
    elif section == "2. Website Connection":
        st.markdown("### 🌐 Connecting to the NFL Website")
        st.code("""
# connecting to the website
r = requests.get('http://www.pro-football-reference.com/years/2017/draft.htm')
data = html.fromstring(r.text)
        """, language="python")
        
        st.markdown("""
        **What this does:**
        1. `requests.get()` - Downloads the webpage
        2. `html.fromstring()` - Converts HTML text into a parseable structure
        3. Now we can search through the webpage content
        """)
        
        if st.button("🔍 Show Website Structure Preview"):
            st.markdown("""
            ```html
            <table>
                <tbody>
                    <tr>
                        <td data-stat='draft_pick'>1</td>
                        <td data-stat='player'><a>Myles Garrett</a></td>
                        <td data-stat='pos'>DE</td>
                        <td data-stat='age'>21</td>
                        <!-- more columns... -->
                    </tr>
                </tbody>
            </table>
            ```
            """)
    
    elif section == "3. Data Extraction":
        st.markdown("### 🎯 XPath Data Extraction")
        st.code("""
# develop Xpaths and collect data
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
        """, language="python")
        
        st.markdown("""
        **XPath Explanation:**
        - `//tbody/tr` - Find all table rows
        - `td[@data-stat='draft_pick']` - Find cell with specific attribute
        - `/text()` - Extract the text content
        - `|` - OR operator for multiple possible paths
        """)
    
    elif section == "4. Data Cleaning":
        st.markdown("### 🧹 Data Cleaning Steps")
        
        cleaning_step = st.radio(
            "Choose cleaning step:",
            ["Convert to DataFrame", "Remove Brackets", "Handle Empty Rows", "Rename Columns", "Convert Data Types"]
        )
        
        if cleaning_step == "Convert to DataFrame":
            st.code("""
# convert the collected data to dataframe and view first five rows
df = pd.DataFrame(final)
df.head()
            """, language="python")
            
        elif cleaning_step == "Remove Brackets":
            st.code("""
# remove bracket from scraped data
df[0] = df[0].str[0]  # Pick
df[1] = df[1].str[0]  # Player  
df[2] = df[2].str[0]  # Position
df[3] = df[3].str[0]  # Age
df[4] = df[4].str[0]  # Games
df[5] = df[5].str[0]  # Completions
df[6] = df[6].str[0]  # Attempts
df[7] = df[7].str[0]  # College
            """, language="python")
            
        elif cleaning_step == "Handle Empty Rows":
            st.code("""
# remove empty rows
df = df.dropna(how='all')
print(f"Rows after cleaning: {len(df)}")
            """, language="python")
            
        elif cleaning_step == "Rename Columns":
            st.code("""
# rename column names for better understanding
df = df.rename(columns={
    0: 'Pick', 
    1: 'Player', 
    2: 'Pos', 
    3: 'Age', 
    4: 'G', 
    5: 'cmp', 
    6: 'att', 
    7: 'college'
})
            """, language="python")
            
        elif cleaning_step == "Convert Data Types":
            st.code("""
# convert object to number for analysis
df['G'] = pd.to_numeric(df['G'])
df['att'] = pd.to_numeric(df['att']) 
df['cmp'] = pd.to_numeric(df['cmp'])

# check data types
print(df.dtypes)
            """, language="python")
    
    elif section == "5. Business Analysis":
        st.markdown("### 📊 Business Intelligence Analysis")
        
        analysis_type = st.selectbox(
            "Choose analysis:",
            ["Pass Completion Rate", "Top Performers", "College Analysis", "Position Analysis"]
        )
        
        if analysis_type == "Pass Completion Rate":
            st.code("""
# Calculate pass completion rate
df['pass_completion_rate'] = df['cmp'] / df['att']

# View top 5 players by completion rate
top_completion = df.sort_values('pass_completion_rate', ascending=False).head()
print(top_completion[['Player', 'pass_completion_rate']])
            """, language="python")
            
        elif analysis_type == "Top Performers":
            st.code("""
# Top 5 players by games played
top_games = df.sort_values('G', ascending=False).head()
print("Top 5 by Games Played:")
print(top_games[['Player', 'Pos', 'G']])
            """, language="python")
            
        elif analysis_type == "College Analysis":
            st.code("""
# Top 10 colleges by number of draft picks
college_counts = df['college'].value_counts().head(10)
print("Top 10 Colleges:")
print(college_counts)
            """, language="python")
            
        elif analysis_type == "Position Analysis":
            st.code("""
# Most popular draft positions
position_counts = df['Pos'].value_counts()
print("Most Drafted Positions:")
print(position_counts.head())
            """, language="python")

def show_sample_analysis():
    st.markdown('<div class="section-header"><h2>📊 Sample Data Analysis</h2></div>', unsafe_allow_html=True)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    sample_data = generate_sample_data()
    
    st.markdown("### 🎯 Interactive Data Analysis Demo")
    st.markdown("*This uses simulated data to show what your analysis should look like*")
    
    # Analysis options
    analysis_option = st.selectbox(
        "Choose analysis to explore:",
        [
            "📊 Dataset Overview",
            "🏆 Top Performers", 
            "🎓 College Analysis",
            "📈 Position Distribution",
            "⚡ Pass Completion Analysis"
        ]
    )
    
    if analysis_option == "📊 Dataset Overview":
        st.markdown("### Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Players", len(sample_data))
        with col2:
            st.metric("Total Colleges", sample_data['college'].nunique())
        with col3:
            st.metric("Positions", sample_data['Pos'].nunique())
        with col4:
            st.metric("Avg Games", f"{sample_data['G'].mean():.1f}")
        
        st.markdown("### Sample Data Preview")
        st.dataframe(sample_data.head(10))
        
        st.markdown("### Data Types")
        st.text(str(sample_data.dtypes))
    
    elif analysis_option == "🏆 Top Performers":
        st.markdown("### Top 5 Players by Games Played")
        top_games = sample_data.nlargest(5, 'G')[['Player', 'Pos', 'G', 'college']]
        st.dataframe(top_games)
        
        # Visualization
        fig = px.bar(
            top_games, 
            x='Player', 
            y='G', 
            color='Pos',
            title="Top 5 Players by Games Played"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Top 5 by Pass Completion Rate")
        completion_data = sample_data.dropna(subset=['pass_completion_rate'])
        if not completion_data.empty:
            top_completion = completion_data.nlargest(5, 'pass_completion_rate')[['Player', 'Pos', 'pass_completion_rate']]
            st.dataframe(top_completion)
    
    elif analysis_option == "🎓 College Analysis":
        st.markdown("### Top 10 Colleges by Draft Picks")
        college_counts = sample_data['college'].value_counts().head(10)
        
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
        
        # Find specific college
        st.markdown("### Find Players from Specific College")
        college_search = st.selectbox("Select a college:", sample_data['college'].dropna().unique())
        if college_search:
            college_players = sample_data[sample_data['college'] == college_search]
            st.dataframe(college_players[['Player', 'Pos', 'Pick', 'G']])
    
    elif analysis_option == "📈 Position Distribution":
        st.markdown("### Position Analysis")
        pos_counts = sample_data['Pos'].value_counts()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Most Popular Positions:**")
            st.dataframe(pos_counts.head(10).reset_index())
        
        with col2:
            fig = px.bar(
                x=pos_counts.index[:10],
                y=pos_counts.values[:10],
                title="Top 10 Most Drafted Positions"
            )
            fig.update_xaxis(title="Position")
            fig.update_yaxis(title="Number of Players")
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_option == "⚡ Pass Completion Analysis":
        st.markdown("### Pass Completion Rate Analysis")
        
        # Filter players with passing stats
        passers = sample_data.dropna(subset=['cmp', 'att'])
        passers = passers[passers['att'] > 0]
        
        if not passers.empty:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                avg_completion = passers['pass_completion_rate'].mean()
                st.metric("Average Completion Rate", f"{avg_completion:.1%}")
                
                # Distribution
                fig = px.histogram(
                    passers,
                    x='pass_completion_rate',
                    title="Pass Completion Rate Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Scatter plot
                fig = px.scatter(
                    passers,
                    x='att',
                    y='cmp',
                    color='Pos',
                    title="Completions vs Attempts",
                    hover_data=['Player']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No passing statistics available in sample data")

def generate_sample_data():
    """Generate sample NFL draft data for demonstration"""
    np.random.seed(42)
    
    # Sample data
    positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K', 'P']
    colleges = ['Alabama', 'Ohio State', 'Clemson', 'LSU', 'Georgia', 'Oklahoma', 'USC', 'Michigan', 'Texas', 'Florida', 'Penn State', 'Wisconsin', 'Stanford', 'Washington', 'Miami']
    
    n_players = 253
    
    data = {
        'Pick': range(1, n_players + 1),
        'Player': [f"Player_{i}" for i in range(1, n_players + 1)],
        'Pos': np.random.choice(positions, n_players, p=[0.05, 0.08, 0.15, 0.05, 0.2, 0.15, 0.1, 0.15, 0.03, 0.04]),
        'Age': np.random.randint(20, 25, n_players),
        'G': np.random.randint(0, 17, n_players),
        'college': np.random.choice(colleges + [None] * 5, n_players)
    }
    
    # Add passing stats (mainly for QBs)
    cmp = []
    att = []
    
    for pos in data['Pos']:
        if pos == 'QB':
            attempts = np.random.randint(50, 400)
            completions = int(attempts * np.random.uniform(0.55, 0.75))
        elif pos in ['WR', 'RB']:
            # Some trick plays
            if np.random.random() < 0.1:
                attempts = np.random.randint(1, 5)
                completions = int(attempts * np.random.uniform(0.3, 0.8))
            else:
                attempts = None
                completions = None
        else:
            attempts = None
            completions = None
        
        att.append(attempts)
        cmp.append(completions)
    
    data['att'] = att
    data['cmp'] = cmp
    
    df = pd.DataFrame(data)
    
    # Calculate pass completion rate
    df['pass_completion_rate'] = df['cmp'] / df['att']
    
    return df

def show_practice_exercises():
    st.markdown('<div class="section-header"><h2>🎯 Practice Exercises</h2></div>', unsafe_allow_html=True)
    
    st.markdown("### Interactive Coding Practice")
    st.markdown("*Practice the key concepts before working on your assignment*")
    
    exercise = st.selectbox(
        "Choose a practice exercise:",
        [
            "🔍 XPath Practice",
            "🧹 Data Cleaning Practice", 
            "📊 Analysis Practice",
            "💻 Full Code Review"
        ]
    )
    
    if exercise == "🔍 XPath Practice":
        st.markdown("### XPath Selector Practice")
        st.markdown("Understanding XPath is crucial for web scraping. Practice with these examples:")
        
        xpath_quiz = st.radio(
            "What does this XPath select: `//td[@data-stat='player']/text()`",
            [
                "All table cells",
                "Text content of cells with data-stat='player'", 
                "Player names only",
                "All data-stat attributes"
            ]
        )
        
        if xpath_quiz == "Text content of cells with data-stat='player'":
            st.success("✅ Correct! This XPath finds table cells with the specific attribute and extracts their text.")
        elif xpath_quiz:
            st.error("❌ Try again. Think about what each part of the XPath does.")
        
        st.markdown("### XPath Builder")
        st.code("""
# Practice building XPath expressions:
# Structure: //element[@attribute='value']/text()

# Your turn - build XPath for draft pick:
pick_xpath = "//td[@data-stat='draft_pick']/text()"

# Build XPath for college name:
college_xpath = "//td[@data-stat='college_id']/a/text()"
        """, language="python")
    
    elif exercise == "🧹 Data Cleaning Practice":
        st.markdown("### Data Cleaning Simulation")
        
        # Show messy data
        messy_data = [
            [['1'], ['John Smith'], ['QB'], ['21']],
            [['2'], ['Jane Doe'], ['RB'], ['22']],
            [[], [], [], []],  # Empty row
            [['3'], ['Bob Wilson'], ['WR'], ['20']]
        ]
        
        st.markdown("**Original messy data:**")
        st.json(messy_data)
        
        if st.button("🧹 Clean the data"):
            # Clean it
            cleaned = []
            for row in messy_data:
                if any(row):  # Skip empty rows
                    clean_row = [item[0] if item else None for item in row]
                    cleaned.append(clean_row)
            
            df_clean = pd.DataFrame(cleaned, columns=['Pick', 'Player', 'Pos', 'Age'])
            st.markdown("**Cleaned data:**")
            st.dataframe(df_clean)
            st.success("✅ Data cleaned successfully!")
    
    elif exercise == "📊 Analysis Practice":
        st.markdown("### Analysis Practice")
        
        # Generate practice data
        practice_df = generate_sample_data().head(20)
        
        st.markdown("**Practice Dataset (20 players):**")
        st.dataframe(practice_df)
        
        analysis_task = st.selectbox(
            "Choose analysis task:",
            [
                "Find top 3 players by games",
                "Calculate average age",
                "Count players by position",
                "Find specific college players"
            ]
        )
        
        if analysis_task == "Find top 3 players by games":
            if st.button("Show Solution"):
                result = practice_df.nlargest(3, 'G')[['Player', 'G']]
                st.dataframe(result)
                st.code("df.nlargest(3, 'G')[['Player', 'G']]", language="python")
        
        elif analysis_task == "Calculate average age":
            if st.button("Show Solution"):
                avg_age = practice_df['Age'].mean()
                st.metric("Average Age", f"{avg_age:.1f}")
                st.code("df['Age'].mean()", language="python")
        
        elif analysis_task == "Count players by position":
            if st.button("Show Solution"):
                pos_counts = practice_df['Pos'].value_counts()
                st.dataframe(pos_counts)
                st.code("df['Pos'].value_counts()", language="python")
    
    elif exercise == "💻 Full Code Review":
        st.markdown("### Complete Code Review")
        st.markdown("Review the complete assignment solution:")
        
        with st.expander("🔍 Click to see full solution code"):
            st.code("""
# 1. Import packages
import requests
from lxml import html
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 2. Connect to website
r = requests.get('http://www.pro-football-reference.com/years/2017/draft.htm')
data = html.fromstring(r.text)

# 3. Extract data using XPath
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

# 4. Create DataFrame and clean data
df = pd.DataFrame(final)

# Remove brackets
for i in range(8):
    df[i] = df[i].str[0]

# Remove empty rows
df = df.dropna(how='all')

# Rename columns
df = df.rename(columns={0: 'Pick', 1: 'Player', 2:'Pos', 3:'Age', 4:'G', 5:'cmp', 6:'att', 7:'college'})

# Convert to numeric
df['G'] = pd.to_numeric(df['G'])
df['att'] = pd.to_numeric(df['att'])
df['cmp'] = pd.to_numeric(df['cmp'])

# 5. Business Intelligence Analysis
# Calculate pass completion rate
df['pass_completion_rate'] = df['cmp'] / df['att']

# Top 5 players by games played
top_games = df.sort_values('G', ascending=False).head()

# Top 5 players by pass completion rate
top_completion = df.sort_values('pass_completion_rate', ascending=False).head()

# Top 10 colleges by number of players
top_colleges = df['college'].value_counts().head(10)

# Kansas State players
kansas_players = df.loc[df['college'] == 'Kansas St.']

# Most popular position
popular_position = df['Pos'].value_counts().head()

print(f"Total players: {len(df)}")
print("Analysis complete!")
            """, language="python")

def show_requirements_checklist():
    st.markdown('<div class="section-header"><h2>✅ Requirements Checklist</h2></div>', unsafe_allow_html=True)
    
    st.markdown("### 📋 Assignment Requirements")
    st.markdown("Use this checklist to ensure you've completed all requirements:")
    
    # Create interactive checklist
    requirements = {
        "Data Collection": [
            "Import required packages (requests, lxml, pandas, matplotlib)",
            "Connect to pro-football-reference.com/years/2017/draft.htm",
            "Extract data using XPath selectors",
            "Collect all 8 required columns: Pick, Player, Pos, Age, G, Cmp, Att, College",
            "Store data in appropriate data structure"
        ],
        "Data Cleaning": [
            "Convert scraped data to pandas DataFrame", 
            "Remove brackets from list data (use .str[0])",
            "Remove empty rows using dropna()",
            "Rename columns to meaningful names",
            "Convert G, Cmp, Att columns to numeric data types",
            "Verify final dataset has 253 rows"
        ],
        "Business Intelligence": [
            "Calculate pass completion rate (Cmp/Att)",
            "Find top 5 players by games played",
            "Find top 5 players by pass completion rate", 
            "Identify top 10 colleges by number of draft picks",
            "Find all Kansas State players",
            "Determine most popular draft position",
            "Answer your own analytical question"
        ],
        "Code Quality": [
            "Include proper comments explaining each step",
            "Display outputs in Jupyter notebook",
            "Handle missing data appropriately",
            "Use proper pandas methods (head(), sort_values(), etc.)",
            "Include data type verification (df.dtypes, df.info())"
        ]
    }
    
    for category, items in requirements.items():
        st.markdown(f"### {category}")
        for i, item in enumerate(items):
            checked = st.checkbox(item, key=f"{category}_{i}")
            if checked:
                st.markdown(f"<span style='color: green'>✅ {item}</span>", unsafe_allow_html=True)
    
    # Progress calculation
    if st.button("📊 Calculate Completion Progress"):
        total_items = sum(len(items) for items in requirements.values())
        st.info(f"Total requirements: {total_items}")
        st.markdown("Mark checkboxes above to track your progress!")

def show_help_resources():
    st.markdown('<div class="section-header"><h2>🆘 Help & Resources</h2></div>', unsafe_allow_html=True)
    
    help_section = st.selectbox(
        "Choose help topic:",
        [
            "🔧 Troubleshooting Common Issues",
            "📚 Learning Resources", 
            "💡 Tips & Best Practices",
            "🤔 FAQ",
            "📞 Getting Additional Help"
        ]
    )
    
    if help_section == "🔧 Troubleshooting Common Issues":
        st.markdown("### Common Issues & Solutions")
        
        issue = st.selectbox(
            "Select your issue:",
            [
                "Website connection problems",
                "XPath not finding data",
                "Data cleaning errors",
                "Type conversion problems",
                "Empty DataFrame results"
            ]
        )
        
        if issue == "Website connection problems":
            st.markdown("""
            **Problem**: `requests.get()` fails or times out
            
            **Solutions**:
            1. Check internet connection
            2. Try different URL (website might be down)
            3. Add headers to mimic browser request:
            ```python
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            r = requests.get(url, headers=headers)
            ```
            4. Add timeout parameter:
            ```python
            r = requests.get(url, timeout=30)
            ```
            """)
            
        elif issue == "XPath not finding data":
            st.markdown("""
            **Problem**: XPath returns empty lists
            
            **Solutions**:
            1. Inspect the webpage HTML structure
            2. Test XPath in browser console
            3. Check for different HTML structure:
            ```python
            # Try alternative XPath expressions
            player = i.xpath("td[@data-stat='player']//text()")  # Get all text
            player = i.xpath(".//a/text()")  # Look for any link text
            ```
            4. Print intermediate results to debug:
            ```python
            print("Row HTML:", html.tostring(i))
            ```
            """)
            
        elif issue == "Data cleaning errors":
            st.markdown("""
            **Problem**: `.str[0]` fails or data still has brackets
            
            **Solutions**:
            1. Check data structure first:
            ```python
            print("Sample data:", df[0].iloc[0])
            print("Data type:", type(df[0].iloc[0]))
            ```
            2. Handle different data types:
            ```python
            # For lists
            df[0] = df[0].apply(lambda x: x[0] if isinstance(x, list) and x else x)
            
            # For mixed types
            df[0] = df[0].astype(str).str.replace('[', '').str.replace(']', '').str.replace("'", "")
            ```
            """)
            
        elif issue == "Type conversion problems":
            st.markdown("""
            **Problem**: `pd.to_numeric()` fails
            
            **Solutions**:
            1. Handle errors gracefully:
            ```python
            df['G'] = pd.to_numeric(df['G'], errors='coerce')  # NaN for invalid
            ```
            2. Clean non-numeric characters:
            ```python
            df['G'] = df['G'].astype(str).str.replace(r'[^0-9]', '', regex=True)
            df['G'] = pd.to_numeric(df['G'])
            ```
            3. Check for missing values:
            ```python
            print("Missing values:", df['G'].isna().sum())
            ```
            """)
            
        elif issue == "Empty DataFrame results":
            st.markdown("""
            **Problem**: DataFrame has no data or wrong number of rows
            
            **Solutions**:
            1. Debug the scraping process:
            ```python
            print("Total rows found:", len(final))
            print("First few rows:", final[:3])
            ```
            2. Check website structure changes
            3. Verify XPath expressions are correct
            4. Make sure you're getting the right table:
            ```python
            tables = data.xpath("//table")
            print("Number of tables found:", len(tables))
            ```
            """)
    
    elif help_section == "📚 Learning Resources":
        st.markdown("### Learning Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Web Scraping:**
            - [XPath Tutorial](https://www.w3schools.com/xml/xpath_intro.asp)
            - [Requests Documentation](https://docs.python-requests.org/)
            - [lxml Tutorial](https://lxml.de/tutorial.html)
            - [Web Scraping with Python](https://realpython.com/web-scraping-with-python/)
            
            **Data Analysis:**
            - [Pandas Documentation](https://pandas.pydata.org/docs/)
            - [Data Cleaning Guide](https://realpython.com/python-data-cleaning-numpy-pandas/)
            """)
        
        with col2:
            st.markdown("""
            **Practice Sites:**
            - [XPath Tester](https://www.freeformatter.com/xpath-tester.html)
            - [CSS Selector Tester](https://www.w3schools.com/cssref/trysel.asp)
            - [Pandas Exercises](https://github.com/guipsamora/pandas_exercises)
            
            **Video Tutorials:**
            - YouTube: "Web Scraping with Python"
            - YouTube: "Pandas Data Analysis"
            """)
    
    elif help_section == "💡 Tips & Best Practices":
        st.markdown("### Tips & Best Practices")
        
        tip_category = st.radio(
            "Choose tip category:",
            ["Web Scraping", "Data Cleaning", "Analysis", "Code Organization"]
        )
        
        if tip_category == "Web Scraping":
            st.markdown("""
            **🌐 Web Scraping Best Practices:**
            
            1. **Always check robots.txt** first (website.com/robots.txt)
            2. **Add delays** between requests to be respectful:
            ```python
            import time
            time.sleep(1)  # Wait 1 second between requests
            ```
            3. **Use headers** to identify your scraper:
            ```python
            headers = {'User-Agent': 'Educational Project - Your Name'}
            ```
            4. **Handle errors gracefully**:
            ```python
            try:
                r = requests.get(url)
                r.raise_for_status()  # Raise exception for bad status codes
            except requests.RequestException as e:
                print(f"Error: {e}")
            ```
            5. **Test XPath expressions** in browser console first
            """)
            
        elif tip_category == "Data Cleaning":
            st.markdown("""
            **🧹 Data Cleaning Best Practices:**
            
            1. **Always examine raw data first**:
            ```python
            print("Raw data sample:", final[:3])
            print("Data types:", [type(item) for item in final[0]])
            ```
            
            2. **Clean step by step and verify**:
            ```python
            print("Before cleaning:", len(df))
            df = df.dropna(how='all')
            print("After removing empty rows:", len(df))
            ```
            
            3. **Handle missing values appropriately**:
            ```python
            # Check missing values
            print(df.isnull().sum())
            
            # Decide how to handle them
            df['age'].fillna(df['age'].median(), inplace=True)  # Fill with median
            ```
            
            4. **Validate data types after conversion**:
            ```python
            print("Data types after conversion:")
            print(df.dtypes)
            ```
            """)
            
        elif tip_category == "Analysis":
            st.markdown("""
            **📊 Analysis Best Practices:**
            
            1. **Always verify your results**:
            ```python
            # Sanity check
            total_players = len(df)
            print(f"Analyzing {total_players} players")
            ```
            
            2. **Handle edge cases**:
            ```python
            # Check for division by zero
            df['completion_rate'] = df['cmp'] / df['att'].replace(0, np.nan)
            ```
            
            3. **Use descriptive variable names**:
            ```python
            top_performers_by_games = df.nlargest(5, 'G')
            highest_completion_rates = df.nlargest(5, 'completion_rate')
            ```
            
            4. **Add context to your findings**:
            ```python
            print(f"Top player played {top_games['G'].iloc[0]} games")
            print(f"Average games played: {df['G'].mean():.1f}")
            ```
            """)
            
        elif tip_category == "Code Organization":
            st.markdown("""
            **💻 Code Organization Best Practices:**
            
            1. **Use clear section headers**:
            ```python
            # =============================================================================
            # DATA COLLECTION
            # =============================================================================
            ```
            
            2. **Comment your XPath expressions**:
            ```python
            # Extract player name from either strong/a or just a tag
            player = i.xpath("td[@data-stat='player']/strong/a/text() | td[@data-stat='player']/a/text()")
            ```
            
            3. **Save intermediate results**:
            ```python
            # Save raw data in case you need to restart
            raw_df = pd.DataFrame(final)
            raw_df.to_csv('raw_nfl_data.csv', index=False)
            ```
            
            4. **Create reusable functions**:
            ```python
            def clean_column(series):
                \"\"\"Remove brackets and convert to string\"\"\"
                return series.str[0] if series.dtype == 'object' else series
            ```
            """)
    
    elif help_section == "🤔 FAQ":
        st.markdown("### Frequently Asked Questions")
        
        faq_items = {
            "Why do I get only 252 rows instead of 253?": """
            This is usually due to:
            1. Empty rows not being handled properly
            2. Header row being included/excluded
            3. Different table structure than expected
            
            Solution: Check your data cleaning steps and verify the website structure.
            """,
            
            "What if some players don't have college information?": """
            This is expected! The assignment notes mention that some players may not have complete information for Age, Pass_cmp, Pass_att, and College. This is normal in real-world data.
            
            Handle it with: `pd.to_numeric(errors='coerce')` for numeric columns.
            """,
            
            "My XPath isn't finding any data. What's wrong?": """
            Common causes:
            1. Website structure changed
            2. Incorrect XPath syntax
            3. Data is loaded dynamically (requires Selenium)
            
            Solution: Inspect the webpage HTML and test your XPath in browser console.
            """,
            
            "How do I handle the '|' operator in XPath?": """
            The '|' operator means "OR" in XPath. It tries the first expression, and if it fails, tries the second:
            
            ```python
            # Try strong/a first, then just a
            player = i.xpath("td[@data-stat='player']/strong/a/text() | td[@data-stat='player']/a/text()")
            ```
            """,
            
            "What's the difference between .text() and //text()?": """
            - `.text()` gets direct text content
            - `//text()` gets all text including nested elements
            
            Use `.text()` for cleaner results in most cases.
            """
        }
        
        for question, answer in faq_items.items():
            with st.expander(f"❓ {question}"):
                st.markdown(answer)
    
    elif help_section == "📞 Getting Additional Help":
        st.markdown("### Getting Additional Help")
        
        st.markdown("""
        **🎓 Academic Resources:**
        - Office hours with your instructor
        - Study groups with classmates
        - University tutoring center
        - Library research assistance
        
        **💻 Online Communities:**
        - Stack Overflow (programming questions)
        - Reddit r/learnpython
        - Python Discord communities
        - GitHub Discussions
        
        **📧 When to Ask for Help:**
        - You've tried debugging for 30+ minutes
        - You've searched online for solutions
        - You've reviewed the course materials
        - You can clearly describe what you've tried
        
        **❓ How to Ask Good Questions:**
        1. Describe what you're trying to do
        2. Show your current code
        3. Explain what you expected vs. what happened
        4. Include any error messages
        5. Mention what you've already tried
        """)
        
        st.markdown("""
        <div class="success-box">
        <strong>💡 Pro Tip:</strong> The best way to learn web scraping is by practicing! 
        Start with simple websites and gradually work up to more complex ones.
        </div>
        """, unsafe_allow_html=True)

# Add a final deployment note
def show_deployment_note():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;">
    📚 Interactive Web Scraping Assignment Guide<br>
    Built with Streamlit for enhanced student learning<br>
    Contact your instructor for technical support
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_deployment_note()
