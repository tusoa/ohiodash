import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Ohio Traffic Safety Survey",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM THEME - Sidebar
# =============================================================================
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar - green */
    [data-testid="stSidebar"] {
        background-color: #26686d;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    /* Sidebar metrics - white text */
    [data-testid="stSidebar"] [data-testid="metric-container"] * {
        color: #ffffff !important;
        background: transparent !important;
    }
    
    /* Sidebar collapse button */
    [data-testid="stSidebar"] button svg,
    [data-testid="collapsedControl"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    
    /* Override multiselect pill colors - green background, white text */
    [data-testid="stSidebar"] span[data-baseweb="tag"] {
        background-color: #1a4a4f !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] span[data-baseweb="tag"] span {
        color: #ffffff !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #666666 !important;
    }
    
    /* Tab styling - ensure visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f5f9;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #475569 !important;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #666666 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Metric styling - base */
    [data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
    }
    
    /* Override Streamlit's metric value color - slate for main area */
    div[data-testid="stMetricValue"] {
        color: #666666 !important;
    }
    
    /* Keep sidebar metrics white */
    [data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Main area metrics - slate text */
    [data-testid="stMainBlockContainer"] [data-testid="metric-container"] * {
        color: #666666 !important;
        background: transparent !important;
    }
    
    /* Main content area text - slate gray */
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] span,
    [data-testid="stMainBlockContainer"] em,
    [data-testid="stMainBlockContainer"] strong,
    [data-testid="stMainBlockContainer"] li {
        color: #666666 !important;
    }
    
    /* Selectbox and input labels */
    .stSelectbox label, .stMultiSelect label, .stRadio label {
        color: #666666 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# COLOR SCHEME - Institution Branding
# =============================================================================
COLORS = {
    'primary': '#26686d',      # Teal
    'secondary': '#5d1542',    # Burgundy
    'accent': '#dcaa38',       # Gold
    'neutral': '#666666',      # Slate gray
    'warm': '#5d1542',         # Burgundy (for "dangerous/likely")
    'categorical': ['#26686d', '#5d1542', '#dcaa38', '#666666', '#94a3b8'],
    'diverging': ['#26686d', '#666666', '#5d1542']  # Safe/Unsure/Dangerous or Unlikely/Unsure/Likely
}

# =============================================================================
# VARIABLE DEFINITIONS
# =============================================================================
# Ohio cities
CITY_ORDER = ['Columbus', 'Cincinnati', 'Cleveland', 'Dayton', 'Toledo', 'Akron', 
              'I live in Ohio, but not in one of the..']

# Race variables (binary)
RACE_VARS = {
    'white': 'White',
    'black': 'Black/African American',
    'latine': 'Latine/Hispanic',
    'asian': 'Asian/Pacific Islander',
    'multirace': 'Multiracial/Biracial',
    'me': 'Middle Eastern',
    'northafrican': 'North African',
    'notlisted': 'Not Listed',
    'pnarace': 'Prefer Not to Answer'
}

# Risky driving behavior variables (binary: 0/1)
BEHAVIOR_VARS = {
    'alcoholbin': 'Driving within 2 hrs of 3+ drinks',
    'cann5bin': 'Driving within 5 hrs of smoking cannabis',
    'cann9bin': 'Driving within 9 hrs of ingesting cannabis',
    'simbin': 'Driving within 2 hrs of alcohol + cannabis',
    'rxbin': 'Driving while feeling effects of Rx/drugs',
    'textbin': 'Texting/manually using phone while driving',
    'speedbin': 'Driving 10+ mph over speed limit',
    'drowsybin': 'Driving after less than 5 hrs sleep',
    'seatbeltbin': 'Driving without seatbelt'
}

# Danger perception variables
DANGER_VARS = {
    'dangeralc': 'Driving within 2 hrs of 3+ drinks',
    'dangercann5': 'Driving within 5 hrs of smoking cannabis',
    'dangercann9': 'Driving within 9 hrs of ingesting cannabis',
    'dangersim': 'Driving within 2 hrs of alcohol + cannabis',
    'dangerrx': 'Driving while feeling effects of Rx/drugs',
    'dangertext': 'Texting/manually using phone while driving',
    'dangerspeed': 'Driving 10+ mph over speed limit',
    'dangerdrowsy': 'Driving after less than 5 hrs sleep',
    'dangerseatbelt': 'Driving without seatbelt'
}

# Enforcement perception variables
LEGAL_VARS = {
    'legalalc': 'Driving within 2 hrs of 3+ drinks',
    'legalcann5': 'Driving within 5 hrs of smoking cannabis',
    'legalcann9': 'Driving within 9 hrs of ingesting cannabis',
    'legalsim': 'Driving within 2 hrs of alcohol + cannabis',
    'legalrx': 'Driving while feeling effects of Rx/drugs',
    'legaltext': 'Texting/manually using phone while driving',
    'legalspeed': 'Driving 10+ mph over speed limit',
    'legaldrowsy': 'Driving after less than 5 hrs sleep',
    'legalseatbelt': 'Driving without seatbelt'
}

# Peer norms variables
NORMS_VARS = {
    'normsalc': 'Driving within 2 hrs of 3+ drinks',
    'normscann5': 'Driving within 5 hrs of smoking cannabis',
    'normscann9': 'Driving within 9 hrs of ingesting cannabis',
    'normssim': 'Driving within 2 hrs of alcohol + cannabis',
    'normsrx': 'Driving while feeling effects of Rx/drugs',
    'normstext': 'Texting/manually using phone while driving',
    'normsspeed': 'Driving 10+ mph over speed limit',
    'normsdrowsy': 'Driving after less than 5 hrs sleep',
    'normsbelt': 'Driving without seatbelt'
}

# Response orders
DANGER_ORDER = ['Very dangerous', 'Somewhat dangerous', 'Unsure', 'Somewhat safe', 'Very safe', 'Prefer not to answer']
LIKELIHOOD_ORDER = ['Very likely', 'Somewhat likely', 'Unsure', 'Somewhat unlikely', 'Very unlikely', 'Prefer not to answer']
FREQUENCY_ORDER = ['Never', 'Once', 'Twice', 'More than twice', 'Prefer not to answer']

# Driving for income categories
INCOME_DRIVING_ORDER = ['No', 'Yes, for delivery (e.g., Amazon Flex,..', 
                        'Yes, for rideshare (e.g., Uber, Lyft)', 
                        'Yes, other type of paid driving', 'Prefer not to answer']

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data():
    """Load and prepare the survey data."""
    try:
        df = pd.read_csv('ohiodash.csv')
        return df
    except FileNotFoundError:
        st.error("Data file 'ohiodash.csv' not found. Please ensure it's in the same directory as this script.")
        return None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def calculate_prevalence_binary(df, var):
    """Calculate prevalence for binary (0/1) variables."""
    if var not in df.columns:
        return None
    valid = df[var].dropna()
    if len(valid) == 0:
        return None
    return (valid.sum() / len(valid)) * 100

def calculate_distribution(df, var, order=None):
    """Calculate distribution of categorical variable."""
    if var not in df.columns:
        return pd.DataFrame()
    
    counts = df[var].value_counts()
    total = counts.sum()
    
    result = pd.DataFrame({
        'Category': counts.index,
        'Count': counts.values,
        'Percentage': (counts.values / total * 100)
    })
    
    if order:
        result['Category'] = pd.Categorical(result['Category'], categories=order, ordered=True)
        result = result.sort_values('Category').dropna(subset=['Category'])
    
    return result

def create_prevalence_chart(df, var_dict, title, color=None, is_binary=True):
    """Create horizontal bar chart showing prevalence rates."""
    if color is None:
        color = COLORS['primary']
    
    data = []
    for var, label in var_dict.items():
        if is_binary:
            prev = calculate_prevalence_binary(df, var)
        else:
            # For non-binary, calculate % who did at least once
            if var in df.columns:
                valid = df[var].dropna()
                if len(valid) > 0:
                    prev = (valid[valid != 'Never'].count() / len(valid)) * 100
                else:
                    prev = None
            else:
                prev = None
        if prev is not None:
            data.append({'Behavior': label, 'Prevalence': prev})
    
    if not data:
        return None
    
    chart_df = pd.DataFrame(data).sort_values('Prevalence', ascending=True)
    n = len(df)
    
    fig = px.bar(
        chart_df,
        x='Prevalence',
        y='Behavior',
        orientation='h',
        text=chart_df['Prevalence'].apply(lambda x: f'{x:.1f}%'),
        color_discrete_sequence=[color]
    )
    
    fig.update_traces(textposition='outside', textfont_color='#000000')
    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font={'color': '#666666', 'size': 13},
        title={'text': f'{title} (N={n})', 'font': {'size': 16, 'color': '#666666'}},
        xaxis_title='% who engaged at least once (past 30 days)',
        yaxis_title='',
        xaxis=dict(range=[0, 100], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
        yaxis=dict(tickfont={'color': '#666666', 'size': 12}, showgrid=True, gridcolor='#e2e8f0'),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        shapes=[{
            'type': 'rect', 'xref': 'paper', 'yref': 'paper',
            'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
            'line': {'color': '#e2e8f0', 'width': 1}
        }]
    )
    
    return fig

def create_distribution_chart(df, var, title, order=None, colors=None):
    """Create bar chart showing response distribution."""
    dist = calculate_distribution(df, var, order)
    
    if dist.empty:
        return None
    
    if colors is None:
        colors = COLORS['categorical']
    
    n = dist['Count'].sum()
    
    fig = px.bar(
        dist,
        x='Category',
        y='Percentage',
        text=dist['Percentage'].apply(lambda x: f'{x:.1f}%'),
        color='Category',
        color_discrete_sequence=colors
    )
    
    fig.update_traces(textposition='outside', showlegend=False, textfont_color='#000000')
    title_text = f'{title} (N={n})' if title else f'(N={n})'
    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font={'color': '#666666', 'size': 13},
        title={'text': title_text, 'font': {'size': 16, 'color': '#666666'}},
        xaxis_title='',
        yaxis_title='Percentage',
        xaxis=dict(tickfont={'color': '#666666', 'size': 12}, showgrid=True, gridcolor='#e2e8f0'),
        yaxis=dict(range=[0, max(dist['Percentage']) * 1.25], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        shapes=[{
            'type': 'rect', 'xref': 'paper', 'yref': 'paper',
            'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
            'line': {'color': '#e2e8f0', 'width': 1}
        }]
    )
    
    return fig

def create_crosstab_chart(df, behavior_var, demo_var, demo_label, behavior_label, is_binary=True):
    """Create bar chart for cross-tabulation."""
    if behavior_var not in df.columns or demo_var not in df.columns:
        return None
    
    # Exclude small groups
    exclude_values = ['Prefer not to answer', 'Prefer Not to Answer']
    
    data = []
    for group in df[demo_var].dropna().unique():
        if group in exclude_values:
            continue
        subset = df[df[demo_var] == group]
        if is_binary:
            prev = calculate_prevalence_binary(subset, behavior_var)
        else:
            valid = subset[behavior_var].dropna()
            if len(valid) > 0:
                prev = (valid[valid != 'Never'].count() / len(valid)) * 100
            else:
                prev = None
        if prev is not None:
            data.append({'Group': str(group), 'Prevalence': prev, 'N': len(subset)})
    
    if not data:
        return None
    
    chart_df = pd.DataFrame(data)
    total_n = chart_df['N'].sum()
    
    fig = px.bar(
        chart_df,
        x='Group',
        y='Prevalence',
        text=chart_df['Prevalence'].apply(lambda x: f'{x:.1f}%'),
        color_discrete_sequence=[COLORS['primary']]
    )
    
    fig.update_traces(textposition='outside', textfont_color='#000000')
    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font={'color': '#666666', 'size': 13},
        title={'text': f'{behavior_label} by {demo_label} (N={total_n})', 'font': {'size': 16, 'color': '#666666'}},
        xaxis_title=demo_label,
        yaxis_title='% engaged at least once',
        xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
        yaxis=dict(range=[0, 100], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        shapes=[{
            'type': 'rect', 'xref': 'paper', 'yref': 'paper',
            'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
            'line': {'color': '#e2e8f0', 'width': 1}
        }]
    )
    
    return fig

def create_stacked_perception_chart(df, var_dict, title, order, colors):
    """Create stacked horizontal bar chart for perception items - COLLAPSED into binary categories."""
    data = []
    
    # Determine if this is danger or likelihood based on order
    is_danger = 'Very dangerous' in order
    
    for var, label in var_dict.items():
        if var not in df.columns:
            continue
            
        counts = df[var].value_counts()
        total = counts.sum()
        
        if is_danger:
            dangerous = counts.get('Very dangerous', 0) + counts.get('Somewhat dangerous', 0)
            safe = counts.get('Very safe', 0) + counts.get('Somewhat safe', 0)
            unsure = counts.get('Unsure', 0)
            valid_total = dangerous + safe + unsure
            
            if valid_total > 0:
                data.append({'Behavior': label, 'Response': 'Dangerous', 'Percentage': dangerous/valid_total*100, 'Order': 1})
                data.append({'Behavior': label, 'Response': 'Unsure', 'Percentage': unsure/valid_total*100, 'Order': 2})
                data.append({'Behavior': label, 'Response': 'Safe', 'Percentage': safe/valid_total*100, 'Order': 3})
        else:
            likely = counts.get('Very likely', 0) + counts.get('Somewhat likely', 0)
            unlikely = counts.get('Very unlikely', 0) + counts.get('Somewhat unlikely', 0)
            unsure = counts.get('Unsure', 0)
            valid_total = likely + unlikely + unsure
            
            if valid_total > 0:
                data.append({'Behavior': label, 'Response': 'Likely', 'Percentage': likely/valid_total*100, 'Order': 1})
                data.append({'Behavior': label, 'Response': 'Unsure', 'Percentage': unsure/valid_total*100, 'Order': 2})
                data.append({'Behavior': label, 'Response': 'Unlikely', 'Percentage': unlikely/valid_total*100, 'Order': 3})
    
    if not data:
        return None
    
    chart_df = pd.DataFrame(data)
    
    if is_danger:
        response_order = ['Safe', 'Unsure', 'Dangerous']
        color_map = {'Dangerous': COLORS['warm'], 'Unsure': COLORS['neutral'], 'Safe': COLORS['primary']}
    else:
        response_order = ['Unlikely', 'Unsure', 'Likely']
        color_map = {'Likely': COLORS['warm'], 'Unsure': COLORS['neutral'], 'Unlikely': COLORS['primary']}
    
    chart_df['Response'] = pd.Categorical(chart_df['Response'], categories=response_order, ordered=True)
    
    n = len(df)
    
    fig = px.bar(
        chart_df,
        x='Percentage',
        y='Behavior',
        color='Response',
        orientation='h',
        color_discrete_map=color_map,
        category_orders={'Response': response_order}
    )
    
    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font={'color': '#666666', 'size': 13},
        title={'text': f'{title} (N={n})', 'font': {'size': 16, 'color': '#666666'}},
        xaxis_title='Percentage',
        yaxis_title='',
        xaxis=dict(range=[0, 100], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
        yaxis=dict(tickfont={'color': '#666666', 'size': 12}),
        height=450,
        margin=dict(l=20, r=20, t=50, b=80),
        barmode='stack',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5,
            font={'color': '#666666', 'size': 12},
            title_text=''
        ),
        shapes=[{
            'type': 'rect', 'xref': 'paper', 'yref': 'paper',
            'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
            'line': {'color': '#e2e8f0', 'width': 1}
        }]
    )
    
    return fig

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create binary gender labels
    if 'genderbin' in df.columns:
        df['gender_label'] = df['genderbin'].map({0: 'Male', 1: 'Female'})
    
    # Create binary enrollment labels
    if 'enrollbin' in df.columns:
        df['enroll_label'] = df['enrollbin'].map({0: 'Not Enrolled', 1: 'Enrolled'})
    
    # Create binary driver labels
    if 'drivebin' in df.columns:
        df['driver_label'] = df['drivebin'].map({0: 'No', 1: 'Yes'})
    
    # ==========================================================================
    # SIDEBAR FILTERS
    # ==========================================================================
    with st.sidebar:
        st.title("Ohio Traffic Safety")
        st.markdown("### Filters")
        
        # Age filter
        if 'age' in df.columns:
            ages = sorted(df['age'].dropna().unique())
            selected_ages = st.multiselect("Age", options=ages, default=ages)
        else:
            selected_ages = None
        
        # Gender filter
        if 'gender_label' in df.columns:
            genders = df['gender_label'].dropna().unique().tolist()
            selected_genders = st.multiselect("Gender", options=genders, default=genders)
        else:
            selected_genders = None
        
        # City filter
        if 'city' in df.columns:
            cities = df['city'].dropna().unique().tolist()
            selected_cities = st.multiselect("City", options=cities, default=cities)
        else:
            selected_cities = None
        
        # Apply filters
        filtered_df = df.copy()
        if selected_ages:
            filtered_df = filtered_df[filtered_df['age'].isin(selected_ages)]
        if selected_genders:
            filtered_df = filtered_df[filtered_df['gender_label'].isin(selected_genders)]
        if selected_cities:
            filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]
        
        st.markdown("---")
        st.metric("Filtered Sample", len(filtered_df))
        st.metric("Total Sample", len(df))
    
    # ==========================================================================
    # MAIN CONTENT - TABS
    # ==========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Demographics", "Risky Driving", "Danger Perceptions", 
        "Enforcement", "Peer Norms", "Micromobility"
    ])
    
    # =========================================================================
    # TAB 1: DEMOGRAPHICS
    # =========================================================================
    with tab1:
        st.header("Demographics Overview")
        st.markdown(f"**{len(filtered_df)} respondents** based on current filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            if 'age' in filtered_df.columns:
                age_dist = filtered_df['age'].value_counts().sort_index()
                age_pct = (age_dist / age_dist.sum() * 100)
                fig_age = px.bar(
                    x=age_pct.index.astype(int),
                    y=age_pct.values,
                    labels={'x': 'Age', 'y': 'Percentage'},
                    text=age_pct.apply(lambda x: f'{x:.1f}%'),
                    color_discrete_sequence=[COLORS['primary']]
                )
                fig_age.update_traces(textposition='outside', textfont_color='#000000')
                fig_age.update_layout(
                    title={'text': f'Age Distribution (N={len(filtered_df)})', 'font': {'size': 16, 'color': '#666666'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'color': '#666666', 'size': 13},
                    xaxis=dict(tickmode='linear', tick0=18, dtick=1, tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
                    yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(age_pct.values) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
                )
                st.plotly_chart(fig_age, use_container_width=True)
            
            # City
            if 'city' in filtered_df.columns:
                city_dist = filtered_df['city'].value_counts()
                city_pct = (city_dist / city_dist.sum() * 100)
                fig_city = px.bar(
                    x=city_pct.index,
                    y=city_pct.values,
                    labels={'x': 'City', 'y': 'Percentage'},
                    text=city_pct.apply(lambda x: f'{x:.1f}%'),
                    color_discrete_sequence=[COLORS['primary']]
                )
                fig_city.update_traces(textposition='outside', textfont_color='#000000')
                fig_city.update_layout(
                    title={'text': f'City (N={len(filtered_df)})', 'font': {'size': 16, 'color': '#666666'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'color': '#666666', 'size': 13},
                    xaxis=dict(tickfont={'color': '#666666', 'size': 10}, title={'text': ''}, tickangle=-45, showgrid=True, gridcolor='#e2e8f0'),
                    yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(city_pct.values) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=120),
                    shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
                )
                st.plotly_chart(fig_city, use_container_width=True)
        
        with col2:
            # Gender
            if 'gender_label' in filtered_df.columns:
                gender_dist = filtered_df['gender_label'].value_counts()
                gender_pct = (gender_dist / gender_dist.sum() * 100)
                fig_gender = px.bar(
                    x=gender_pct.index,
                    y=gender_pct.values,
                    labels={'x': 'Gender', 'y': 'Percentage'},
                    text=gender_pct.apply(lambda x: f'{x:.1f}%'),
                    color_discrete_sequence=[COLORS['primary']]
                )
                fig_gender.update_traces(textposition='outside', textfont_color='#000000')
                fig_gender.update_layout(
                    title={'text': f'Gender (N={gender_dist.sum()})', 'font': {'size': 16, 'color': '#666666'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'color': '#666666', 'size': 13},
                    xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'text': ''}, showgrid=True, gridcolor='#e2e8f0'),
                    yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(gender_pct.values) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
                )
                st.plotly_chart(fig_gender, use_container_width=True)
            
            # Enrollment
            if 'enroll_label' in filtered_df.columns:
                enroll_dist = filtered_df['enroll_label'].value_counts()
                enroll_pct = (enroll_dist / enroll_dist.sum() * 100)
                fig_enroll = px.bar(
                    x=enroll_pct.index,
                    y=enroll_pct.values,
                    labels={'x': 'Enrollment', 'y': 'Percentage'},
                    text=enroll_pct.apply(lambda x: f'{x:.1f}%'),
                    color_discrete_sequence=[COLORS['primary']]
                )
                fig_enroll.update_traces(textposition='outside', textfont_color='#000000')
                fig_enroll.update_layout(
                    title={'text': f'College Enrollment (N={enroll_dist.sum()})', 'font': {'size': 16, 'color': '#666666'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'color': '#666666', 'size': 13},
                    xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'text': ''}, showgrid=True, gridcolor='#e2e8f0'),
                    yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(enroll_pct.values) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
                )
                st.plotly_chart(fig_enroll, use_container_width=True)
        
        # Race/ethnicity
        st.markdown("*Respondents could select multiple race/ethnicity categories*")
        
        race_data = []
        for var, label in RACE_VARS.items():
            if var in filtered_df.columns:
                pct = (filtered_df[var].sum() / len(filtered_df) * 100)
                race_data.append({'Race/Ethnicity': label, 'Percentage': pct})
        
        if race_data:
            race_df = pd.DataFrame(race_data).sort_values('Percentage', ascending=True)
            fig_race = px.bar(
                race_df,
                x='Percentage',
                y='Race/Ethnicity',
                orientation='h',
                text=race_df['Percentage'].apply(lambda x: f'{x:.1f}%'),
                color_discrete_sequence=[COLORS['primary']]
            )
            fig_race.update_traces(textposition='outside', textfont_color='#000000')
            fig_race.update_layout(
                title={'text': f'Race/Ethnicity (N={len(filtered_df)})', 'font': {'size': 16, 'color': '#666666'}},
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font={'color': '#666666', 'size': 13},
                xaxis=dict(range=[0, 80], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
                yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'text': ''}, showgrid=True, gridcolor='#e2e8f0'),
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
            )
            st.plotly_chart(fig_race, use_container_width=True)
        
        # Driver status
        col3, col4 = st.columns(2)
        with col3:
            if 'driver_label' in filtered_df.columns:
                drive_dist = filtered_df['driver_label'].value_counts()
                drive_pct = (drive_dist / drive_dist.sum() * 100)
                fig_drive = px.bar(
                    x=drive_pct.index,
                    y=drive_pct.values,
                    labels={'x': 'Driven', 'y': 'Percentage'},
                    text=drive_pct.apply(lambda x: f'{x:.1f}%'),
                    color_discrete_sequence=[COLORS['primary']]
                )
                fig_drive.update_traces(textposition='outside', textfont_color='#000000')
                fig_drive.update_layout(
                    title={'text': f'Driven in Last 30 Days (N={drive_dist.sum()})', 'font': {'size': 16, 'color': '#666666'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'color': '#666666', 'size': 13},
                    xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'text': ''}, showgrid=True, gridcolor='#e2e8f0'),
                    yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(drive_pct.values) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
                )
                st.plotly_chart(fig_drive, use_container_width=True)
    
    # =========================================================================
    # TAB 2: RISKY DRIVING
    # =========================================================================
    with tab2:
        st.header("Risky Driving Behavior Prevalence")
        
        # Filter to drivers only
        drivers_df = filtered_df[filtered_df['drivebin'] == 1] if 'drivebin' in filtered_df.columns else filtered_df
        st.markdown(f"**{len(drivers_df)} drivers** based on current filters")
        
        fig_behavior = create_prevalence_chart(
            drivers_df, BEHAVIOR_VARS,
            'Risky Driving Behaviors',
            color=COLORS['primary'],
            is_binary=True
        )
        if fig_behavior:
            st.plotly_chart(fig_behavior, use_container_width=True)
        
        st.subheader("Cross-tabulation")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_behavior = st.selectbox(
                "Select behavior:",
                options=list(BEHAVIOR_VARS.keys()),
                format_func=lambda x: BEHAVIOR_VARS[x],
                key='behavior_select'
            )
        with col2:
            demo_options = {'gender_label': 'Gender', 'enroll_label': 'Enrollment Status'}
            selected_demo = st.selectbox(
                "Cross-tab by:",
                options=list(demo_options.keys()),
                format_func=lambda x: demo_options[x],
                key='demo_select'
            )
        
        fig_crosstab = create_crosstab_chart(
            drivers_df, selected_behavior, selected_demo,
            demo_options[selected_demo], BEHAVIOR_VARS[selected_behavior],
            is_binary=True
        )
        if fig_crosstab:
            st.plotly_chart(fig_crosstab, use_container_width=True)
    
    # =========================================================================
    # TAB 3: DANGER PERCEPTIONS
    # =========================================================================
    with tab3:
        st.header("Danger Perceptions")
        st.markdown(f"**{len(filtered_df)} respondents** | *How dangerous do you feel it is to...*")
        
        fig_danger = create_stacked_perception_chart(
            filtered_df, DANGER_VARS,
            'Perceived Danger of Driving Behaviors',
            DANGER_ORDER, COLORS['diverging']
        )
        if fig_danger:
            st.plotly_chart(fig_danger, use_container_width=True)
        
        st.subheader("Cross-tabulation")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_danger = st.selectbox(
                "Select item:",
                options=list(DANGER_VARS.keys()),
                format_func=lambda x: DANGER_VARS[x],
                key='danger_select'
            )
        with col2:
            demo_options = {'gender_label': 'Gender', 'enroll_label': 'Enrollment Status'}
            selected_demo_danger = st.selectbox(
                "Cross-tab by:",
                options=list(demo_options.keys()),
                format_func=lambda x: demo_options[x],
                key='demo_danger_select'
            )
        
        # Calculate % rating as dangerous by demographic
        if selected_danger in filtered_df.columns and selected_demo_danger in filtered_df.columns:
            data = []
            for group in filtered_df[selected_demo_danger].dropna().unique():
                if group in ['Prefer not to answer']:
                    continue
                subset = filtered_df[filtered_df[selected_demo_danger] == group]
                danger_count = subset[selected_danger].isin(['Very dangerous', 'Somewhat dangerous']).sum()
                valid_count = subset[selected_danger].notna().sum()
                if valid_count > 0:
                    pct = (danger_count / valid_count) * 100
                    data.append({'Group': str(group), 'Pct': pct, 'N': valid_count})
            
            if data:
                chart_df = pd.DataFrame(data)
                total_n = chart_df['N'].sum()
                fig = px.bar(
                    chart_df, x='Group', y='Pct',
                    text=chart_df['Pct'].apply(lambda x: f'{x:.1f}%'),
                    color_discrete_sequence=[COLORS['warm']]
                )
                fig.update_traces(textposition='outside', textfont_color='#000000')
                fig.update_layout(
                    title={'text': f'% Rating as Dangerous: {DANGER_VARS[selected_danger]} (N={total_n})', 'font': {'size': 16, 'color': '#666666'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'color': '#666666', 'size': 13},
                    xaxis_title=demo_options[selected_demo_danger],
                    yaxis_title='% rating as dangerous',
                    xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
                    yaxis=dict(range=[0, 100], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 4: ENFORCEMENT PERCEPTIONS
    # =========================================================================
    with tab4:
        st.header("Enforcement Risk Perceptions")
        st.markdown(f"**{len(filtered_df)} respondents** | *How likely is it you'd be pulled over while...*")
        
        fig_legal = create_stacked_perception_chart(
            filtered_df, LEGAL_VARS,
            'Perceived Likelihood of Being Pulled Over',
            LIKELIHOOD_ORDER, COLORS['diverging']
        )
        if fig_legal:
            st.plotly_chart(fig_legal, use_container_width=True)
        
        st.subheader("Cross-tabulation")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_legal = st.selectbox(
                "Select item:",
                options=list(LEGAL_VARS.keys()),
                format_func=lambda x: LEGAL_VARS[x],
                key='legal_select'
            )
        with col2:
            demo_options = {'gender_label': 'Gender', 'enroll_label': 'Enrollment Status'}
            selected_demo_legal = st.selectbox(
                "Cross-tab by:",
                options=list(demo_options.keys()),
                format_func=lambda x: demo_options[x],
                key='demo_legal_select'
            )
        
        # Calculate % rating as likely by demographic
        if selected_legal in filtered_df.columns and selected_demo_legal in filtered_df.columns:
            data = []
            for group in filtered_df[selected_demo_legal].dropna().unique():
                if group in ['Prefer not to answer']:
                    continue
                subset = filtered_df[filtered_df[selected_demo_legal] == group]
                likely_count = subset[selected_legal].isin(['Very likely', 'Somewhat likely']).sum()
                valid_count = subset[selected_legal].notna().sum()
                if valid_count > 0:
                    pct = (likely_count / valid_count) * 100
                    data.append({'Group': str(group), 'Pct': pct, 'N': valid_count})
            
            if data:
                chart_df = pd.DataFrame(data)
                total_n = chart_df['N'].sum()
                fig = px.bar(
                    chart_df, x='Group', y='Pct',
                    text=chart_df['Pct'].apply(lambda x: f'{x:.1f}%'),
                    color_discrete_sequence=[COLORS['primary']]
                )
                fig.update_traces(textposition='outside', textfont_color='#000000')
                fig.update_layout(
                    title={'text': f'% Rating as Likely: {LEGAL_VARS[selected_legal]} (N={total_n})', 'font': {'size': 16, 'color': '#666666'}},
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'color': '#666666', 'size': 13},
                    xaxis_title=demo_options[selected_demo_legal],
                    yaxis_title='% rating as likely',
                    xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
                    yaxis=dict(range=[0, 100], tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, showgrid=True, gridcolor='#e2e8f0'),
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 5: PEER NORMS
    # =========================================================================
    with tab5:
        st.header("Peer Behavior Perceptions")
        st.markdown(f"**{len(filtered_df)} respondents** | *How often do you think people your age...*")
        
        fig_norms = create_prevalence_chart(
            filtered_df, NORMS_VARS,
            'Descriptive Norms: % Believing Peers Engaged in Behavior (At least once)',
            color=COLORS['neutral'],
            is_binary=False
        )
        if fig_norms:
            st.plotly_chart(fig_norms, use_container_width=True)
    
    # =========================================================================
    # TAB 6: MICROMOBILITY
    # =========================================================================
    with tab6:
        st.header("Micromobility")
        st.markdown("*Overall prevalence only — small samples prevent demographic cross-tabs*")
        
        # Count riders
        bikers = filtered_df[filtered_df['bike'].notna()] if 'bike' in filtered_df.columns else pd.DataFrame()
        scooters = filtered_df[filtered_df['scoot'].notna()] if 'scoot' in filtered_df.columns else pd.DataFrame()
        skaters = filtered_df[filtered_df['skate'].notna()] if 'skate' in filtered_df.columns else pd.DataFrame()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Bicyclists", len(bikers))
        col2.metric("Scooter Riders", len(scooters))
        col3.metric("Skateboarders", len(skaters))
        
        st.markdown("---")
        
        # Response orders for helmet/electric charts
        HELMET_ORDER = ['All', 'Half', 'A Few', 'None', 'Prefer not to answer']
        ELECTRIC_ORDER = ['All', 'Half', 'A Few', 'None', 'Prefer not to answer']
        
        # BIKING
        if len(bikers) > 0:
            st.subheader("Bicycling")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = create_distribution_chart(bikers, 'bikehelmet', 'Helmet Use', HELMET_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_distribution_chart(bikers, 'electricbike', 'Electric Bike Use', ELECTRIC_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            bike_vars = {'bikealc': '2 hrs after 3+ drinks', 'bikecann5': '5 hrs after cannabis', 'bikecann9': '9 hrs after cannabis', 'bikesim': '2 hrs after both', 'biketext': 'While manually using phone'}
            fig = create_prevalence_chart(bikers, bike_vars, 'Impaired Biking Prevalence', COLORS['primary'], is_binary=False)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # SCOOTERING
        if len(scooters) > 0:
            st.subheader("Scootering")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = create_distribution_chart(scooters, 'scoothelmet', 'Helmet Use', HELMET_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_distribution_chart(scooters, 'escoot', 'Electric Scooter Use', ELECTRIC_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            scoot_vars = {'alcscoot': '2 hrs after 3+ drinks', 'cann5scoot': '5 hrs after cannabis', 'cann9scoot': '9 hrs after cannabis', 'simscoot': '2 hrs after both', 'textscoot': 'While manually using phone'}
            fig = create_prevalence_chart(scooters, scoot_vars, 'Impaired Scootering Prevalence', COLORS['accent'], is_binary=False)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # SKATEBOARDING
        if len(skaters) > 0:
            st.subheader("Skateboarding")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = create_distribution_chart(skaters, 'skatehelmet', 'Helmet Use', HELMET_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_distribution_chart(skaters, 'electricskate', 'Electric Skateboard Use', ELECTRIC_ORDER, COLORS['categorical'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            skate_vars = {'skatealc': '2 hrs after 3+ drinks', 'skatecann5': '5 hrs after cannabis', 'skatecann9': '9 hrs after cannabis', 'skatesim': '2 hrs after both', 'skatetext': 'While manually using phone'}
            fig = create_prevalence_chart(skaters, skate_vars, 'Impaired Skateboarding Prevalence', COLORS['neutral'], is_binary=False)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Negative riding experience
        st.subheader("Negative Riding Experiences")
        if 'negativeriding' in filtered_df.columns:
            neg_dist = filtered_df['negativeriding'].value_counts()
            neg_pct = (neg_dist / neg_dist.sum() * 100)
            
            fig_neg = px.bar(
                x=neg_pct.index,
                y=neg_pct.values,
                labels={'x': 'Response', 'y': 'Percentage'},
                text=neg_pct.apply(lambda x: f'{x:.1f}%'),
                color_discrete_sequence=[COLORS['primary']]
            )
            fig_neg.update_traces(textposition='outside', textfont_color='#000000')
            fig_neg.update_layout(
                title={'text': f'Experienced Negative/Dangerous Event While Riding (N={neg_dist.sum()})', 'font': {'size': 16, 'color': '#666666'}},
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font={'color': '#666666', 'size': 13},
                xaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'text': ''}, showgrid=True, gridcolor='#e2e8f0'),
                yaxis=dict(tickfont={'color': '#666666', 'size': 12}, title={'font': {'color': '#666666'}}, range=[0, max(neg_pct.values) * 1.2], showgrid=True, gridcolor='#e2e8f0'),
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                shapes=[{'type': 'rect', 'xref': 'paper', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'line': {'color': '#e2e8f0', 'width': 1}}]
            )
            st.plotly_chart(fig_neg, use_container_width=True)

if __name__ == "__main__":
    main()
