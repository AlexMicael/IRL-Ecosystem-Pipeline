import numpy as np
import pandas as pd
import altair as alt
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt

from scipy import stats
from datetime import timedelta
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import CountVectorizer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IRL Streaming Ecosystem Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING  ---
@st.cache_data
def load_data():
    """Loads and pre-processes all datasets."""
    try:
        twitch_df = pd.read_csv("twitch_streams_data.csv")
        twitch_df['collection_timestamp'] = pd.to_datetime(twitch_df['collection_timestamp'], utc=True).dt.tz_localize(None)
        users_df = pd.read_csv("twitch_users_data.csv")
        videos_df = pd.read_csv("youtube_videos_data.csv")
        videos_df['published_at'] = pd.to_datetime(videos_df['published_at'], utc=True).dt.tz_localize(None)
        comments_df = pd.read_csv("youtube_comments_data.csv")
        comments_df['published_at'] = pd.to_datetime(comments_df['published_at'], utc=True).dt.tz_localize(None)
        comments_df['toxicity_score'] = pd.to_numeric(comments_df['toxicity_score'], errors='coerce')
        map_df = pd.read_csv("streamer_map.csv")
        
        return twitch_df, users_df, videos_df, comments_df, map_df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Please ensure CSV files are in the same directory.")
        return None, None, None, None, None

twitch_df, users_df, videos_df, comments_df, map_df = load_data()

if twitch_df is None:
    st.stop()

# --- HELPER FUNCTION: STATS FORMATTER ---
def display_stats(r, rho, slope, r2, p_val):
    """Helper to display regression stats nicely."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pearson (r)", f"{r:.4f}", help="Linear correlation (-1 to 1)")
    c2.metric("Spearman (rho)", f"{rho:.4f}", help="Rank correlation (-1 to 1)")
    c3.metric("Regression Slope", f"{slope:.4f}", help="Elasticity/Rate of change")
    c4.metric("R-Squared", f"{r2:.4f}", help="Variance explained (0 to 1)")
    
    if p_val < 0.001:
        p_text = "< 0.001 (Significant)"
        color = "green"
    else:
        p_text = f"{p_val:.4f} (Not Significant)"
        color = "red"
    st.caption(f"**Statistical Significance (P-value):** :{color}[{p_text}]")

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Home", "RQ1: Temporal Toxicity", "RQ2: Cross-Platform Predictor", "RQ3: Content Themes"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project 3: Interactive Dashboard**\n\n"
    "This tool analyzes the cross-platform ecosystem of IRL streamers on Twitch and YouTube."
)

# --- PAGE 1: DASHBOARD HOME ---
if page == "Dashboard Home":
    st.title("IRL Streaming Ecosystem Dashboard")
    st.markdown("""
    Welcome to the interactive data explorer for our CS 415 Project. This dashboard provides a live view into our collected dataset
    and allows you to explore the research questions defined in Project 2.
    """)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Twitch Snapshots", f"{len(twitch_df):,}")
    col2.metric("Unique Streamers", f"{len(users_df):,}")
    col3.metric("YouTube Videos", f"{len(videos_df):,}")
    col4.metric("YouTube Comments", f"{len(comments_df):,}")
    
    # Daily Collection Volume
    st.subheader("Daily Data Collection Volume")
    
    # Aggregate Daily Counts
    twitch_daily = twitch_df.set_index('collection_timestamp').resample('D').size().reset_index(name='count')
    twitch_daily['source'] = 'Twitch Snapshots'
    twitch_daily.rename(columns={'collection_timestamp': 'date'}, inplace=True)

    videos_daily = videos_df.set_index('published_at').resample('D').size().reset_index(name='count')
    videos_daily['source'] = 'YouTube Videos'
    videos_daily.rename(columns={'published_at': 'date'}, inplace=True)
    
    comments_daily = comments_df.set_index('published_at').resample('D').size().reset_index(name='count')
    comments_daily['source'] = 'YouTube Comments'
    comments_daily.rename(columns={'published_at': 'date'}, inplace=True)
    
    combined_daily = pd.concat([twitch_daily, videos_daily, comments_daily])
    
    # Date Filter for Main Graph
    if not combined_daily.empty:
        combined_daily['date'] = pd.to_datetime(combined_daily['date'])
        combined_daily['date_only'] = combined_daily['date'].dt.date
        
        min_date = combined_daily['date_only'].min()
        max_date = combined_daily['date_only'].max()

        default_start_date = max_date - timedelta(days=90)
        if default_start_date < min_date:
            default_start_date = min_date
        
        date_range = st.date_input(
            "Filter Date Range",
            value=(default_start_date, max_date), 
            min_value=min_date,
            max_value=max_date,
            key="home_date_filter"
        )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            mask = (combined_daily['date_only'] >= start_date) & (combined_daily['date_only'] <= end_date)
            chart_data = combined_daily.loc[mask]
        else:
            chart_data = combined_daily
            
        chart = alt.Chart(chart_data).mark_bar().encode(
            x='date:T',
            y='count:Q',
            color='source:N',
            tooltip=['date', 'count', 'source']
        ).interactive()
        
        st.altair_chart(chart, width='stretch')
    else:
        st.info("No data available to display.")

    # Dataset Overview
    st.markdown("---")
    st.header("General Dataset Overview")
    st.markdown("Below are key findings from our exploratory analysis (Project 2), providing context for the specific RQs.")

    st.subheader("Toxicity Distribution (Histogram)")
    
    # Toxicity Histogram
    if not comments_df.empty:
        hist_data = comments_df['toxicity_score'].value_counts(bins=50, sort=False).reset_index()
        hist_data.columns = ['range', 'count']
        hist_data['toxicity_score'] = hist_data['range'].apply(lambda x: x.left)
        
        hist_chart = alt.Chart(hist_data).mark_bar().encode(
            x=alt.X("toxicity_score:Q", bin=alt.Bin(maxbins=50), title="Toxicity Score"),
            y=alt.Y('count:Q', title='Count'),
            tooltip=['toxicity_score', 'count']
        ).properties(title="Distribution of Comment Toxicity (Binned)")
        
        st.altair_chart(hist_chart, width='stretch')
    else:
        st.warning("No comment data for toxicity histogram.")

    st.subheader("Cross-Platform Engagement (Scatter Plot)")
    if not twitch_df.empty and not comments_df.empty:
        avg_twitch = twitch_df.groupby('user_id')['viewer_count'].mean().reset_index()
        comments_per_video = comments_df.groupby('video_id').size().reset_index(name='comment_count')
        video_counts = pd.merge(videos_df, comments_per_video, on='video_id')
        avg_yt = video_counts.groupby('channel_id')['comment_count'].mean().reset_index()
        
        # Merge
        user_map_mini = pd.merge(users_df[['user_id', 'login_name', 'display_name']], map_df, left_on='login_name', right_on='twitch_login_name')
        
        merged_metrics = pd.merge(avg_twitch, user_map_mini, on='user_id')
        merged_metrics = pd.merge(merged_metrics, avg_yt, left_on='youtube_channel_id', right_on='channel_id')
        
        if not merged_metrics.empty:
            scatter_chart = alt.Chart(merged_metrics).mark_circle(size=60).encode(
                x=alt.X('viewer_count', scale=alt.Scale(type='log', nice=True), title='Avg Twitch Viewers (Log)'),
                y=alt.Y('comment_count', scale=alt.Scale(type='log', nice=True), title='Avg YouTube Comments (Log)'),
                tooltip=['display_name', 'viewer_count', 'comment_count']
            ).properties(title="Twitch Viewership vs. YouTube Engagement").interactive()
            
            st.altair_chart(scatter_chart, width='stretch')
        else:
            st.warning("Insufficient overlap data for scatter plot.")

    st.subheader("Top Content Keywords")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Top Twitch Keywords**")
        if not twitch_df.empty:
            stop_words = "english"
            vec = CountVectorizer(stop_words=stop_words, max_features=10)
            try:
                bow = vec.fit_transform(twitch_df['stream_title'].dropna().astype(str))
                word_counts = pd.DataFrame({'word': vec.get_feature_names_out(), 'count': bow.toarray().sum(axis=0)})
                word_counts = word_counts.sort_values('count', ascending=False)
                
                bar_twitch = alt.Chart(word_counts).mark_bar().encode(
                    x=alt.X('count', title='Frequency'),
                    y=alt.Y('word', sort='-x', title='Keyword'),
                    tooltip=['word', 'count']
                )
                st.altair_chart(bar_twitch, width='stretch')
            except ValueError:
                st.info("Not enough text data for Twitch analysis.")

    with col_b:
        st.markdown("**Top YouTube Keywords (Video Titles)**")
        if not videos_df.empty:
            vec_yt = CountVectorizer(stop_words="english", max_features=10)
            try:
                bow_yt = vec_yt.fit_transform(videos_df['video_title'].dropna().astype(str))
                word_counts_yt = pd.DataFrame({'word': vec_yt.get_feature_names_out(), 'count': bow_yt.toarray().sum(axis=0)})
                word_counts_yt = word_counts_yt.sort_values('count', ascending=False)
                
                bar_yt = alt.Chart(word_counts_yt).mark_bar(color='red').encode(
                    x=alt.X('count', title='Frequency'),
                    y=alt.Y('word', sort='-x', title='Keyword'),
                    tooltip=['word', 'count']
                )
                st.altair_chart(bar_yt, width='stretch')
            except ValueError:
                st.info("Not enough text data for YouTube analysis.")

# --- PAGE 2: RQ1 (TEMPORAL TOXICITY) ---
elif page == "RQ1: Temporal Toxicity":
    st.title("RQ1: Temporal Evolution of Toxicity")
    st.markdown("**Question:** How does toxicity evolve over time in response to creator-specific events?")

    merged_map = pd.merge(map_df, users_df, left_on='twitch_login_name', right_on='login_name')
    creator = st.sidebar.selectbox("Select Creator", merged_map['display_name'].unique())
    
    channel_id = merged_map[merged_map['display_name'] == creator]['youtube_channel_id'].values[0]
    creator_vids = videos_df[videos_df['channel_id'] == channel_id]['video_id']
    df = comments_df[comments_df['video_id'].isin(creator_vids)].copy()
    
    if not df.empty:
        # Aggregation
        granularity = st.sidebar.select_slider("Granularity", ["D", "W", "M"], value="W")
        timeline = df.set_index('published_at').resample(granularity)['toxicity_score'].agg(['mean', 'count']).reset_index()
        timeline.rename(columns={'mean': 'avg_toxicity', 'count': 'comment_volume'}, inplace=True) # Rename for clarity

        # Statistics
        mean_tox = df['toxicity_score'].mean()
        max_tox = timeline['avg_toxicity'].max()
        std_tox = df['toxicity_score'].std()
        
        s1, s2, s3 = st.columns(3)
        s1.metric("Baseline Toxicity", f"{mean_tox:.3f}")
        s2.metric("Max Spike", f"{max_tox:.3f}", delta=f"{(max_tox-mean_tox):.3f} above avg", delta_color="inverse")
        s3.metric("Volatility (Std Dev)", f"{std_tox:.3f}")
        
        # Chart
        st.subheader(f"Toxicity Trends for {creator}")
        base = alt.Chart(timeline.reset_index()).encode(x='published_at:T')
        line = base.mark_line(color='red').encode(
            y=alt.Y('avg_toxicity', title='Avg Toxicity'),
            tooltip=['published_at', 'avg_toxicity']
        )
        bar = base.mark_bar(opacity=0.3).encode(
            y=alt.Y('comment_volume', title='Volume'),
             tooltip=['published_at', 'comment_volume']
        )
        st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent').interactive(), width='stretch')

        # Insights
        st.markdown("### Insights")
        st.write(f"Showing analysis for **{len(df)}** comments.")
        
        st.subheader("Most Toxic Periods")
        toxic_periods = timeline.sort_values('avg_toxicity', ascending=False).head(5)
        st.dataframe(toxic_periods)

# --- PAGE 3: RQ2 - CROSS-PLATFORM PREDICTOR ---
elif page == "RQ2: Cross-Platform Predictor":
    st.title("RQ2: Cross-Platform Engagement Prediction")
    st.markdown("**Question:** Do Twitch metrics (viewers, duration) predict YouTube outcomes?")
    
    st.sidebar.subheader("Regression Parameters")
    
    # Aggregate Twitch Metrics per User
    twitch_metrics = twitch_df.groupby('user_id').agg({
        'viewer_count': 'mean',
        'started_at': 'count' # Proxy for frequency/duration
    }).rename(columns={'viewer_count': 'avg_viewers', 'started_at': 'stream_count'}).reset_index()
    
    # Aggregate YouTube Metrics per Channel
    comments_with_channel = pd.merge(comments_df, videos_df[['video_id', 'channel_id']], on='video_id')
    
    yt_metrics = comments_with_channel.groupby('channel_id').agg({
        'comment_id': 'count',
        'toxicity_score': 'mean',
        'like_count': 'mean'
    }).rename(columns={'comment_id': 'total_comments', 'toxicity_score': 'avg_channel_toxicity', 'like_count': 'avg_comment_likes'}).reset_index()
    
    # Map Twitch User ID to YouTube Channel ID
    user_map = pd.merge(users_df[['user_id', 'login_name', 'display_name']], map_df, left_on='login_name', right_on='twitch_login_name')
    
    # Merge
    combined_metrics = pd.merge(twitch_metrics, user_map, on='user_id')
    combined_metrics = pd.merge(combined_metrics, yt_metrics, left_on='youtube_channel_id', right_on='channel_id')
    
    if combined_metrics.empty:
        st.error("Insufficient overlapping data between Twitch and YouTube to generate plot.")
    else:
        # Controls
        x_metric = st.sidebar.selectbox("X-Axis (Twitch)", ["avg_viewers", "stream_count"])
        y_metric = st.sidebar.selectbox("Y-Axis (YouTube)", ["total_comments", "avg_channel_toxicity", "avg_comment_likes"])
        
        valid_data = combined_metrics[[x_metric, y_metric]].dropna()
        
        if len(valid_data) < 3:
            st.error(f"Insufficient valid data for {x_metric} vs {y_metric}. Only {len(valid_data)} creators have both metrics.")
            st.info("This typically means toxicity scores are missing for most channels. Try selecting a different Y-axis metric.")
        else:
            valid_indices = valid_data.index
            plot_data = combined_metrics.loc[valid_indices]
            
            # Pearson
            pearson_r, p_p = stats.pearsonr(plot_data[x_metric], plot_data[y_metric])
            
            # Spearman
            spearman_r, s_p = stats.spearmanr(plot_data[x_metric], plot_data[y_metric])
            
            # Linear Regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[x_metric], plot_data[y_metric])
            r_squared = r_value**2
            
            display_stats(pearson_r, spearman_r, slope, r_squared, p_value)
            
            # Data Quality Info
            st.info(f"Analysis based on **{len(plot_data)}** creators with valid data for both metrics. "
                   f"({len(combined_metrics) - len(plot_data)} excluded due to missing toxicity scores)")

            # Plot
            st.subheader(f"{x_metric} vs. {y_metric}")
            
            scatter = alt.Chart(plot_data).mark_circle(size=60).encode(
                x=alt.X(x_metric, scale=alt.Scale(zero=False)),
                y=alt.Y(y_metric, scale=alt.Scale(zero=False)),
                tooltip=['display_name', x_metric, y_metric]
            ).interactive()
            
            # Regression Line
            reg_line = scatter.transform_regression(x_metric, y_metric).mark_line(color='red')
            
            st.altair_chart(scatter + reg_line, width='stretch')
            
            # Data Table
            st.markdown("### Creator Data")
            st.dataframe(plot_data[['display_name', x_metric, y_metric]].sort_values(x_metric, ascending=False))

# --- PAGE 4: RQ3 (CONTENT THEMES) ---
elif page == "RQ3: Content Themes":
    st.title("RQ3: Content Theme Analyzer")
    st.markdown("**Question:** How do specific keywords influence engagement, and what related themes emerge?")
    
    keyword = st.sidebar.text_input("Keyword", "drama").lower()
    
    df = videos_df.copy()
    df['has_keyword'] = df['video_title'].str.lower().str.contains(keyword).fillna(False)
    
    # Merge Counts
    counts = comments_df.groupby('video_id').size().reset_index(name='comments')
    df = pd.merge(df, counts, on='video_id', how='left').fillna(0)
    
    if df['has_keyword'].sum() > 0:
        group_yes = df[df['has_keyword']]['comments']
        group_no = df[~df['has_keyword']]['comments']
        
        st.markdown(f"### Statistical Test: Does '{keyword}' drive engagement?")
        
        # T-Test
        t_stat, p_val = stats.ttest_ind(group_yes, group_no, equal_var=False)
        
        m1 = group_yes.mean()
        m2 = group_no.mean()
        lift = ((m1 - m2) / m2) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Comments (With)", f"{m1:.1f}")
        c2.metric("Avg Comments (Without)", f"{m2:.1f}")
        c3.metric("Engagement Lift", f"{lift:+.1f}%", delta_color="normal")
        
        if p_val < 0.05:
            st.success(f"**Statistically Significant** (p = {p_val:.4e}). The keyword '{keyword}' has a real impact on engagement.")
        else:
            st.warning(f"**Not Significant** (p = {p_val:.4f}). The difference might be due to chance.")

        # Comparative Density Plot
        st.subheader("Engagement Distribution by Keyword Presence")
        chart_data = df[['has_keyword', 'comments']].copy()
        chart_data['Type'] = chart_data['has_keyword'].map({True: 'With Keyword', False: 'Without Keyword'})
        
        chart = alt.Chart(chart_data).transform_density(
            'comments',
            as_=['comments', 'density'],
            groupby=['Type']
        ).mark_area(opacity=0.5).encode(
            x=alt.X('comments:Q', title='Comment Count'),
            y='density:Q',
            color='Type:N'
        )
        st.altair_chart(chart, width='stretch')
        
        # Snowball Sampling
        st.subheader(f"Snowball Sampling: What co-occurs with '{keyword}'?")
        try:
            subset = df[df['has_keyword']]['video_title']
            vec = CountVectorizer(stop_words='english', max_features=15)
            bow = vec.fit_transform(subset)
            words = pd.DataFrame({'word': vec.get_feature_names_out(), 'count': bow.toarray().sum(axis=0)})
            words = words[words['word'] != keyword].sort_values('count', ascending=False)
            
            bar = alt.Chart(words).mark_bar().encode(
                x='count:Q', y=alt.Y('word:N', sort='-x')
            )
            st.altair_chart(bar, width='stretch')
        except:
            st.info("Not enough data for snowball sampling.")

        # Show Examples
        st.subheader("Example Videos")
        st.dataframe(df[df['has_keyword']][['video_title', 'video_id']].head(10))
            
    else:
        st.error("Keyword not found in any titles.")