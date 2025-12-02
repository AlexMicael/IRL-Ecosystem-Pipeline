import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from datetime import timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IRL Streaming Ecosystem Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING (Cached for Performance) ---
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

# Load data
twitch_df, users_df, videos_df, comments_df, map_df = load_data()

if twitch_df is None:
    st.stop() # Stop execution if data fails to load

# --- SIDEBAR NAVIGATION ---
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
    
    st.subheader("System Status")
    st.success(f"🟢 Data collection is ACTIVE. Last Twitch snapshot: {twitch_df['collection_timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    
    # --- 1. Daily Collection Volume (Main Live Chart) ---
    st.subheader("Daily Data Collection Volume")
    
    # Aggregate daily counts
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

        # Default to past 3 months
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

    # --- 2. Dataset Overview Context (Project 2 Findings) ---
    st.markdown("---")
    st.header("General Dataset Overview")
    st.markdown("Below are key findings from our exploratory analysis (Project 2), providing context for the specific RQs.")

    st.subheader("Toxicity Distribution (Histogram)")
    # Toxicity Histogram
    if not comments_df.empty:
        # Create 50 bins for toxicity score (0.0 to 1.0)
        hist_data = comments_df['toxicity_score'].value_counts(bins=50, sort=False).reset_index()
        hist_data.columns = ['range', 'count']
        # Extract the left edge of the bin for plotting
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
        # 1. Twitch metrics
        avg_twitch = twitch_df.groupby('user_id')['viewer_count'].mean().reset_index()
        
        # 2. YouTube metrics
        comments_per_video = comments_df.groupby('video_id').size().reset_index(name='comment_count')
        video_counts = pd.merge(videos_df, comments_per_video, on='video_id')
        avg_yt = video_counts.groupby('channel_id')['comment_count'].mean().reset_index()
        
        # 3. Merge
        # User map: Twitch login -> YouTube channel ID
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

    # Version with comment keywords as well
    # col_a, col_b, col_c = st.columns(3)
    
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
    
    # with col_c:
    #     st.markdown("**Top YouTube Keywords (Comments)**")
    #     if not comments_df.empty:
    #         vec_comments = CountVectorizer(stop_words="english", max_features=10)
    #         try:
    #             bow_comments = vec_comments.fit_transform(comments_df['comment_text'].dropna().astype(str))
    #             word_counts_comments = pd.DataFrame({'word': vec_comments.get_feature_names_out(), 'count': bow_comments.toarray().sum(axis=0)})
    #             word_counts_comments = word_counts_comments.sort_values('count', ascending=False)
                
    #             bar_comments = alt.Chart(word_counts_comments).mark_bar(color='green').encode(
    #                 x=alt.X('count', title='Frequency'),
    #                 y=alt.Y('word', sort='-x', title='Keyword'),
    #                 tooltip=['word', 'count']
    #             )
    #             st.altair_chart(bar_comments, width='stretch')
    #         except ValueError:
    #             st.info("Not enough text data for YouTube comments analysis.")

# --- PAGE 2: RQ1 - TEMPORAL TOXICITY ---
elif page == "RQ1: Temporal Toxicity":
    st.title("RQ1: Temporal Evolution of Toxicity")
    st.markdown("**Question:** How does toxicity evolve over time in response to creator-specific events?")
    
    # Sidebar Controls
    st.sidebar.subheader("Parameters")
    
    # Get list of creators
    creators = sorted(comments_df['video_id'].unique()) # Ideally map to creator names, using IDs for now if map is incomplete
    merged_map = pd.merge(map_df, users_df, left_on='twitch_login_name', right_on='login_name')
    creator_options = merged_map['display_name'].unique()
    
    selected_creator = st.sidebar.selectbox("Select Creator", creator_options)
    
    # Filter Data
    # 1. Get YouTube channel ID for this creator
    channel_id = merged_map[merged_map['display_name'] == selected_creator]['youtube_channel_id'].iloc[0]
    
    # 2. Get videos for this channel
    creator_videos = videos_df[videos_df['channel_id'] == channel_id]['video_id']
    
    # 3. Get comments for these videos
    creator_comments = comments_df[comments_df['video_id'].isin(creator_videos)].copy()
    
    if creator_comments.empty:
        st.warning(f"No comment data found for {selected_creator}.")
    else:
        # Time Granularity
        granularity = st.sidebar.select_slider("Time Granularity", options=["D", "W", "M"], value="D", format_func=lambda x: {"D":"Daily", "W":"Weekly", "M":"Monthly"}[x])
        
        # Aggregate Toxicity
        timeline_df = creator_comments.set_index('published_at').resample(granularity)['toxicity_score'].agg(['mean', 'count']).reset_index()
        timeline_df.rename(columns={'mean': 'avg_toxicity', 'count': 'comment_volume'}, inplace=True)
        
        # Plot
        st.subheader(f"Toxicity Trends for {selected_creator}")
        
        # Dual-axis chart: Line for toxicity, Bar for volume
        base = alt.Chart(timeline_df).encode(x='published_at:T')
        
        line = base.mark_line(color='red').encode(
            y=alt.Y('avg_toxicity', axis=alt.Axis(title='Average Toxicity Score', titleColor='red')),
            tooltip=['published_at', 'avg_toxicity']
        )
        
        bar = base.mark_bar(opacity=0.3).encode(
            y=alt.Y('comment_volume', axis=alt.Axis(title='Comment Volume')),
            tooltip=['published_at', 'comment_volume']
        )
        
        st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent'), width='stretch')
        
        st.markdown("### Insights")
        st.write(f"Analysis based on **{len(creator_comments)}** comments across **{len(creator_videos)}** videos.")
        
        # Highlight highly toxic days
        st.subheader("Most Toxic Periods")
        toxic_days = timeline_df.sort_values('avg_toxicity', ascending=False).head(5)
        st.dataframe(toxic_days)


# --- PAGE 3: RQ2 - CROSS-PLATFORM PREDICTOR ---
elif page == "RQ2: Cross-Platform Predictor":
    st.title("RQ2: Cross-Platform Engagement Prediction")
    st.markdown("**Question:** Do Twitch metrics (viewers, duration) predict YouTube outcomes?")
    
    st.sidebar.subheader("Regression Parameters")
    
    # --- Data Prep ---
    # 1. Aggregate Twitch Metrics per User
    twitch_metrics = twitch_df.groupby('user_id').agg({
        'viewer_count': 'mean',
        'started_at': 'count' # Proxy for frequency/duration
    }).rename(columns={'viewer_count': 'avg_viewers', 'started_at': 'stream_count'}).reset_index()
    
    # 2. Aggregate YouTube Metrics per Channel
    # Join comments to videos to get channel_id
    comments_with_channel = pd.merge(comments_df, videos_df[['video_id', 'channel_id']], on='video_id')
    
    yt_metrics = comments_with_channel.groupby('channel_id').agg({
        'comment_id': 'count',
        'toxicity_score': 'mean',
        'like_count': 'mean'
    }).rename(columns={'comment_id': 'total_comments', 'toxicity_score': 'avg_channel_toxicity', 'like_count': 'avg_comment_likes'}).reset_index()
    
    # 3. Map Twitch User ID to YouTube Channel ID
    # (Twitch ID -> Login -> Map -> Channel ID)
    user_map = pd.merge(users_df[['user_id', 'login_name', 'display_name']], map_df, left_on='login_name', right_on='twitch_login_name')
    
    # 4. Final Merge
    combined_metrics = pd.merge(twitch_metrics, user_map, on='user_id')
    combined_metrics = pd.merge(combined_metrics, yt_metrics, left_on='youtube_channel_id', right_on='channel_id')
    
    if combined_metrics.empty:
        st.error("Insufficient overlapping data between Twitch and YouTube to generate plot.")
    else:
        # Controls
        x_metric = st.sidebar.selectbox("X-Axis (Twitch)", ["avg_viewers", "stream_count"])
        y_metric = st.sidebar.selectbox("Y-Axis (YouTube)", ["total_comments", "avg_channel_toxicity", "avg_comment_likes"])
        
        # Plot
        st.subheader(f"{x_metric} vs. {y_metric}")
        
        scatter = alt.Chart(combined_metrics).mark_circle(size=60).encode(
            x=alt.X(x_metric, scale=alt.Scale(zero=False)),
            y=alt.Y(y_metric, scale=alt.Scale(zero=False)),
            tooltip=['display_name', x_metric, y_metric]
        ).interactive()
        
        # Add regression line
        reg_line = scatter.transform_regression(x_metric, y_metric).mark_line(color='red')
        
        st.altair_chart(scatter + reg_line, width='stretch')
        
        # Data Table
        st.markdown("### Creator Data")
        st.dataframe(combined_metrics[['display_name', x_metric, y_metric]].sort_values(x_metric, ascending=False))


# --- PAGE 4: RQ3 - CONTENT THEMES (With Snowball Sampling) ---
elif page == "RQ3: Content Themes":
    st.title("RQ3: Content Theme Analyzer")
    st.markdown("**Question:** How do specific keywords influence engagement, and what related themes emerge?")
    
    st.sidebar.subheader("Keyword Analysis")
    target_keyword = st.sidebar.text_input("Enter a seed keyword (e.g., 'drama', 'irl', 'ranking')", value="irl").lower()
    
    # --- Data Prep ---
    # Use YouTube video titles
    df_analysis = videos_df.copy()
    # Handle case where titles might be NaN
    df_analysis.dropna(subset=['video_title'], inplace=True)
    df_analysis['has_keyword'] = df_analysis['video_title'].str.lower().str.contains(target_keyword)
    
    # Get engagement stats
    comment_counts = comments_df.groupby('video_id').size().reset_index(name='comment_count')
    df_analysis = pd.merge(df_analysis, comment_counts, on='video_id', how='left')
    df_analysis['comment_count'] = df_analysis['comment_count'].fillna(0)
    
    # Calculate Stats
    count_with = df_analysis['has_keyword'].sum()
    
    # Display
    st.subheader(f"Analysis for keyword: '{target_keyword}'")
    
    if count_with > 0:
        # 1. Engagement Impact (Original Analysis)
        avg_with = df_analysis[df_analysis['has_keyword']]['comment_count'].mean()
        avg_without = df_analysis[~df_analysis['has_keyword']]['comment_count'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Videos Found", f"{count_with}")
        col2.metric("Avg Comments (With Keyword)", f"{avg_with:.1f}")
        col3.metric("Avg Comments (Without)", f"{avg_without:.1f}", delta=f"{avg_with - avg_without:.1f}")
        
        # 2. Snowball Sampling: Find Co-occurring Words
        st.subheader(f"Snowball Sampling: Themes related to '{target_keyword}'")
        st.markdown(f"These words frequently appear in titles along with *'{target_keyword}'*. This reveals the sub-topics associated with your search.")
        
        # Get titles that contain the keyword
        subset_titles = df_analysis[df_analysis['has_keyword']]['video_title']
        
        # Tokenize and count words in this subset
        # We exclude the target keyword itself from the results
        try:
            stop_words = "english" # Use default english stop words
            vec = CountVectorizer(stop_words=stop_words, max_features=15)
            bow = vec.fit_transform(subset_titles)
            
            # Sum word counts
            word_counts = pd.DataFrame({
                'word': vec.get_feature_names_out(),
                'count': bow.toarray().sum(axis=0)
            })
            
            # Filter out the target keyword itself so it doesn't dominate the chart
            word_counts = word_counts[word_counts['word'] != target_keyword]
            word_counts = word_counts.sort_values('count', ascending=False)
            
            if not word_counts.empty:
                # Bar Chart of Co-occurring words
                snowball_chart = alt.Chart(word_counts).mark_bar(color='orange').encode(
                    x=alt.X('count:Q', title='Frequency of Co-occurrence'),
                    y=alt.Y('word:N', sort='-x', title='Related Word'),
                    tooltip=['word', 'count']
                )
                st.altair_chart(snowball_chart, use_container_width=True)
            else:
                st.info("No significant co-occurring words found (titles might be too short).")
                
        except ValueError:
            st.warning("Not enough text data to perform snowball sampling.")

        # 3. Engagement Distribution (Boxplot)
        # Uncomment below to enable boxplot visualization
        # st.subheader("Engagement Distribution")
        # chart_data = df_analysis[['has_keyword', 'comment_count']]
        # chart_data['Type'] = chart_data['has_keyword'].map({True: f"With '{target_keyword}'", False: "Without"})
        
        # if len(df_analysis) - count_with == 0:
        #      chart_data = chart_data[chart_data['has_keyword'] == True]

        # chart = alt.Chart(chart_data).mark_boxplot().encode(
        #     x='Type:N',
        #     y=alt.Y('comment_count:Q', scale=alt.Scale(type='log'), title='Comments (Log Scale)'),
        #     color='Type:N'
        # )
        # st.altair_chart(chart, use_container_width=True)

        # Show examples
        st.subheader("Example Videos")
        st.dataframe(df_analysis[df_analysis['has_keyword']][['video_title', 'comment_count']].head(10))
    else:
        st.warning(f"No videos found with the keyword '{target_keyword}'. Try a different term.")