#!/usr/bin/env python3
"""
Human Survey Analysis: Comparing Human Judgments with Automated Semantic Scores
Analyzes survey data where humans rated sentence similarity and preferences,
and compares these ratings with our automated composite scores.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from datetime import datetime
from pathlib import Path
import re
from sklearn.linear_model import LinearRegression

def load_survey_data(csv_filename="Sentence Meaning Compraison.csv"):
    """Load and parse the human survey data."""
    try:
        # Read the CSV file with proper encoding
        df = pd.read_csv(csv_filename, encoding='utf-8')
        print(f"✅ Loaded survey data with {len(df)} responses")
        return df
    except Exception as e:
        print(f"❌ Error loading survey data: {e}")
        return pd.DataFrame()

def parse_survey_columns(df):
    """Parse the complex column structure of the survey data."""
    if df.empty:
        return df
    
    # The survey has pairs of columns for each sentence (similarity rating + preference)
    # We need to identify these pairs
    columns = df.columns.tolist()
    
    # Skip timestamp column
    data_columns = columns[1:]  # Skip 'Zeitstempel'
    
    # Group columns in pairs (similarity, preference)
    sentence_pairs = []
    for i in range(0, len(data_columns), 2):
        if i + 1 < len(data_columns):
            sentence_pairs.append({
                'similarity_col': data_columns[i],
                'preference_col': data_columns[i + 1],
                'sentence_num': (i // 2) + 1
            })
    
    print(f"📊 Identified {len(sentence_pairs)} sentence pairs in survey")
    return sentence_pairs

def convert_similarity_to_score(similarity_rating):
    """Convert human similarity ratings to numerical scores (0-1 scale)."""
    similarity_mapping = {
        'Extremely similar in meaning': 1.0,
        'Somewhat similar in meaning': 0.75,
        'Adjacent in meaning': 0.5,
        'Loosely connected': 0.25,
        'Not connected at all': 0.0
    }
    
    return similarity_mapping.get(similarity_rating, np.nan)

def extract_survey_metrics(df, sentence_pairs):
    """Extract and process survey metrics for each sentence."""
    survey_results = []
    
    for pair in sentence_pairs:
        sentence_num = pair['sentence_num']
        similarity_col = pair['similarity_col']
        preference_col = pair['preference_col']
        
        # Get all responses for this sentence
        similarity_ratings = df[similarity_col].dropna()
        preference_ratings = df[preference_col].dropna()
        
        # Convert similarity ratings to scores
        similarity_scores = similarity_ratings.apply(convert_similarity_to_score)
        
        # Calculate metrics
        mean_similarity = similarity_scores.mean()
        std_similarity = similarity_scores.std()
        
        # Calculate preference percentages
        sentence1_pref = (preference_ratings == 'Sentence 1').sum() / len(preference_ratings) * 100
        sentence2_pref = (preference_ratings == 'Sentence 2').sum() / len(preference_ratings) * 100
        
        # Count rating distributions
        rating_counts = similarity_ratings.value_counts()
        
        survey_results.append({
            'sentence_num': sentence_num,
            'num_responses': len(similarity_scores),
            'mean_human_similarity': mean_similarity,
            'std_human_similarity': std_similarity,
            'sentence1_preference_pct': sentence1_pref,
            'sentence2_preference_pct': sentence2_pref,
            'rating_distribution': rating_counts.to_dict(),
            'raw_similarity_scores': similarity_scores.tolist()
        })
    
    return survey_results

def load_automated_scores():
    """Load the automated composite scores for comparison."""
    try:
        # Load Gemini results (since the survey uses Gemini outputs)
        gemini_file = "compression_experiments_gemini_20250630_220351.csv"
        df = pd.read_csv(gemini_file)
        
        # Get the first 5 sentences (matching survey data)
        first_5_sentences = df.head(5).copy()
        first_5_sentences['sentence_num'] = range(1, 6)
        
        print(f"✅ Loaded automated scores for {len(first_5_sentences)} sentences")
        return first_5_sentences[['sentence_num', 'composite_score', 
                                 'embedding_similarity', 'bleu_score',
                                 'keyword_preservation', 'llm_similarity']].to_dict('records')
    except Exception as e:
        print(f"❌ Error loading automated scores: {e}")
        return []

def create_comparison_analysis(survey_results, automated_scores):
    """Create comprehensive comparison analysis."""
    # Merge survey and automated data
    comparison_data = []
    
    for survey in survey_results:
        sentence_num = survey['sentence_num']
        
        # Find matching automated score
        automated = next((a for a in automated_scores if a['sentence_num'] == sentence_num), None)
        
        if automated:
            comparison_data.append({
                'sentence_num': sentence_num,
                'human_similarity': survey['mean_human_similarity'],
                'human_similarity_std': survey['std_human_similarity'],
                'automated_composite': automated['composite_score'],
                'automated_embedding': automated['embedding_similarity'],
                'automated_bleu': automated['bleu_score'],
                'automated_keyword': automated['keyword_preservation'],
                'automated_llm': automated['llm_similarity'],
                'num_responses': survey['num_responses'],
                'sentence2_preference': survey['sentence2_preference_pct'],
                'rating_distribution': survey['rating_distribution']
            })
    
    return comparison_data

def create_human_vs_automated_chart(comparison_data):
    """Create chart comparing human ratings with automated scores."""
    sentences = [d['sentence_num'] for d in comparison_data]
    human_scores = [d['human_similarity'] for d in comparison_data]
    human_stds = [d['human_similarity_std'] for d in comparison_data]
    automated_scores = [d['automated_composite'] for d in comparison_data]
    
    # Calculate correlation
    correlation = np.corrcoef(human_scores, automated_scores)[0, 1]
    
    fig = go.Figure()
    
    # Human ratings with error bars
    fig.add_trace(go.Scatter(
        x=sentences,
        y=human_scores,
        error_y=dict(
            type='data',
            array=human_stds,
            visible=True,
            color='rgba(231, 76, 60, 0.3)'
        ),
        mode='markers+lines',
        name='Human Ratings',
        marker=dict(color='#E74C3C', size=10),
        line=dict(color='#E74C3C', width=3)
    ))
    
    # Automated scores
    fig.add_trace(go.Scatter(
        x=sentences,
        y=automated_scores,
        mode='markers+lines',
        name='Automated Composite Score',
        marker=dict(color='#2E86C1', size=10),
        line=dict(color='#2E86C1', width=3)
    ))
    
    fig.update_layout(
        title=f'<b>Human vs Automated Semantic Similarity Ratings</b><br><sub>Correlation: r = {correlation:.3f}</sub>',
        xaxis_title='Sentence Number',
        yaxis_title='Similarity Score (0-1 scale)',
        yaxis=dict(range=[0, 1.1]),
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(x=0.02, y=0.98),
        height=500,
        width=800
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_rating_distribution_chart(comparison_data):
    """Create chart showing distribution of human ratings."""
    fig = make_subplots(
        rows=1, cols=len(comparison_data),
        subplot_titles=[f"Sentence {d['sentence_num']}" for d in comparison_data],
        specs=[[{"type": "xy"}] * len(comparison_data)]
    )
    
    rating_order = ['Extremely similar in meaning', 'Somewhat similar in meaning', 
                   'Adjacent in meaning', 'Loosely connected', 'Not connected at all']
    colors = ['#27AE60', '#F39C12', '#E67E22', '#E74C3C', '#8E44AD']
    
    for i, data in enumerate(comparison_data):
        ratings = []
        counts = []
        rating_colors = []
        
        for rating in rating_order:
            count = data['rating_distribution'].get(rating, 0)
            if count > 0:
                ratings.append(rating.replace(' in meaning', '').replace(' ', '<br>'))
                counts.append(count)
                rating_colors.append(colors[rating_order.index(rating)])
        
        fig.add_trace(
            go.Bar(
                x=ratings,
                y=counts,
                name=f"Sentence {data['sentence_num']}",
                marker_color=rating_colors,
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        # Update individual subplot
        fig.update_xaxes(tickangle=45, row=1, col=i+1)
        fig.update_yaxes(title_text="Number of Responses" if i == 0 else "", row=1, col=i+1)
    
    fig.update_layout(
        title='<b>Distribution of Human Similarity Ratings by Sentence</b>',
        height=500,
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=11)
    )
    
    return fig

def create_detailed_metrics_comparison(comparison_data):
    """Create detailed comparison of all metrics with trend lines."""
    sentences = [d['sentence_num'] for d in comparison_data]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Human vs Composite Score',
            'Human vs Embedding Similarity', 
            'Human vs LLM Similarity',
            'Preference vs Similarity Ratings'
        ],
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )
    
    # Prepare data
    human_scores = [d['human_similarity'] for d in comparison_data]
    composite_scores = [d['automated_composite'] for d in comparison_data]
    embedding_scores = [d['automated_embedding'] for d in comparison_data]
    llm_scores = [d['automated_llm'] for d in comparison_data]
    preferences = [d['sentence2_preference'] for d in comparison_data]
    
    # Helper function to add trend line
    def add_trend_line(x_data, y_data, color, row, col):
        if len(x_data) > 1:
            # Fit linear regression
            X = np.array(x_data).reshape(-1, 1)
            y = np.array(y_data)
            reg = LinearRegression().fit(X, y)
            
            # Generate trend line points
            x_trend = np.linspace(min(x_data), max(x_data), 100)
            y_trend = reg.predict(x_trend.reshape(-1, 1))
            
            # Add trend line
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    line=dict(color=color, width=2, dash='dot'),
                    name=f'Trend (r²={reg.score(X, y):.3f})',
                    showlegend=False
                ),
                row=row, col=col
            )
    
    # 1. Human vs Composite
    fig.add_trace(
        go.Scatter(
            x=human_scores,
            y=composite_scores,
            mode='markers+text',
            text=[f"S{s}" for s in sentences],
            textposition="top center",
            marker=dict(color='#E74C3C', size=12),
            name='Composite Score'
        ),
        row=1, col=1
    )
    add_trend_line(human_scores, composite_scores, '#E74C3C', 1, 1)
    
    # 2. Human vs Embedding
    fig.add_trace(
        go.Scatter(
            x=human_scores,
            y=embedding_scores,
            mode='markers+text',
            text=[f"S{s}" for s in sentences],
            textposition="top center",
            marker=dict(color='#2E86C1', size=12),
            name='Embedding Similarity'
        ),
        row=1, col=2
    )
    add_trend_line(human_scores, embedding_scores, '#2E86C1', 1, 2)
    
    # 3. Human vs LLM
    fig.add_trace(
        go.Scatter(
            x=human_scores,
            y=llm_scores,
            mode='markers+text',
            text=[f"S{s}" for s in sentences],
            textposition="top center",
            marker=dict(color='#27AE60', size=12),
            name='LLM Similarity'
        ),
        row=2, col=1
    )
    add_trend_line(human_scores, llm_scores, '#27AE60', 2, 1)
    
    # 4. Preference vs Similarity
    fig.add_trace(
        go.Scatter(
            x=human_scores,
            y=preferences,
            mode='markers+text',
            text=[f"S{s}" for s in sentences],
            textposition="top center",
            marker=dict(color='#F39C12', size=12),
            name='Gemini Preference %'
        ),
        row=2, col=2
    )
    add_trend_line(human_scores, preferences, '#F39C12', 2, 2)
    
    # Update layout with better spacing to prevent cutoff
    fig.update_layout(
        title='<b>Detailed Human vs Automated Metrics Comparison</b>',
        height=750,
        width=1300,
        showlegend=False,
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    # Update axes labels with proper spacing
    fig.update_xaxes(title_text="Human Similarity Rating", title_font=dict(size=12), row=1, col=1)
    fig.update_yaxes(title_text="Automated Composite Score", title_font=dict(size=12), row=1, col=1)
    
    fig.update_xaxes(title_text="Human Similarity Rating", title_font=dict(size=12), row=1, col=2)
    fig.update_yaxes(title_text="Embedding Similarity", title_font=dict(size=12), row=1, col=2)
    
    fig.update_xaxes(title_text="Human Similarity Rating", title_font=dict(size=12), row=2, col=1)
    fig.update_yaxes(title_text="LLM Similarity Score", title_font=dict(size=12), row=2, col=1)
    
    fig.update_xaxes(title_text="Human Similarity Rating", title_font=dict(size=12), row=2, col=2)
    fig.update_yaxes(title_text="Gemini Preference (%)", title_font=dict(size=12), row=2, col=2)
    
    # Add subtle grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def print_analysis_summary(comparison_data):
    """Print comprehensive analysis summary."""
    print("\n" + "="*80)
    print("📊 HUMAN vs AUTOMATED SEMANTIC EVALUATION ANALYSIS")
    print("="*80)
    
    # Calculate overall correlations
    human_scores = [d['human_similarity'] for d in comparison_data]
    composite_scores = [d['automated_composite'] for d in comparison_data]
    embedding_scores = [d['automated_embedding'] for d in comparison_data]
    llm_scores = [d['automated_llm'] for d in comparison_data]
    
    corr_composite = np.corrcoef(human_scores, composite_scores)[0, 1]
    corr_embedding = np.corrcoef(human_scores, embedding_scores)[0, 1]
    corr_llm = np.corrcoef(human_scores, llm_scores)[0, 1]
    
    print(f"\n🔗 CORRELATIONS WITH HUMAN RATINGS:")
    print(f"   Composite Score:     r = {corr_composite:.3f}")
    print(f"   Embedding Similarity: r = {corr_embedding:.3f}")
    print(f"   LLM Similarity:      r = {corr_llm:.3f}")
    
    print(f"\n📈 SENTENCE-BY-SENTENCE ANALYSIS:")
    for data in comparison_data:
        print(f"\n   Sentence {data['sentence_num']}:")
        print(f"     Human Rating: {data['human_similarity']:.3f} (±{data['human_similarity_std']:.3f})")
        print(f"     Automated Score: {data['automated_composite']:.3f}")
        print(f"     Difference: {abs(data['human_similarity'] - data['automated_composite']):.3f}")
        print(f"     Gemini Preference: {data['sentence2_preference']:.1f}%")
        print(f"     Responses: {data['num_responses']}")
    
    # Calculate average differences
    differences = [abs(d['human_similarity'] - d['automated_composite']) for d in comparison_data]
    avg_difference = np.mean(differences)
    
    print(f"\n📏 OVERALL ALIGNMENT:")
    print(f"   Average Absolute Difference: {avg_difference:.3f}")
    print(f"   Human Rating Range: {min(human_scores):.3f} - {max(human_scores):.3f}")
    print(f"   Automated Score Range: {min(composite_scores):.3f} - {max(composite_scores):.3f}")
    
    # Preference analysis
    avg_gemini_preference = np.mean([d['sentence2_preference'] for d in comparison_data])
    print(f"\n👥 PREFERENCE ANALYSIS:")
    print(f"   Average Gemini Preference: {avg_gemini_preference:.1f}%")
    print(f"   Average Original Preference: {100-avg_gemini_preference:.1f}%")

def save_analysis_figures(figures, output_dir="human_survey_analysis"):
    """Save analysis figures."""
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    figure_names = ["human_vs_automated", "rating_distribution", "detailed_metrics"]
    
    for name, fig in zip(figure_names, figures):
        # Save as HTML
        html_filename = f"{output_dir}/{name}_{timestamp}.html"
        fig.write_html(html_filename)
        saved_files.append(html_filename)
        print(f"✅ Saved HTML: {html_filename}")
        
        # Save as PNG
        png_filename = f"{output_dir}/{name}_{timestamp}.png"
        try:
            # Use different dimensions for different chart types
            if name == "detailed_metrics":
                fig.write_image(png_filename, width=1300, height=750, scale=2)
            else:
                fig.write_image(png_filename, width=1200, height=700, scale=2)
            saved_files.append(png_filename)
            print(f"✅ Saved PNG: {png_filename}")
        except Exception as e:
            print(f"⚠️  PNG export failed for {name}: {e}")
    
    return saved_files

def main():
    """Main analysis function."""
    print("🔍 HUMAN SURVEY vs AUTOMATED SCORING ANALYSIS")
    print("="*60)
    
    # Load survey data
    survey_df = load_survey_data()
    if survey_df.empty:
        print("❌ No survey data loaded. Exiting.")
        return
    
    # Parse survey structure
    sentence_pairs = parse_survey_columns(survey_df)
    if not sentence_pairs:
        print("❌ Could not parse survey structure. Exiting.")
        return
    
    # Extract survey metrics
    print("📊 Processing survey responses...")
    survey_results = extract_survey_metrics(survey_df, sentence_pairs)
    
    # Load automated scores
    print("📈 Loading automated scores...")
    automated_scores = load_automated_scores()
    
    if not automated_scores:
        print("❌ Could not load automated scores. Exiting.")
        return
    
    # Create comparison analysis
    print("🔗 Creating comparison analysis...")
    comparison_data = create_comparison_analysis(survey_results, automated_scores)
    
    # Print summary
    print_analysis_summary(comparison_data)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    figures = []
    figures.append(create_human_vs_automated_chart(comparison_data))
    figures.append(create_rating_distribution_chart(comparison_data))
    figures.append(create_detailed_metrics_comparison(comparison_data))
    
    # Save figures
    print("\n💾 Saving analysis figures...")
    saved_files = save_analysis_figures(figures)
    
    print(f"\n✅ Human survey analysis complete!")
    print(f"📁 Files saved: {len(saved_files)}")
    print(f"📂 Output directory: human_survey_analysis/")
    print(f"🎯 Analysis includes: correlation analysis, rating distributions, detailed metrics")

if __name__ == "__main__":
    main() 