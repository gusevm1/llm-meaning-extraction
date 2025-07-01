#!/usr/bin/env python3
"""
Performance Comparison Analysis: Gemini vs OpenAI
Comprehensive comparison of compression and semantic preservation performance
between Gemini and OpenAI models using scientific visualizations.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import glob
import os
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def find_latest_csv_files():
    """Find the most recent CSV files for each provider."""
    files = {}
    
    # Look for OpenAI files
    openai_files = glob.glob("compression_experiments_openai*.csv")
    if openai_files:
        openai_files.sort(key=os.path.getmtime, reverse=True)
        files['openai'] = openai_files[0]
    
    # Look for Gemini files  
    gemini_files = glob.glob("compression_experiments_gemini*.csv")
    if gemini_files:
        gemini_files.sort(key=os.path.getmtime, reverse=True)
        files['gemini'] = gemini_files[0]
    
    return files

def load_comparison_data():
    """Load and combine data from both providers for comparison."""
    files = find_latest_csv_files()
    
    data = {}
    combined_data = []
    
    for provider, filename in files.items():
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            df['provider'] = provider  # Ensure provider column exists
            data[provider] = df
            combined_data.append(df)
            print(f"✅ Loaded {len(df)} experiments from {filename} ({provider})")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        return data, combined_df
    else:
        return {}, pd.DataFrame()

def calculate_performance_metrics(data):
    """Calculate aggregate performance metrics for each provider."""
    metrics = {}
    
    for provider, df in data.items():
        # Filter successful experiments only for most metrics
        successful_df = df[df['compression_success'] == True]
        
        provider_metrics = {
            'total_experiments': len(df),
            'successful_compressions': len(successful_df),
            'success_rate': len(successful_df) / len(df) * 100 if len(df) > 0 else 0,
            
            # Compression metrics (only for successful compressions)
            'avg_compression_ratio': successful_df['achieved_compression_ratio'].mean() if len(successful_df) > 0 else 0,
            'best_compression_ratio': successful_df['achieved_compression_ratio'].min() if len(successful_df) > 0 else 0,
            'compression_std': successful_df['achieved_compression_ratio'].std() if len(successful_df) > 0 else 0,
            
            # Semantic preservation metrics (for all experiments)
            'avg_composite_score': df['composite_score'].mean(),
            'best_composite_score': df['composite_score'].max(),
            'composite_score_std': df['composite_score'].std(),
            
            # Individual semantic metrics
            'avg_embedding_similarity': df['embedding_similarity'].mean(),
            'avg_bleu_score': df['bleu_score'].mean(),
            'avg_keyword_preservation': df['keyword_preservation'].mean(),
            'avg_llm_similarity': df['llm_similarity'].mean(),
            
            # Quality distribution
            'excellent_quality': len(df[df['semantic_interpretation'] == 'Excellent']) / len(df) * 100,
            'good_quality': len(df[df['semantic_interpretation'] == 'Good']) / len(df) * 100,
            'moderate_quality': len(df[df['semantic_interpretation'] == 'Moderate']) / len(df) * 100,
            'poor_quality': len(df[df['semantic_interpretation'] == 'Poor']) / len(df) * 100,
        }
        
        metrics[provider] = provider_metrics
    
    return metrics

def create_performance_overview_chart(metrics):
    """Create a comprehensive performance overview comparing both providers."""
    providers = list(metrics.keys())
    
    # Create subplot with 2x2 grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Success Rate Comparison",
            "Average Semantic Preservation",
            "Quality Distribution by Model",
            "Individual Semantic Metrics"
        ],
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Professional color scheme
    colors = {
        'openai': '#2E86C1',    # Professional blue
        'gemini': '#E74C3C'     # Professional red
    }
    
    provider_labels = [p.upper() for p in providers]
    
    # 1. Success Rate Only (Clean bars)
    success_rates = [metrics[p]['success_rate'] for p in providers]
    
    fig.add_trace(
        go.Bar(
            x=provider_labels,
            y=success_rates,
            name="Success Rate",
            marker_color=[colors[p] for p in providers],
            opacity=0.8,
            text=[f"{rate:.0f}%" for rate in success_rates],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Semantic Preservation Scores
    semantic_scores = [metrics[p]['avg_composite_score'] for p in providers]
    
    fig.add_trace(
        go.Bar(
            x=provider_labels,
            y=semantic_scores,
            name="Composite Score",
            marker_color=[colors[p] for p in providers],
            opacity=0.8,
            text=[f"{score:.3f}" for score in semantic_scores],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Quality Distribution (Stacked Bar with local legend)
    qualities = ['Excellent', 'Good', 'Moderate', 'Poor']
    quality_colors = ['#27AE60', '#F39C12', '#E67E22', '#E74C3C']
    
    for i, quality in enumerate(qualities):
        quality_key = f'{quality.lower()}_quality'
        values = [metrics[p][quality_key] for p in providers]
        
        fig.add_trace(
            go.Bar(
                x=provider_labels,
                y=values,
                name=f"{quality}",
                marker_color=quality_colors[i],
                opacity=0.8,
                legendgroup="quality",
                showlegend=False  # Hide from global legend
            ),
            row=2, col=1
        )
    
    # Add custom legend annotations for quality distribution
    for i, (quality, color) in enumerate(zip(qualities, quality_colors)):
        fig.add_annotation(
            x=0.02,  # Position relative to the quality subplot
            y=0.45 - (i * 0.04),  # Stack vertically
            xref="paper",
            yref="paper",
            text=f"<span style='color:{color}'>●</span> {quality}",
            showarrow=False,
            font=dict(size=11, family="Arial"),
            align="left",
            xanchor="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(128,128,128,0.3)",
            borderwidth=1
        )
    
    # 4. Individual Semantic Metrics (Grouped bars with legend)
    semantic_metrics = ['embedding_similarity', 'bleu_score', 'keyword_preservation', 'llm_similarity']
    metric_labels = ['Embedding<br>Similarity', 'BLEU<br>Score', 'Keyword<br>Preservation', 'LLM<br>Similarity']
    
    for i, provider in enumerate(providers):
        values = [metrics[provider][f'avg_{metric}'] for metric in semantic_metrics]
        
        fig.add_trace(
            go.Bar(
                x=metric_labels,
                y=values,
                name=f"{provider.upper()}",
                marker_color=colors[provider],
                opacity=0.8,
                legendgroup="providers",
                showlegend=True
            ),
            row=2, col=2
        )
    
    # Update layout with better formatting
    fig.update_layout(
        height=850,
        width=1400,
        showlegend=True,
        title={
            'text': "<b>Performance Comparison: Gemini vs OpenAI</b><br><sub>Compression and Semantic Preservation Analysis</sub>",
            'x': 0.5,
            'font': {'size': 22, 'family': 'Arial'}
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        barmode='group',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(size=11)
        )
    )
    
    # Update individual subplot layouts with cleaner labels
    fig.update_xaxes(title_text="Model", title_font=dict(size=14), row=1, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", title_font=dict(size=14), range=[0, 105], row=1, col=1)
    
    fig.update_xaxes(title_text="Model", title_font=dict(size=14), row=1, col=2)
    fig.update_yaxes(title_text="Composite Score", title_font=dict(size=14), range=[0, 0.8], row=1, col=2)
    
    fig.update_xaxes(title_text="Model", title_font=dict(size=14), row=2, col=1)
    fig.update_yaxes(title_text="Percentage (%)", title_font=dict(size=14), row=2, col=1)
    
    fig.update_xaxes(title_text="Semantic Metrics", title_font=dict(size=14), row=2, col=2)
    fig.update_yaxes(title_text="Average Score", title_font=dict(size=14), range=[0, 1], row=2, col=2)
    
    # Add subtle grid lines for better readability
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_detailed_scatter_analysis(combined_df):
    """Create detailed scatter plot analysis of compression vs semantic preservation."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Compression vs Semantic Quality", "Provider Performance Distribution"],
        horizontal_spacing=0.1
    )
    
    # Color mapping for providers
    color_map = {'openai': '#2E86C1', 'gemini': '#E74C3C'}
    
    # 1. Scatter plot: Compression Ratio vs Composite Score
    for provider in combined_df['provider'].unique():
        provider_data = combined_df[combined_df['provider'] == provider]
        
        fig.add_trace(
            go.Scatter(
                x=provider_data['achieved_compression_ratio'],
                y=provider_data['composite_score'],
                mode='markers',
                name=provider.upper(),
                marker=dict(
                    color=color_map[provider],
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=provider_data['sentence_index'],
                hovertemplate=f"<b>{provider.upper()}</b><br>" +
                             "Compression Ratio: %{x:.3f}<br>" +
                             "Semantic Score: %{y:.3f}<br>" +
                             "Sentence: %{text}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # 2. Box plots for provider comparison
    for i, provider in enumerate(combined_df['provider'].unique()):
        provider_data = combined_df[combined_df['provider'] == provider]
        
        fig.add_trace(
            go.Box(
                y=provider_data['composite_score'],
                name=provider.upper(),
                marker_color=color_map[provider],
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8 + i*0.8,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=1200,
        title={
            'text': "<b>Detailed Performance Analysis</b>",
            'x': 0.5,
            'font': {'size': 18}
        },
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(title_text="Compression Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Composite Semantic Score", row=1, col=1)
    
    fig.update_xaxes(title_text="Provider", row=1, col=2)
    fig.update_yaxes(title_text="Composite Semantic Score", row=1, col=2)
    
    return fig

def create_sentence_level_comparison(combined_df):
    """Create sentence-by-sentence comparison chart."""
    # Pivot data for side-by-side comparison
    pivot_compression = combined_df.pivot(index='sentence_index', columns='provider', values='achieved_compression_ratio')
    pivot_semantic = combined_df.pivot(index='sentence_index', columns='provider', values='composite_score')
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Compression Ratio by Sentence", "Semantic Score by Sentence"],
        vertical_spacing=0.15
    )
    
    colors = {'openai': '#2E86C1', 'gemini': '#E74C3C'}
    
    # Compression comparison
    for provider in pivot_compression.columns:
        fig.add_trace(
            go.Scatter(
                x=pivot_compression.index,
                y=pivot_compression[provider],
                mode='lines+markers',
                name=f"{provider.upper()} - Compression",
                line=dict(color=colors[provider], width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
    
    # Semantic comparison
    for provider in pivot_semantic.columns:
        fig.add_trace(
            go.Scatter(
                x=pivot_semantic.index,
                y=pivot_semantic[provider],
                mode='lines+markers',
                name=f"{provider.upper()} - Semantic",
                line=dict(color=colors[provider], width=2, dash='dot'),
                marker=dict(size=6),
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=700,
        width=1200,
        title={
            'text': "<b>Sentence-Level Performance Comparison</b>",
            'x': 0.5,
            'font': {'size': 18}
        },
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(title_text="Sentence Index", row=2, col=1)
    fig.update_yaxes(title_text="Compression Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Semantic Score", row=2, col=1)
    
    return fig

def print_comparative_summary(metrics):
    """Print a detailed comparative summary."""
    print("\n" + "="*80)
    print("🏆 COMPREHENSIVE PERFORMANCE COMPARISON: GEMINI vs OPENAI")
    print("="*80)
    
    providers = list(metrics.keys())
    
    print(f"\n📊 SUCCESS RATES:")
    for provider in providers:
        rate = metrics[provider]['success_rate']
        successful = metrics[provider]['successful_compressions']
        total = metrics[provider]['total_experiments']
        print(f"   {provider.upper()}: {rate:.1f}% ({successful}/{total} experiments)")
    
    print(f"\n🗜️  COMPRESSION PERFORMANCE:")
    for provider in providers:
        avg_ratio = metrics[provider]['avg_compression_ratio']
        best_ratio = metrics[provider]['best_compression_ratio']
        print(f"   {provider.upper()}:")
        print(f"     Average Compression: {avg_ratio:.3f} ({(1-avg_ratio)*100:.1f}% size reduction)")
        print(f"     Best Compression: {best_ratio:.3f} ({(1-best_ratio)*100:.1f}% size reduction)")
    
    print(f"\n🎭 SEMANTIC PRESERVATION:")
    for provider in providers:
        avg_score = metrics[provider]['avg_composite_score']
        best_score = metrics[provider]['best_composite_score']
        print(f"   {provider.upper()}:")
        print(f"     Average Semantic Score: {avg_score:.3f}")
        print(f"     Best Semantic Score: {best_score:.3f}")
    
    print(f"\n📈 QUALITY DISTRIBUTION:")
    qualities = ['excellent', 'good', 'moderate', 'poor']
    for provider in providers:
        print(f"   {provider.upper()}:")
        for quality in qualities:
            percentage = metrics[provider][f'{quality}_quality']
            print(f"     {quality.title()}: {percentage:.1f}%")
    
    # Winner analysis
    print(f"\n🏅 WINNER ANALYSIS:")
    
    # Success rate winner
    success_winner = max(providers, key=lambda p: metrics[p]['success_rate'])
    print(f"   📊 Success Rate: {success_winner.upper()} ({metrics[success_winner]['success_rate']:.1f}%)")
    
    # Compression winner (lower is better)
    compression_winner = min(providers, key=lambda p: metrics[p]['avg_compression_ratio'])
    print(f"   🗜️  Best Compression: {compression_winner.upper()} ({metrics[compression_winner]['avg_compression_ratio']:.3f})")
    
    # Semantic winner
    semantic_winner = max(providers, key=lambda p: metrics[p]['avg_composite_score'])
    print(f"   🎭 Semantic Preservation: {semantic_winner.upper()} ({metrics[semantic_winner]['avg_composite_score']:.3f})")
    
    # Quality winner (excellent percentage)
    quality_winner = max(providers, key=lambda p: metrics[p]['excellent_quality'])
    print(f"   ⭐ Excellent Quality Rate: {quality_winner.upper()} ({metrics[quality_winner]['excellent_quality']:.1f}%)")

def save_comparison_figures(figures, output_dir="performance_comparison"):
    """Save comparison figures."""
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    figure_names = ["overview", "detailed_analysis", "sentence_comparison"]
    
    for i, (name, fig) in enumerate(zip(figure_names, figures)):
        # Save as HTML
        html_filename = f"{output_dir}/{name}_{timestamp}.html"
        fig.write_html(html_filename)
        saved_files.append(html_filename)
        print(f"✅ Saved HTML: {html_filename}")
        
        # Save as PNG
        png_filename = f"{output_dir}/{name}_{timestamp}.png"
        try:
            fig.write_image(png_filename, width=1400 if i == 0 else 1200, height=800 if i != 1 else 700, scale=2)
            saved_files.append(png_filename)
            print(f"✅ Saved PNG: {png_filename}")
        except Exception as e:
            print(f"⚠️  PNG export failed for {name}: {e}")
    
    return saved_files

def main():
    """Main function for performance comparison analysis."""
    print("🚀 Performance Comparison Analysis: Gemini vs OpenAI")
    print("="*60)
    
    # Load data
    data, combined_df = load_comparison_data()
    
    if len(data) < 2:
        print("❌ Need data from both providers for comparison!")
        print(f"Available data: {list(data.keys())}")
        return
    
    if combined_df.empty:
        print("❌ No data loaded successfully!")
        return
    
    # Calculate performance metrics
    print("\n📊 Calculating performance metrics...")
    metrics = calculate_performance_metrics(data)
    
    # Print comparative summary
    print_comparative_summary(metrics)
    
    # Create visualizations
    print("\n📈 Creating comparison visualizations...")
    
    figures = []
    
    # 1. Performance overview
    overview_fig = create_performance_overview_chart(metrics)
    figures.append(overview_fig)
    
    # 2. Detailed scatter analysis
    detailed_fig = create_detailed_scatter_analysis(combined_df)
    figures.append(detailed_fig)
    
    # 3. Sentence-level comparison
    sentence_fig = create_sentence_level_comparison(combined_df)
    figures.append(sentence_fig)
    
    # Save figures
    print("\n💾 Saving comparison figures...")
    saved_files = save_comparison_figures(figures)
    
    print(f"\n✅ Performance comparison analysis complete!")
    print(f"📁 Files saved: {len(saved_files)}")
    print(f"📂 Output directory: performance_comparison/")
    print(f"🎯 Analysis includes: success rates, compression efficiency, semantic preservation")
    
    # Export comparison data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"performance_comparison/comparison_data_{timestamp}.csv"
    Path("performance_comparison").mkdir(exist_ok=True)
    combined_df.to_csv(export_filename, index=False)
    print(f"📄 Combined data exported to: {export_filename}")

if __name__ == "__main__":
    main() 