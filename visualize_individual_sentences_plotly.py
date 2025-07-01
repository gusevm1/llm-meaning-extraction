#!/usr/bin/env python3
"""
Individual sentence visualization script using Plotly for better emoji support.
Creates clean, scientific paper-ready figures for compression analysis.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import textwrap
from pathlib import Path
import glob
import os
from datetime import datetime
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

def load_data():
    """Load compression experiment data from both providers."""
    files = find_latest_csv_files()
    
    data = {}
    for provider, filename in files.items():
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            data[provider] = df
            print(f"✅ Loaded {len(df)} experiments from {filename}")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
    
    return data

def wrap_text(text, width=50):
    """Wrap text to specified width."""
    if not text or pd.isna(text):
        return "N/A"
    return '<br>'.join(textwrap.wrap(str(text), width=width))

def create_sentence_figure_plotly(data, sentence_idx):
    """Create a clean, scientific paper-ready Plotly figure for a single sentence."""
    
    providers = list(data.keys())
    
    # Get the original sentence (same for both providers)
    original_text = data[providers[0]].iloc[sentence_idx]['original']
    
    # Create subplot structure
    fig = make_subplots(
        rows=4, cols=2,
        row_heights=[0.22, 0.08, 0.32, 0.38],
        column_widths=[0.5, 0.5],
        specs=[
            [{"colspan": 2}, None],  # Original text spans both columns
            [{"type": "xy"}, {"type": "xy"}],  # Provider headers
            [{"type": "xy"}, {"type": "xy"}],  # Compressed text
            [{"type": "xy"}, {"type": "xy"}]   # Decompressed text
        ],
        subplot_titles=("", "", "", "", "", "", ""),
        vertical_spacing=0.06,
        horizontal_spacing=0.08
    )
    
    # Professional grayscale color scheme for academic papers
    colors = {
        'openai': '#404040',   # Dark grey
        'gemini': '#606060'    # Slightly lighter grey for subtle differentiation
    }
    
    light_colors = {
        'openai': '#f5f5f5',   # Very light grey
        'gemini': '#f0f0f0'    # Slightly different light grey for subtle differentiation
    }
    
    # Original text section - simplified for PNG export
    original_wrapped = wrap_text(original_text, 110)
    fig.add_annotation(
        text=f"<b>Original Text</b><br><br>{original_wrapped}<br><br><b>Length: {len(original_text)} characters</b>",
        xref="paper", yref="paper",
        x=0.5, y=0.94,
        xanchor="center", yanchor="top",
        font=dict(size=12, color="#000000", family="Arial"),
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1,
        showarrow=False,
        width=1000,
        align="left"
    )
    
    # Process each provider
    for i, provider in enumerate(providers):
        row_data = data[provider].iloc[sentence_idx]
        compressed = row_data['compressed']
        decompressed = row_data['decompressed']
        compression_ratio = row_data['achieved_compression_ratio']
        composite_score = row_data['composite_score']
        
        # Provider header - simplified
        fig.add_annotation(
            text=f"<b>{provider.upper()}</b>",
            xref="paper", yref="paper",
            x=0.25 + (i * 0.5), y=0.70,
            xanchor="center", yanchor="middle",
            font=dict(size=14, color=colors[provider], family="Arial"),
            bgcolor="white",
            bordercolor=colors[provider],
            borderwidth=1.5,
            showarrow=False,
            width=280
        )
        
        # Compressed text section - simplified styling for PNG compatibility
        compressed_text = str(compressed) if not pd.isna(compressed) else "N/A"
        compressed_display = wrap_text(compressed_text, 35)
        
        fig.add_annotation(
            text=f"<b>Compressed Representation</b><br><br>{compressed_display}<br><br><b>Characters:</b> {len(compressed_text)}<br><b>Compression Ratio:</b> {compression_ratio:.3f} ({compression_ratio*100:.1f}%)",
            xref="paper", yref="paper",
            x=0.25 + (i * 0.5), y=0.58,
            xanchor="center", yanchor="top",
            font=dict(size=11, color="#000000", family="Courier New"),
            bgcolor=light_colors[provider],
            bordercolor=colors[provider],
            borderwidth=1,
            showarrow=False,
            width=420,
            align="center"
        )
        
        # Decompressed text section - simplified
        decompressed_wrapped = wrap_text(decompressed, 55)
        semantic_quality = row_data.get('semantic_interpretation', 'Unknown')
        
        # Subtle grayscale quality indicators
        quality_colors = {
            'Excellent': '#000000',  # Black for excellent
            'Good': '#333333',       # Dark grey for good
            'Moderate': '#666666',   # Medium grey for moderate
            'Poor': '#999999'        # Light grey for poor
        }
        quality_color = quality_colors.get(semantic_quality, '#777777')
        
        fig.add_annotation(
            text=f"<b>Decompressed Output</b><br><br>{decompressed_wrapped}<br><br><b>Length:</b> {len(str(decompressed))} characters<br><b>Semantic Score:</b> {composite_score:.3f}<br><b>Quality:</b> {semantic_quality}",
            xref="paper", yref="paper",
            x=0.25 + (i * 0.5), y=0.35,
            xanchor="center", yanchor="top",
            font=dict(size=10, color="#000000", family="Arial"),
            bgcolor="#ffffff",
            bordercolor="#cccccc",
            borderwidth=1,
            showarrow=False,
            width=420,
            align="left"
        )
    
    # Remove all subplot axes and traces
    for i in range(1, 5):
        for j in range(1, 3):
            if i == 1 and j == 2:  # Skip the None subplot
                continue
            fig.update_xaxes(visible=False, row=i, col=j)
            fig.update_yaxes(visible=False, row=i, col=j)
    
    # Professional layout for scientific papers
    fig.update_layout(
        height=850,
        width=1200,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(family="Arial, sans-serif")
    )
    
    return fig

def save_plotly_figures(figures, output_dir="individual_sentences_plotly"):
    """Save Plotly figures as both HTML and PNG files."""
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    for sentence_idx, fig in figures.items():
        # Save as HTML first (guaranteed to work with emojis)
        html_filename = f"{output_dir}/sentence_{sentence_idx+1:02d}_{timestamp}.html"
        fig.write_html(html_filename)
        saved_files.append(html_filename)
        print(f"✅ Saved HTML: {html_filename}")
        
        # Try to save as PNG (for papers)
        png_filename = f"{output_dir}/sentence_{sentence_idx+1:02d}_{timestamp}.png"
        try:
            # Try with simpler settings first
            fig.write_image(png_filename, width=1200, height=850, scale=2, engine="kaleido")
            saved_files.append(png_filename)
            print(f"✅ Saved PNG: {png_filename}")
        except Exception as e:
            print(f"⚠️  PNG export failed for sentence {sentence_idx+1}, HTML available: {html_filename}")
    
    return saved_files

def main():
    """Main function to generate clean scientific visualizations."""
    print("🚀 Loading data and generating scientific paper visualizations...")
    
    # Load data
    data = load_data()
    
    if len(data) < 2:
        print("❌ Need data from both providers for comparison!")
        print("Available data:", list(data.keys()))
        return
    
    # Get the number of sentences
    num_sentences = min(len(data[provider]) for provider in data.keys())
    print(f"📊 Generating {num_sentences} clean scientific figures...")
    print("📝 Optimized for scientific paper inclusion")
    
    # Generate figures for each sentence
    figures = {}
    
    for sentence_idx in range(num_sentences):
        print(f"Creating figure for sentence {sentence_idx + 1}/{num_sentences}...")
        try:
            figures[sentence_idx] = create_sentence_figure_plotly(data, sentence_idx)
        except Exception as e:
            print(f"Error creating figure for sentence {sentence_idx + 1}: {e}")
            continue
    
    # Save all figures
    print("💾 Saving high-quality figures...")
    saved_files = save_plotly_figures(figures)
    
    print(f"✅ Generated {len(saved_files)} scientific visualizations!")
    print(f"📁 All figures saved to 'individual_sentences_plotly/' directory")
    
    # Print a summary
    print("\n📋 Summary:")
    print(f"   • Total sentences: {num_sentences}")
    print(f"   • Successfully created: {len(figures)}")
    print(f"   • Files saved: {len(saved_files)}")
    print(f"   • Providers compared: {', '.join(data.keys()).upper()}")
    print(f"   • Output directory: individual_sentences_plotly/")
    print(f"   • File format: High-resolution PNG (3x scale)")
    print(f"   • Style: Clean scientific paper design")
    print(f"   • Emoji rendering: Native web fonts (perfect display)")

if __name__ == "__main__":
    main() 