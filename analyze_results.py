"""
Analysis script for compression experiment results.
Loads CSV data from both OpenAI and Gemini experiments and provides comparative analysis and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import glob
import os

def find_latest_csv_by_provider(provider: str = None) -> str:
    """Find the most recent timestamped CSV file for a specific provider."""
    if provider:
        pattern = f"compression_experiments_{provider}_*.csv"
    else:
        pattern = "compression_experiments_*.csv"
    
    timestamped_csvs = glob.glob(pattern)
    
    if not timestamped_csvs:
        # Fall back to default filename
        default_name = f"compression_experiments_{provider}.csv" if provider else "compression_experiments.csv"
        if os.path.exists(default_name):
            return default_name
        else:
            return None
    
    # Sort by modification time (most recent first)
    timestamped_csvs.sort(key=os.path.getmtime, reverse=True)
    return timestamped_csvs[0]

def find_provider_files() -> dict:
    """Find the latest CSV files for each provider."""
    providers = {}
    
    # Look for OpenAI files
    openai_file = find_latest_csv_by_provider("openai")
    if openai_file:
        providers["openai"] = openai_file
    
    # Look for Gemini files
    gemini_file = find_latest_csv_by_provider("gemini")
    if gemini_file:
        providers["gemini"] = gemini_file
    
    return providers

def list_available_csvs() -> list:
    """List all available CSV files."""
    all_csvs = glob.glob("compression_experiments*.csv")
    return sorted(all_csvs, key=os.path.getmtime, reverse=True)

def load_results(csv_filename: str = None) -> pd.DataFrame:
    """Load compression experiment results from CSV."""
    if csv_filename is None:
        # Try to find latest file
        all_files = list_available_csvs()
        if not all_files:
            print("❌ No compression experiment CSV files found!")
            print("   Run experiments first with: python extreme_compression_experiment_openai.py")
            print("   Or: python extreme_compression_experiment_gemini.py")
            return pd.DataFrame()
        
        csv_filename = all_files[0]
        print(f"📁 Auto-detected latest CSV: {csv_filename}")
    
    try:
        df = pd.read_csv(csv_filename)
        print(f"📊 Loaded {len(df)} experiments from {csv_filename}")
        return df
    except FileNotFoundError:
        print(f"❌ File {csv_filename} not found!")
        
        # Show available files
        available = list_available_csvs()
        if available:
            print("📁 Available CSV files:")
            for i, file in enumerate(available):
                mtime = os.path.getmtime(file)
                from datetime import datetime
                time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                provider = "openai" if "openai" in file else ("gemini" if "gemini" in file else "unknown")
                print(f"   {i+1}. {file} ({provider}) - {time_str}")
            print(f"\n💡 Use: python analyze_results.py [filename] to specify a file")
        
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return pd.DataFrame()

def load_comparative_results() -> dict:
    """Load results from both OpenAI and Gemini if available."""
    provider_files = find_provider_files()
    results = {}
    
    for provider, filename in provider_files.items():
        print(f"📁 Loading {provider.upper()} results from: {filename}")
        df = load_results(filename)
        if not df.empty:
            results[provider] = df
    
    return results

def print_summary_statistics(df: pd.DataFrame, provider: str = ""):
    """Print summary statistics of the experiments."""
    if df.empty:
        print("No data to analyze")
        return
    
    provider_title = f" ({provider.upper()})" if provider else ""
    print(f"\n📈 COMPRESSION EXPERIMENT SUMMARY{provider_title}")
    print("="*60)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Total Experiments: {len(df)}")
    print(f"   Successful Compressions: {df['compression_success'].sum()}")
    print(f"   Success Rate: {df['compression_success'].mean()*100:.1f}%")
    
    if 'achieved_compression_ratio' in df.columns:
        print(f"\n🗜️  COMPRESSION PERFORMANCE:")
        print(f"   Mean Compression Ratio: {df['achieved_compression_ratio'].mean():.3f}")
        print(f"   Best Compression: {df['achieved_compression_ratio'].min():.3f}")
        print(f"   Worst Compression: {df['achieved_compression_ratio'].max():.3f}")
        print(f"   Mean Compression %: {df['compression_percentage'].mean():.1f}%")
    
    if 'composite_score' in df.columns:
        print(f"\n🎭 SEMANTIC PRESERVATION:")
        print(f"   Mean Composite Score: {df['composite_score'].mean():.3f}")
        print(f"   Best Semantic Score: {df['composite_score'].max():.3f}")
        print(f"   Worst Semantic Score: {df['composite_score'].min():.3f}")
        
        # Semantic quality distribution
        if 'semantic_interpretation' in df.columns:
            quality_counts = df['semantic_interpretation'].value_counts()
            print(f"\n   Quality Distribution:")
            for quality, count in quality_counts.items():
                print(f"     {quality}: {count} ({count/len(df)*100:.1f}%)")

def print_comparative_summary(results: dict):
    """Print comparative summary between providers."""
    if len(results) < 2:
        print("⚠️  Need results from both providers for comparison")
        return
    
    print("\n" + "="*80)
    print("🔍 COMPARATIVE ANALYSIS: OPENAI vs GEMINI")
    print("="*80)
    
    providers = list(results.keys())
    metrics = {}
    
    for provider in providers:
        df = results[provider]
        metrics[provider] = {
            'count': len(df),
            'success_rate': df['compression_success'].mean() * 100,
            'mean_compression': df['achieved_compression_ratio'].mean(),
            'mean_semantic': df['composite_score'].mean() if 'composite_score' in df.columns else 0,
            'best_compression': df['achieved_compression_ratio'].min(),
            'best_semantic': df['composite_score'].max() if 'composite_score' in df.columns else 0
        }
    
    print(f"\n📊 HEAD-TO-HEAD COMPARISON:")
    print(f"{'Metric':<25} {'OpenAI':<15} {'Gemini':<15} {'Winner':<10}")
    print("-" * 70)
    
    # Success Rate
    openai_success = metrics.get('openai', {}).get('success_rate', 0)
    gemini_success = metrics.get('gemini', {}).get('success_rate', 0)
    winner = "OpenAI" if openai_success > gemini_success else "Gemini" if gemini_success > openai_success else "Tie"
    print(f"{'Success Rate (%)':<25} {openai_success:<15.1f} {gemini_success:<15.1f} {winner:<10}")
    
    # Mean Compression
    openai_comp = metrics.get('openai', {}).get('mean_compression', 0)
    gemini_comp = metrics.get('gemini', {}).get('mean_compression', 0)
    winner = "OpenAI" if openai_comp < gemini_comp else "Gemini" if gemini_comp < openai_comp else "Tie"  # Lower is better
    print(f"{'Mean Compression':<25} {openai_comp:<15.3f} {gemini_comp:<15.3f} {winner:<10}")
    
    # Best Compression
    openai_best_comp = metrics.get('openai', {}).get('best_compression', 0)
    gemini_best_comp = metrics.get('gemini', {}).get('best_compression', 0)
    winner = "OpenAI" if openai_best_comp < gemini_best_comp else "Gemini" if gemini_best_comp < openai_best_comp else "Tie"
    print(f"{'Best Compression':<25} {openai_best_comp:<15.3f} {gemini_best_comp:<15.3f} {winner:<10}")
    
    # Mean Semantic Score
    openai_sem = metrics.get('openai', {}).get('mean_semantic', 0)
    gemini_sem = metrics.get('gemini', {}).get('mean_semantic', 0)
    winner = "OpenAI" if openai_sem > gemini_sem else "Gemini" if gemini_sem > openai_sem else "Tie"  # Higher is better
    print(f"{'Mean Semantic Score':<25} {openai_sem:<15.3f} {gemini_sem:<15.3f} {winner:<10}")
    
    # Best Semantic Score
    openai_best_sem = metrics.get('openai', {}).get('best_semantic', 0)
    gemini_best_sem = metrics.get('gemini', {}).get('best_semantic', 0)
    winner = "OpenAI" if openai_best_sem > gemini_best_sem else "Gemini" if gemini_best_sem > openai_best_sem else "Tie"
    print(f"{'Best Semantic Score':<25} {openai_best_sem:<15.3f} {gemini_best_sem:<15.3f} {winner:<10}")
    
    # Overall efficiency (semantic score / compression ratio)
    openai_eff = openai_sem / openai_comp if openai_comp > 0 else 0
    gemini_eff = gemini_sem / gemini_comp if gemini_comp > 0 else 0
    winner = "OpenAI" if openai_eff > gemini_eff else "Gemini" if gemini_eff > openai_eff else "Tie"
    print(f"{'Efficiency Score':<25} {openai_eff:<15.2f} {gemini_eff:<15.2f} {winner:<10}")

def show_best_compressions(df: pd.DataFrame, n: int = 5, provider: str = ""):
    """Show the best compression results."""
    if df.empty:
        return
    
    provider_title = f" ({provider.upper()})" if provider else ""
    print(f"\n🏆 TOP {n} COMPRESSION RESULTS{provider_title}:")
    print("-" * 60)
    
    # Sort by compression ratio (lower is better)
    best_compressions = df.nsmallest(n, 'achieved_compression_ratio')
    
    for i, (_, row) in enumerate(best_compressions.iterrows(), 1):
        print(f"\n#{i} - Sentence {row['sentence_index']}")
        print(f"   Compression: {row['achieved_compression_ratio']:.3f} ({row['compression_percentage']:.1f}% savings)")
        print(f"   Semantic Score: {row['composite_score']:.3f} ({row['semantic_interpretation']})")
        print(f"   Original: {row['original'][:80]}{'...' if len(row['original']) > 80 else ''}")
        print(f"   Compressed: {row['compressed']}")
        print(f"   Decompressed: {row['decompressed'][:80]}{'...' if len(row['decompressed']) > 80 else ''}")

def show_best_semantic_preservation(df: pd.DataFrame, n: int = 5, provider: str = ""):
    """Show results with best semantic preservation."""
    if df.empty:
        return
    
    provider_title = f" ({provider.upper()})" if provider else ""
    print(f"\n🎭 TOP {n} SEMANTIC PRESERVATION RESULTS{provider_title}:")
    print("-" * 60)
    
    # Sort by composite score (higher is better)
    best_semantic = df.nlargest(n, 'composite_score')
    
    for i, (_, row) in enumerate(best_semantic.iterrows(), 1):
        print(f"\n#{i} - Sentence {row['sentence_index']}")
        print(f"   Semantic Score: {row['composite_score']:.3f} ({row['semantic_interpretation']})")
        print(f"   Compression: {row['achieved_compression_ratio']:.3f}")
        print(f"   Embedding Sim: {row['embedding_similarity']:.3f}")
        print(f"   LLM Similarity: {row['llm_similarity']:.3f}")
        print(f"   Keyword Preservation: {row['keyword_preservation']:.3f}")
        print(f"   Original: {row['original'][:80]}{'...' if len(row['original']) > 80 else ''}")
        print(f"   Compressed: {row['compressed']}")

def analyze_compression_vs_semantic_tradeoff(df: pd.DataFrame, provider: str = ""):
    """Analyze the tradeoff between compression and semantic preservation."""
    if df.empty or len(df) < 2:
        return
    
    provider_title = f" ({provider.upper()})" if provider else ""
    print(f"\n⚖️  COMPRESSION vs SEMANTIC QUALITY ANALYSIS{provider_title}:")
    print("-" * 60)
    
    # Calculate correlation
    if 'achieved_compression_ratio' in df.columns and 'composite_score' in df.columns:
        correlation = df['achieved_compression_ratio'].corr(df['composite_score'])
        print(f"   Correlation (compression ratio vs semantic score): {correlation:.3f}")
        
        if correlation > 0.3:
            print("   📈 Higher compression tends to preserve more semantics (surprising!)")
        elif correlation < -0.3:
            print("   📉 Higher compression tends to lose more semantics (expected)")
        else:
            print("   ➡️  Weak correlation between compression and semantic preservation")
    
    # Find sweet spot
    if len(df) >= 5:
        # Calculate efficiency score (semantic preservation per compression)
        df_copy = df.copy()
        df_copy['efficiency'] = df_copy['composite_score'] / df_copy['achieved_compression_ratio']
        
        best_efficiency = df_copy.loc[df_copy['efficiency'].idxmax()]
        print(f"\n🎯 MOST EFFICIENT COMPRESSION:")
        print(f"   Sentence {best_efficiency['sentence_index']}")
        print(f"   Compression: {best_efficiency['achieved_compression_ratio']:.3f}")
        print(f"   Semantic Score: {best_efficiency['composite_score']:.3f}")
        print(f"   Efficiency Score: {best_efficiency['efficiency']:.2f}")

def create_visualizations(df: pd.DataFrame, provider: str = ""):
    """Create visualizations of the results."""
    if df.empty:
        return
    
    try:
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        provider_suffix = f"_{provider}" if provider else ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        provider_title = f" - {provider.upper()}" if provider else ""
        fig.suptitle(f'Compression Experiment Analysis{provider_title}', fontsize=16)
        
        # 1. Compression ratio distribution
        axes[0, 0].hist(df['achieved_compression_ratio'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Compression Ratio Distribution')
        axes[0, 0].set_xlabel('Compression Ratio')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Semantic score distribution
        axes[0, 1].hist(df['composite_score'], bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Semantic Score Distribution')
        axes[0, 1].set_xlabel('Composite Semantic Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Compression vs Semantic scatter
        axes[1, 0].scatter(df['achieved_compression_ratio'], df['composite_score'], alpha=0.6)
        axes[1, 0].set_title('Compression vs Semantic Preservation')
        axes[1, 0].set_xlabel('Compression Ratio')
        axes[1, 0].set_ylabel('Semantic Score')
        
        # 4. Semantic metrics comparison
        semantic_cols = ['embedding_similarity', 'bleu_score', 'keyword_preservation', 'llm_similarity']
        available_cols = [col for col in semantic_cols if col in df.columns]
        if available_cols:
            df[available_cols].boxplot(ax=axes[1, 1])
            axes[1, 1].set_title('Semantic Metrics Comparison')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filename = f'compression_analysis{provider_suffix}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualizations saved to: {filename}")
        
    except ImportError:
        print("📊 Matplotlib not available for visualizations")
    except Exception as e:
        print(f"📊 Error creating visualizations: {e}")

def create_comparative_visualizations(results: dict):
    """Create comparative visualizations between providers."""
    if len(results) < 2:
        return
    
    try:
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('OpenAI vs Gemini: Compression Experiment Comparison', fontsize=16)
        
        providers = list(results.keys())
        colors = ['skyblue', 'lightcoral']
        
        # 1. Compression ratio comparison
        for i, provider in enumerate(providers):
            df = results[provider]
            axes[0, 0].hist(df['achieved_compression_ratio'], bins=15, alpha=0.7, 
                          color=colors[i], label=provider.upper())
        axes[0, 0].set_title('Compression Ratio Distribution')
        axes[0, 0].set_xlabel('Compression Ratio')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Semantic score comparison
        for i, provider in enumerate(providers):
            df = results[provider]
            axes[0, 1].hist(df['composite_score'], bins=15, alpha=0.7, 
                          color=colors[i], label=provider.upper())
        axes[0, 1].set_title('Semantic Score Distribution')
        axes[0, 1].set_xlabel('Composite Semantic Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Box plot comparison - Compression
        compression_data = [results[provider]['achieved_compression_ratio'] for provider in providers]
        axes[0, 2].boxplot(compression_data, labels=[p.upper() for p in providers])
        axes[0, 2].set_title('Compression Ratio Comparison')
        axes[0, 2].set_ylabel('Compression Ratio')
        
        # 4. Box plot comparison - Semantic scores
        semantic_data = [results[provider]['composite_score'] for provider in providers]
        axes[1, 0].boxplot(semantic_data, labels=[p.upper() for p in providers])
        axes[1, 0].set_title('Semantic Score Comparison')
        axes[1, 0].set_ylabel('Semantic Score')
        
        # 5. Scatter plot overlay
        for i, provider in enumerate(providers):
            df = results[provider]
            axes[1, 1].scatter(df['achieved_compression_ratio'], df['composite_score'], 
                             alpha=0.6, color=colors[i], label=provider.upper())
        axes[1, 1].set_title('Compression vs Semantic Preservation')
        axes[1, 1].set_xlabel('Compression Ratio')
        axes[1, 1].set_ylabel('Semantic Score')
        axes[1, 1].legend()
        
        # 6. Efficiency comparison (semantic/compression)
        efficiency_data = []
        for provider in providers:
            df = results[provider]
            efficiency = df['composite_score'] / df['achieved_compression_ratio']
            efficiency_data.append(efficiency)
        axes[1, 2].boxplot(efficiency_data, labels=[p.upper() for p in providers])
        axes[1, 2].set_title('Efficiency Score Comparison')
        axes[1, 2].set_ylabel('Semantic Score / Compression Ratio')
        
        plt.tight_layout()
        plt.savefig('compression_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Comparative visualizations saved to: compression_comparison.png")
        
    except Exception as e:
        print(f"📊 Error creating comparative visualizations: {e}")

def export_filtered_results(df: pd.DataFrame, output_file: str = None, provider: str = ""):
    """Export filtered/processed results for further analysis."""
    if df.empty:
        return
    
    if output_file is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        provider_suffix = f"_{provider}" if provider else ""
        output_file = f"filtered_results{provider_suffix}_{timestamp}.csv"
    
    # Create a clean export with key metrics
    columns_to_export = [
        'sentence_index', 'original', 'compressed', 'decompressed',
        'achieved_compression_ratio', 'compression_percentage', 'composite_score',
        'semantic_interpretation', 'embedding_similarity', 'llm_similarity',
        'compression_success', 'model_used', 'provider'
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in columns_to_export if col in df.columns]
    export_df = df[available_columns].copy()
    
    export_df.to_csv(output_file, index=False)
    print(f"📄 Clean results exported to: {output_file}")

def main():
    """Main analysis function."""
    print("🔍 COMPRESSION EXPERIMENT ANALYSIS")
    print("="*50)
    
    # Check for command line argument
    csv_filename = None
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
        print(f"📁 Using specified file: {csv_filename}")
        
        # Single file analysis
        df = load_results(csv_filename)
        if df.empty:
            return
        
        provider = "openai" if "openai" in csv_filename else ("gemini" if "gemini" in csv_filename else "")
        
        # Run analyses
        print_summary_statistics(df, provider)
        show_best_compressions(df, 3, provider)
        show_best_semantic_preservation(df, 3, provider)
        analyze_compression_vs_semantic_tradeoff(df, provider)
        
        # Create visualizations
        create_visualizations(df, provider)
        
        # Export clean results
        export_filtered_results(df, provider=provider)
    else:
        # Try to load both OpenAI and Gemini results for comparison
        print("📁 Looking for both OpenAI and Gemini results...")
        results = load_comparative_results()
        
        if len(results) == 0:
            print("❌ No experiment results found!")
            return
        elif len(results) == 1:
            # Only one provider available
            provider, df = next(iter(results.items()))
            print(f"📊 Only {provider.upper()} results found. Running single analysis...")
            
            print_summary_statistics(df, provider)
            show_best_compressions(df, 3, provider)
            show_best_semantic_preservation(df, 3, provider)
            analyze_compression_vs_semantic_tradeoff(df, provider)
            create_visualizations(df, provider)
            export_filtered_results(df, provider=provider)
        else:
            # Both providers available - comparative analysis
            print(f"🎯 Found results from {len(results)} providers. Running comparative analysis...")
            
            # Individual analyses
            for provider, df in results.items():
                print_summary_statistics(df, provider)
                create_visualizations(df, provider)
                export_filtered_results(df, provider=provider)
            
            # Comparative analysis
            print_comparative_summary(results)
            create_comparative_visualizations(results)
    
    print(f"\n✅ Analysis complete!")

def show_file_menu():
    """Interactive file selection if multiple CSV files exist."""
    available = list_available_csvs()
    
    if not available:
        print("❌ No CSV files found!")
        return None
    
    if len(available) == 1:
        return available[0]
    
    print("\n📁 Multiple CSV files found:")
    for i, file in enumerate(available):
        mtime = os.path.getmtime(file)
        from datetime import datetime
        time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        size = os.path.getsize(file)
        provider = "OpenAI" if "openai" in file else ("Gemini" if "gemini" in file else "Unknown")
        print(f"   {i+1}. {file}")
        print(f"      Provider: {provider} | Modified: {time_str} | Size: {size} bytes")
    
    try:
        choice = input(f"\nSelect file (1-{len(available)}) or press Enter for latest: ")
        if choice.strip() == "":
            return available[0]  # Latest file
        
        idx = int(choice) - 1
        if 0 <= idx < len(available):
            return available[idx]
        else:
            print("❌ Invalid selection")
            return None
    except ValueError:
        print("❌ Invalid input")
        return None

if __name__ == "__main__":
    # Check if running interactively or with arguments
    if len(sys.argv) == 1 and sys.stdin.isatty():
        # Interactive mode - show file menu if multiple files exist
        available = list_available_csvs()
        if len(available) > 1:
            selected_file = show_file_menu()
            if selected_file:
                sys.argv.append(selected_file)  # Add to argv for main()
    
    main() 