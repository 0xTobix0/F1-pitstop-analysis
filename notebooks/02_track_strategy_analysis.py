"""
Track Strategy Analysis for F1 Pitstop Optimization
This script analyzes track-specific characteristics and tests the strategy optimizer.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.track_strategy import TrackStrategyOptimizer

# Set up visualization style
plt.style.use('bmh')  # Using a built-in style that's similar to seaborn
pd.set_option('display.max_columns', None)

def load_and_prepare_data():
    """Load and prepare the F1 dataset for analysis."""
    data_path = os.path.join(project_root, 'data', 'f1dataset.csv')
    df = pd.read_csv(data_path)
    # Filter for recent seasons (2015-2021)
    df = df[df['year'] >= 15]
    return df

def analyze_track_characteristics(df):
    """Analyze and visualize track-specific characteristics."""
    track_stats = []
    
    for track in df['name'].unique():
        track_data = df[df['name'] == track]
        optimizer = TrackStrategyOptimizer(track_data)
        track_stats.append({
            'name': track,
            **vars(optimizer.track_characteristics)
        })
    
    track_stats_df = pd.DataFrame(track_stats)
    return track_stats_df

def plot_track_characteristics(track_stats_df, strategy_df):
    """Create visualizations for track characteristics and strategy patterns."""
    # Create two separate figures for better organization
    
    # Figure 1: Track Performance Analysis
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('Track Performance Analysis', fontsize=16)
    
    # Plot 1: Tire Degradation vs Lap Time with Safety Car probability as color
    scatter = axes1[0,0].scatter(track_stats_df['tire_degradation'], 
                                track_stats_df['avg_lap_time'],
                                c=track_stats_df['sc_probability'],
                                cmap='viridis',
                                s=100)
    for i, txt in enumerate(track_stats_df['name']):
        axes1[0,0].annotate(txt.split()[0], 
                           (track_stats_df['tire_degradation'].iloc[i], 
                            track_stats_df['avg_lap_time'].iloc[i]))
    axes1[0,0].set_xlabel('Tire Degradation (ms/lap)')
    axes1[0,0].set_ylabel('Average Lap Time (s)')
    axes1[0,0].set_title('Track Performance Map')
    plt.colorbar(scatter, ax=axes1[0,0], label='Safety Car Probability')
    
    # Plot 2: Strategy Distribution
    strategy_counts = strategy_df['recommended_stops'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_counts)))
    wedges, texts, autotexts = axes1[0,1].pie(strategy_counts, 
                                             labels=[f'{n}-Stop Strategy' for n in strategy_counts.index],
                                             autopct='%1.1f%%', 
                                             colors=colors,
                                             explode=[0.05]*len(strategy_counts))
    axes1[0,1].set_title('Pit Stop Strategy Distribution')
    
    # Plot 3: Track Characteristics Correlation
    features = ['tire_degradation', 'pit_loss_time', 'overtaking_difficulty', 'sc_probability']
    correlation = track_stats_df[features].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, ax=axes1[1,0])
    axes1[1,0].set_title('Track Characteristics Correlation')
    
    # Plot 4: Track Difficulty Score
    # Calculate a composite difficulty score
    track_stats_df['difficulty_score'] = (
        track_stats_df['overtaking_difficulty'] * 0.3 +
        abs(track_stats_df['tire_degradation']) / abs(track_stats_df['tire_degradation'].max()) * 0.3 +
        track_stats_df['pit_loss_time'] / track_stats_df['pit_loss_time'].max() * 0.2 +
        track_stats_df['sc_probability'] * 0.2
    )
    
    difficulty_plot = track_stats_df.sort_values('difficulty_score', ascending=True)
    bars = axes1[1,1].barh(difficulty_plot['name'].str.split().str[0], 
                          difficulty_plot['difficulty_score'])
    axes1[1,1].set_title('Track Difficulty Score')
    axes1[1,1].set_xlabel('Composite Difficulty Score')
    
    # Color bars by recommended stops
    stops_map = {row['track']: row['recommended_stops'] 
                for _, row in strategy_df.iterrows()}
    colors = plt.cm.Set3(np.linspace(0, 1, len(set(stops_map.values()))))
    color_map = {stops: color for stops, color in 
                zip(sorted(set(stops_map.values())), colors)}
    
    for bar, track in zip(bars, difficulty_plot['name']):
        bar.set_color(color_map[stops_map[track]])
    
    # Add legend for strategy colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[stops],
                           label=f'{stops}-Stop Strategy')
                      for stops in sorted(set(stops_map.values()))]
    axes1[1,1].legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    output_path1 = os.path.join(project_root, 'track_performance.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Strategy Analysis
    fig2, axes2 = plt.subplots(2, 1, figsize=(15, 12))
    fig2.suptitle('Track Strategy Analysis', fontsize=16)
    
    # Plot 5: Strategy vs Track Characteristics
    from sklearn.preprocessing import StandardScaler
    features_for_analysis = ['tire_degradation', 'pit_loss_time', 
                           'overtaking_difficulty', 'sc_probability']
    X = StandardScaler().fit_transform(track_stats_df[features_for_analysis])
    
    # PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create scatter plot with PCA results
    scatter = axes2[0].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=strategy_df['recommended_stops'],
                              cmap='Set3',
                              s=100)
    
    # Add track names as annotations
    for i, txt in enumerate(track_stats_df['name']):
        axes2[0].annotate(txt.split()[0], (X_pca[i, 0], X_pca[i, 1]))
    
    axes2[0].set_xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
    axes2[0].set_ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
    axes2[0].set_title('Track Clustering by Characteristics')
    
    # Add colorbar for strategy
    cbar = plt.colorbar(scatter, ax=axes2[0], ticks=sorted(strategy_df['recommended_stops'].unique()))
    cbar.set_label('Number of Pit Stops')
    
    # Plot 6: Feature Importance for Strategy
    # Calculate feature importance using PCA components
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=features_for_analysis
    )
    
    sns.heatmap(feature_importance, annot=True, cmap='RdYlBu', center=0,
                ax=axes2[1], cbar_kws={'label': 'PCA Loading Score'})
    axes2[1].set_title('Feature Importance in Strategy Determination')
    
    plt.tight_layout()
    output_path2 = os.path.join(project_root, 'strategy_analysis.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_optimal_strategies(df):
    """Analyze and compare optimal strategies across tracks."""
    strategy_analysis = []
    
    for track in df['name'].unique():
        track_data = df[df['name'] == track]
        optimizer = TrackStrategyOptimizer(track_data)
        
        # Get strategy recommendations for standard race distance
        strategy = optimizer.recommend_strategy(race_distance=60)
        
        strategy_analysis.append({
            'track': track,
            'recommended_stops': strategy['recommended_stops'],
            'notes': '; '.join(strategy['strategy_notes'])
        })
    
    return pd.DataFrame(strategy_analysis)

def main():
    """Main analysis function."""
    print("Loading data...")
    df = load_and_prepare_data()
    
    print("\nAnalyzing track characteristics...")
    track_stats = analyze_track_characteristics(df)
    print("\nTrack Statistics Summary:")
    print(track_stats.describe())
    
    print("\nGenerating visualizations...")
    strategy_analysis = analyze_optimal_strategies(df)
    plot_track_characteristics(track_stats, strategy_analysis)
    
    print("\nAnalyzing optimal strategies...")
    print("\nStrategy Analysis:")
    print(strategy_analysis)
    
    # Save results
    output_dir = os.path.join(project_root, 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    track_stats.to_csv(os.path.join(output_dir, 'track_characteristics.csv'), index=False)
    strategy_analysis.to_csv(os.path.join(output_dir, 'track_strategies.csv'), index=False)
    
    print("\nAnalysis complete! Results saved in the data directory.")

if __name__ == "__main__":
    main()
