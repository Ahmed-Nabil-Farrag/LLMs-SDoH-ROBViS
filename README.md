# SDoH Risk Assessment Visualizer

This repository contains a Python toolkit for assessing and visualizing the risk of bias in studies that use Large Language Models (LLMs) for identifying Social Determinants of Health (SDoH).

## Overview

The SDoH Risk Assessment Visualizer provides comprehensive tools to analyze and visualize the quality of research studies based on multiple assessment domains. The toolkit includes:

- Risk level calculation based on established criteria
- Multiple visualization types for analyzing study quality
- Configurable assessment framework with priority levels
- Publication-ready charts following scientific journal standards

## Installation

### Requirements

- Python 3.8+
- Required packages:
  - numpy
  - matplotlib
  - seaborn
  - pandas
  - adjustText (automatically installed if not available)

### Setup

```bash
git clone https://github.com/yourusername/sdoh-risk-assessment.git
cd sdoh-risk-assessment
pip install -r requirements.txt
```

## Usage

```python
import sdoh_visualizer

# Define your study data
study_data = {
    'studies': ["Study1 2023", "Study2 2024", ...],
    'categories': ["Error Analysis", "Fairness Assessment", ...],
    'priority_levels': ["High", "High", "Medium", ...],
    'max_points': [3, 3, 2, ...],
    'scores': {
        "Error Analysis": [3, 0, ...],
        "Fairness Assessment": [3, 0, ...],
        ...
    }
}

# Initialize style settings
style = PlotStyle()

# Process data
processor = DataProcessor(study_data)
processed_data = processor.processed_data

# Create visualizations
visualizer = Visualizer(style)
visualizer.plot_domain_completion(processed_data)
visualizer.plot_risk_assessment_heatmap(processed_data)
visualizer.plot_risk_distribution(processed_data)
visualizer.plot_risk_by_year(processed_data)
visualizer.plot_domain_correlation(processed_data)
visualizer.plot_score_scatter(processed_data)
```

## Key Components

### PlotStyle

The `PlotStyle` class manages visualization styling to ensure consistent, publication-ready charts:

```python
style = PlotStyle()
```

Features:
- Professional color palette based on scientific journal standards
- Custom risk level color mapping
- Pre-configured plot parameters for consistent appearance

### DataProcessor

The `DataProcessor` class handles data preparation and risk assessment calculations:

```python
processor = DataProcessor(study_data)
processed_data = processor.processed_data
```

Features:
- Converts raw score data into structured format
- Calculates risk levels based on framework criteria
- Normalizes data for various visualization types
- Categorizes assessment domains by priority level

### Visualizer

The `Visualizer` class contains methods for creating various visualization types:

```python
visualizer = Visualizer(style)
```

#### Available visualizations:

1. **Domain Completion Chart**
   ```python
   visualizer.plot_domain_completion(processed_data)
   ```
   Shows completion rate for each assessment domain, categorized by priority level.

2. **Risk Assessment Heatmap**
   ```python
   visualizer.plot_risk_assessment_heatmap(processed_data)
   ```
   Traffic light visualization showing all studies' scores across domains with circle size indicating maximum possible points.

3. **Risk Distribution Chart**
   ```python
   visualizer.plot_risk_distribution(processed_data)
   ```
   Bar chart showing the distribution of studies across risk levels.

4. **Risk by Publication Year**
   ```python
   visualizer.plot_risk_by_year(processed_data)
   ```
   Stacked bar chart showing risk level distribution by publication year.

5. **Domain Correlation Heatmap**
   ```python
   visualizer.plot_domain_correlation(processed_data)
   ```
   Correlation heatmap showing relationships between assessment domains.

6. **Score Scatter Plot**
   ```python
   visualizer.plot_score_scatter(processed_data)
   ```
   Advanced scatter plot showing relationship between high priority scores and total scores with risk zone visualization.

## Risk Assessment Framework

The tool uses the following criteria to assess risk levels:

- **Low Risk**: ≥14 total points AND ≥6 high priority points AND ≥3 medium priority points
- **Medium Risk**: ≥9 total points AND ≥4 high priority points AND ≥2 medium priority points
- **High Risk**: Below these thresholds

## Example Output

The toolkit generates multiple visualization files including:
- domain_completion_by_priority.png/pdf
- risk_assessment_heatmap.png/pdf
- risk_level_distribution.png/pdf
- risk_distribution_by_year.png/pdf
- domain_correlation.png/pdf
- high_priority_vs_total_score.png/svg

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
