# SDoH Risk Assessment Toolkit

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive toolkit for quantitative assessment and visualization of bias risk in Large Language Models (LLMs) for Social Determinants of Health (SDoH) identification across clinical studies.

## Overview

The SDoH Risk Assessment Toolkit provides a standardized framework for analyzing research quality across multiple assessment domains. This evidence-based approach enables systematic evaluation of methodological rigor in studies utilizing LLMs for SDoH identification.

The toolkit implements a multi-domain risk assessment matrix with configurable priority levels and weighted scoring, generating publication-ready visualizations that conform to scientific journal standards.

## Features

- **Standardized Assessment Framework**: Implements a validated multi-domain evaluation methodology 
- **Prioritized Risk Calculation**: Weights domains by importance (High/Medium/Standard priority)
- **Comprehensive Visualization Suite**: Generates six distinct visualization types for thorough analysis
- **Scientific Publication Standards**: Follows visual style guidelines for academic journals
- **Customizable Assessment Criteria**: Adaptable framework for different research contexts

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Dependencies

```bash
pip install numpy matplotlib seaborn pandas adjustText
```

The toolkit will automatically install any missing dependencies on first run.

## Usage

### Basic Implementation

```python
from sdoh_risk_toolkit import PlotStyle, DataProcessor, Visualizer

# Initialize the visualization style
style = PlotStyle()

# Define assessment data structure
study_data = {
    'studies': ["Study1 2023", "Study2 2024", ...],
    'categories': ["Error Analysis", "Fairness Assessment", ...],
    'priority_levels': ["High", "High", "Medium", ...],
    'max_points': [3, 3, 2, ...],
    'scores': {
        "Error Analysis": [3, 0, ...],
        "Fairness Assessment": [3, 0, ...],
        # Additional domains...
    }
}

# Process assessment data
processor = DataProcessor(study_data)
processed_data = processor.processed_data

# Generate visualizations
visualizer = Visualizer(style)
visualizer.plot_domain_completion(processed_data)
visualizer.plot_risk_assessment_heatmap(processed_data)
visualizer.plot_risk_distribution(processed_data)
visualizer.plot_risk_by_year(processed_data)
visualizer.plot_domain_correlation(processed_data)
visualizer.plot_score_scatter(processed_data)
```

### Quick Start with Example Data

```python
from sdoh_risk_toolkit import main

# Run with example dataset
main()
```

## Core Components

### PlotStyle

Manages visualization aesthetics and ensures consistent styling across all outputs:

```python
style = PlotStyle()
```

**Key Features:**
- Scientific publication color palette based on journal standards
- Custom-designed risk level colormap
- Pre-configured plot parameters for consistency and readability

### DataProcessor

Transforms raw assessment data into structured analytical formats:

```python
processor = DataProcessor(study_data)
processed_data = processor.processed_data
```

**Key Features:**
- Risk categorization based on validated threshold criteria
- Domain prioritization with differential weighting
- Normalization routines for cross-study comparisons

### Visualizer

Generates publication-ready visualizations for comprehensive analysis:

```python
visualizer = Visualizer(style)
```

#### Visualization Portfolio

1. **Domain Completion Analysis**
   ```python
   visualizer.plot_domain_completion(processed_data)
   ```
   Quantifies methodological completeness across assessment domains, segmented by priority level.

2. **Risk Assessment Matrix**
   ```python
   visualizer.plot_risk_assessment_heatmap(processed_data)
   ```
   Comprehensive heat map displaying normalized scores across all domains with weighted visual indicators.

3. **Risk Level Distribution**
   ```python
   visualizer.plot_risk_distribution(processed_data)
   ```
   Quantitative breakdown of studies across risk categories with percentage annotations.

4. **Temporal Risk Analysis**
   ```python
   visualizer.plot_risk_by_year(processed_data)
   ```
   Longitudinal analysis showing risk level trends across publication years.

5. **Inter-domain Correlation Analysis**
   ```python
   visualizer.plot_domain_correlation(processed_data)
   ```
   Statistical correlation visualization between assessment domains to identify methodological relationships.

6. **Multivariate Risk Visualization**
   ```python
   visualizer.plot_score_scatter(processed_data)
   ```
   Advanced visualization plotting high-priority versus total scores with automated label positioning and risk zone demarcation.

## Risk Assessment Methodology

The toolkit implements a three-tier risk assessment framework with the following criteria:

- **Low Risk**: Studies with ≥14 total points AND ≥6 high-priority points AND ≥3 medium-priority points
- **Medium Risk**: Studies with ≥9 total points AND ≥4 high-priority points AND ≥2 medium-priority points
- **High Risk**: Studies failing to meet the above thresholds

## Output Files

The toolkit generates both PNG and PDF/SVG formats for all visualizations:

| Visualization | Files |
|---------------|-------|
| Domain Completion | domain_completion_by_priority.png/pdf |
| Risk Assessment Matrix | risk_assessment_heatmap.png/pdf |
| Risk Distribution | risk_level_distribution.png/pdf |
| Temporal Risk Analysis | risk_distribution_by_year.png/pdf |
| Domain Correlation | domain_correlation.png/pdf |
| Risk Scatter Plot | high_priority_vs_total_score.png/svg |

## Contributing

We welcome contributions to enhance the toolkit's capabilities. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Submit a Pull Request

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this toolkit in your research, please cite:

```
@software{sdoh_risk_toolkit,
  author = {Author, A.},
  title = {SDoH Risk Assessment Toolkit},
  year = {2023},
  url = {https://github.com/author/sdoh-risk-toolkit},
  version = {1.0.0}
}
```
