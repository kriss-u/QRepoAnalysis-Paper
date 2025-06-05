# Quantum Software Anlaysis Paper

## Dataset

The dataset is available under `data/processed` directory. The dataset files use Apache Parquet format and are partitioned because of their large size.

### Data Structure

```bash
data/processed/
├── dataset_statistics.json          
├── common/                          
│   ├── repos_with_top_packages.txt
│   ├── top_packages_analysis.csv
│   └── top_packages_analysis_with_owner.csv
├── issues.parquet/                  # Issues data by year
│   ├── year=2012/
│   ├── year=2013/
│   ├── ...
│   └── year=2024/
├── rq1/                            
│   └── repos.parquet/              # Repository data by year
│       ├── created_year=2012/
│       ├── created_year=2013/
│       ├── ...
│       └── created_year=2024/
└── rq2/                           
    ├── dataset_metadata.json
    ├── processing_stats.json
    ├── commit_patterns/            # Commit patterns by year and fork status
    │   ├── 2008/
    │   │   ├── is_fork=false/
    │   │   └── is_fork=true/
    │   ├── 2009/
    │   │   ├── is_fork=false/
    │   │   └── is_fork=true/
    │   ├── ...
    │   └── 2024/
    │       ├── is_fork=false/
    │       └── is_fork=true/
    └── commit_patterns_classified/ #  Commit patterns by type, year, and fork status
        ├── Adaptive/
        │   ├── 2008/
        │   │   ├── is_fork=false/
        │   │   └── is_fork=true/
        │   ├── ...
        │   └── 2024/
        │       ├── is_fork=false/
        │       └── is_fork=true/
        ├── Adaptive Perfective/
        │   ├── 2008/ ... 2024/ 
        ├── Corrective/
        │   ├── 2008/ ... 2024/ 
        ├── Corrective Adaptive/
        │   ├── 2008/ ... 2024/ 
        ├── Corrective Adaptive Perfective/
        │   ├── 2008/ ... 2024/ 
        ├── Corrective Perfective/
        │   ├── 2008/ ... 2024/ 
        ├── Perfective/
        │   ├── 2008/ ... 2024/ 
        ├── Unknown/
        │   ├── 2008/ ... 2024/ 
        └── nan/
            ├── 2008/ ... 2024/ 
```

## Scripts

Create a virtual environment and activate it.

```bash
python3 -m venv venv
source venv/bin/activate
```

Install packages.

```bash
pip install pipenv
pipenv install
```

`pre_analysis_scripts/` and `scripts` contain the source code to perform the analysis.

## Loading datasets

`loading_dataset.py` provides a sample code on how to load the parquet datasets.