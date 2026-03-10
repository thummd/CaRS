````markdown
# Fantom Experiments

This repository contains code to run experiments of FANTOM on heteroscedastic and non-Gaussian noise for our NeurIPS submission. 

## Requirements

- Python 3.8+
- numpy
- pandas
- torch ≥1.10
- networkx
- matplotlib
- seaborn
- pyro
- scikit-learn
- PyYAML

Install dependencies with:

```bash
pip install -r requirements.txt
````

## Running Experiments

1. **Heteroscedastic noise**

   ```bash
   python run_fantom_hetero.py 
   ```

2. **Non-Gaussian noise**

   ```bash
   python run_fantom_nongauss.py 
   ```

## Configuration

All data-generation are defined in the YAML files:

* `config_gen_hetero.yaml`
* `config_gen_nongauss.yaml`

Adjust any parameter directly in these files.

