# Evaluation Metrics

## Symmetric Percent Change (SPC)

To compare models against a Graphviz baseline, we use **Symmetric Percent Change (SPC)**:

$$\text{SPC} = 100\% \times \frac{1}{N_t} \sum_{i=0}^{N_t} \frac{D_i - G_i}{\max(D_i, G_i)}$$

where $D_i$ is your model's crossing count and $G_i$ is the Graphviz baseline count for the $i$-th graph. SPC ranges from -100% to +100%; **negative means your model outperforms the baseline**.

### SPC Results (vs Neato)

| Model | SPC |
|---|---|
| SFDP | -3.14% |
| SmartGD | -3.60% |
| Diff GD | -19.97% |

### Usage

**From the command line:**
```bash
python metrics.py --csv baseline_metrics.csv --baseline neato_xing
```

**In code:**
```python
from metrics import spc, report_spc

# compute SPC from two arrays
score = spc(my_model_crossings, neato_crossings)

# print a full report from a CSV
report_spc("baseline_metrics.csv", baseline="neato_xing")
```
