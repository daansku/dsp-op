| Model | Features | F1, normal test days | F1, all test days | F1 at 0.5, all test days | AP, normal test days | Laundering in top 1,000 |
|---|---|---|---|---|---|---|
| LightGBM | basic | 12.7 ± 2.3 | 23.3 ± 4.9 | 14.3 ± 2.1 | 5.6 ± 1.1 | 13.0 ± 1.0 |
| GIN | basic | 26.9 ± 2.4 | 33.2 ± 1.8 | 32.8 ± 1.2 | 21.3 ± 3.0 | 30.7 ± 2.3 |
| PNA | basic | 51.4 ± 1.6 | 61.1 ± 1.7 | 60.5 ± 2.1 | 45.3 ± 1.1 | 48.7 ± 1.1 |
| Logistic regression | engineered | 35.0 ± 1.3 | 45.5 ± 1.2 | 42.4 ± 0.8 | 29.5 ± 0.7 | 39.0 ± 0.6 |
| Random forest | engineered | 55.6 ± 0.3 | 67.9 ± 0.6 | 66.4 ± 1.1 | 52.8 ± 0.3 | 52.8 ± 0.6 |
| LightGBM | engineered | 54.8 ± 3.8 | 66.3 ± 2.9 | 49.2 ± 5.0 | 49.8 ± 3.4 | 51.3 ± 2.0 |
| GIN | engineered | 63.6 ± 1.2 | 75.4 ± 1.0 | 75.6 ± 1.1 | 58.6 ± 2.4 | 56.5 ± 0.7 |
| PNA | engineered | 62.2 ± 1.8 | 73.9 ± 0.9 | 71.8 ± 1.5 | 60.7 ± 0.6 | 57.2 ± 0.4 |
