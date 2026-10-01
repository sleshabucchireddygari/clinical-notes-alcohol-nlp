# Results

Test set: 150 synthetic HIV-clinic notes (gold labels known).

## Overall

| Method                       |   Notes |   Precision |   Recall |   F1 |   False alarms (FP) |   Missed (FN) |
|:-----------------------------|--------:|------------:|---------:|-----:|--------------------:|--------------:|
| Keyword search               |     150 |        0.41 |     1    | 0.58 |                  75 |             0 |
| Rules + negation             |     150 |        1    |     0.69 | 0.82 |                   0 |            16 |
| TF-IDF + logistic regression |     150 |        0.94 |     0.65 | 0.77 |                   2 |            18 |
| LLM (llama3.2:3b, local)     |     150 |        0.77 |     0.98 | 0.86 |                  15 |             1 |

## Accuracy by type of sentence

| category        | Keyword search   | Rules + negation   | TF-IDF + logistic regression   | LLM (llama3.2:3b, local)   |   n notes |
|:----------------|:-----------------|:-------------------|:-------------------------------|:---------------------------|----------:|
| negated         | 0%               | 100%               | 94%                            | 100%                       |        35 |
| no_mention      | 100%             | 100%               | 100%                           | 100%                       |        23 |
| past_use        | 0%               | 100%               | 100%                           | 53%                        |        15 |
| light_drinking  | 0%               | 100%               | 100%                           | 100%                       |        15 |
| binge_drinking  | 100%             | 57%                | 86%                            | 100%                       |        14 |
| heavy_drinking  | 100%             | 58%                | 58%                            | 100%                       |        12 |
| tricky_positive | 100%             | 82%                | 82%                            | 100%                       |        11 |
| family_history  | 0%               | 100%               | 100%                           | 20%                        |        10 |
| indirect_signs  | 100%             | 67%                | 33%                            | 100%                       |         9 |
| positive_screen | 100%             | 100%               | 50%                            | 83%                        |         6 |

## LLM evidence check (hallucination test)

The LLM gave an evidence quote for 131 notes. 130 of those quotes were found word-for-word in the note; 1 were not (paraphrased or made up).
