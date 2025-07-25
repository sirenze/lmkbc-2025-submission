# LM-KBC: Knowledge Base Construction from Pre-trained Language Models (4th Edition)

More details about the LM-KBC challenge at ISWC 2025 - https://lm-kbc.github.io/challenge2025/.

This repository contains:

- From the organisers: 
    - The dataset for the challenge (train, val, test)
    - Evaluation and baseline scripts
- Modifications:
    - Submitted scripts 
        - Files modified: models/baseline_generation_model.py, 
        models/baseline_qwen_3_model.py, configs/baseline-qwen-3.yaml, baseline.py,
        prompt_templates/prompts
    - Outputs

## Table of contents

1. [Challenge overview](#challenge-overview)
2. [Dataset](#dataset)
3. [Evaluation metrics](#evaluation-metrics)
4. [Getting started](#getting-started)
    - [Initial Setup](#initial-setup)
    - [Predictions](#generating-and-evaluating-predictions)
5. [Comparisons](#comparisons)
    - [Baselines: Qwen3-8B](#baselines-qwen3-8b)
    - [Submission](#submission-qwen3-8b)
6. [CodaLab](#codalab-leaderboards)

## Challenge overview 

https://github.com/lm-kbc/dataset2025?tab=readme-ov-file#challenge-overview

## Dataset

From https://github.com/lm-kbc/dataset2025?tab=readme-ov-file#dataset : 

Number of unique subject-entities in the data splits.

<table>
<thead>
    <tr>
        <th>Relation</th>
        <th>Train</th>
        <th>Val</th>
        <th>Test</th>
        <th>Special features</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td>countryLandBordersCountry</td>
        <td>68</td>
        <td>68</td>
        <td>67</td>
        <td>Null values possible</td>
    </tr>
    <tr>
        <td>personHasCityOfDeath</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Null values possible</td>
    </tr>
    <tr>
        <td>hasCapacity</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Object is numeric</td>
    </tr>
    <tr>
        <td>awardWonBy</td>
        <td>10</td>
        <td>10</td>
        <td>10</td>
        <td>Many objects per subject</td>
    </tr>
    <tr>
        <td>companyTradesAtStockExchange</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Null values possible</td>
    </tr>
        <tr>
        <td>hasArea</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Object is numeric (square km)</td>
    </tr>
</tbody>
</table>

## Evaluation metrics

From https://github.com/lm-kbc/dataset2025?tab=readme-ov-file#evaluation-metrics : 

We evaluate the predictions using macro precision, recall, and F1-score.
See the evaluation script ([evaluate.py](evaluate.py)) for more details.

```bash
python evaluate.py \
  -g data/val.jsonl \
  -p data/testrun-XYZ.jsonl
```

Parameters: ``-g`` (the ground truth file), ``-p`` (the prediction file).

## Getting started

### Initial Setup

From https://github.com/lm-kbc/dataset2025?tab=readme-ov-file#setup : 

1. Clone this repository:

    ```bash
    mkdir lm-kbc-2025
    cd lm-kbc-2025
    # git clone https://github.com/lm-kbc/dataset2025.git
    git clone https://github.com/sirenze/lmkbc-2025-submission.git
    cd dataset2025
    ```

2. Create a virtual environment and install the requirements:

    ```bash
    conda create -n lm-kbc-2025 python=3.11
    ```

    ```bash
    conda activate lm-kbc-2025
    pip install -r requirements.txt
    ```

### Generating and Evaluating Predictions

3. Since the prediction file follows a similar structure as the baseline, the
same commands can be used, albeit with the correct file names like below:

From https://github.com/lm-kbc/dataset2025?tab=readme-ov-file#baselines :

Config file: [configs/my-qwen-3.yaml](configs/my-qwen-3.yaml)

```bash
python my_baseline.py -c configs/my-qwen-3.yaml -i data/val.jsonl
python evaluate.py -g data/val.jsonl -p outputs/my-qwen-3.jsonl
```

And similarly for the test set :

Config file: [configs/my-qwen-3.yaml](configs/my-qwen-3.yaml)

```bash
python my_baseline.py -c configs/my-qwen-3.yaml -i data/test.jsonl -o outputs/test-outputs.jsonl
```

## Comparisons

### Baselines: Qwen3-8B

From https://github.com/lm-kbc/dataset2025?tab=readme-ov-file#baseline-qwen3-8b : 


**Results (validation, without entity disambiguation):**

```text
                              macro-p  macro-r  macro-f1  micro-p  micro-r  micro-f1  avg. #preds  #empty preds
awardWonBy                      0.173    0.027     0.043    0.173    0.021     0.038       18.500             0
companyTradesAtStockExchange    0.230    0.561     0.215    0.248    0.329     0.283        1.050             0
countryLandBordersCountry       0.653    0.873     0.628    0.787    0.782     0.784        2.618             0
hasArea                         0.180    0.180     0.180    0.180    0.180     0.180        1.000             0
hasCapacity                     0.030    0.030     0.030    0.030    0.030     0.030        1.000             0
personHasCityOfDeath            0.100    0.550     0.100    0.100    0.179     0.128        1.000             0
*** All Relations ***           0.209    0.401     0.200    0.298    0.114     0.165        1.607             0
```

**Results (test, without entity disambiguation):**

```text
                              macro-p  macro-r  macro-f1  micro-p  micro-r  micro-f1  avg. #preds  #empty preds
awardWonBy                      0.240    0.090     0.117    0.261    0.072     0.113       15.300             0
companyTradesAtStockExchange    0.185    0.591     0.167    0.208    0.286     0.240        1.060             0
countryLandBordersCountry       0.768    0.812     0.702    0.892    0.764     0.823        3.463             0
hasArea                         0.240    0.240     0.240    0.240    0.240     0.240        1.000             0
hasCapacity                     0.040    0.040     0.040    0.040    0.040     0.040        1.000             0
personHasCityOfDeath            0.080    0.650     0.080    0.078    0.186     0.110        1.020             0
*** All Relations ***           0.227    0.435     0.212    0.385    0.267     0.315        1.662             0
```

### Submission: Qwen3-8B

**Results (validation, without entity disambiguation):**

<pre>
                              macro-p  macro-r  macro-f1
awardWonBy                      <b>0.193</b>    <b>0.057</b>     <b>0.075</b>
companyTradesAtStockExchange    <b>0.240</b>    <b>0.573</b>     <b>0.225</b>
countryLandBordersCountry       <b>0.674</b>    <b>0.909</b>     <b>0.661</b>
hasArea                         <b>0.200</b>    <b>0.200</b>     <b>0.200</b>
hasCapacity                     <b>0.080</b>    <b>0.080</b>     <b>0.080</b>
personHasCityOfDeath            <b>0.110</b>    <b>0.560</b>     <b>0.110</b>
*** All Relations ***           <b>0.231</b>    <b>0.426</b>     <b>0.224</b>
</pre>

**Results (test, without entity disambiguation):**

<pre>
                              macro-p  macro-r  macro-f1
awardWonBy                      0.191    <b>0.159</b>     <b>0.139</b>
companyTradesAtStockExchange    0.178    <b>0.604</b>     <b>0.170</b>
countryLandBordersCountry       0.737    <b>0.817</b>     0.690
hasArea                         <b>0.260</b>    <b>0.260</b>     <b>0.260</b>
hasCapacity                     <b>0.150</b>    <b>0.150</b>     <b>0.150</b>
personHasCityOfDeath            <b>0.090</b>    <b>0.660</b>     <b>0.090</b>
*** All Relations ***           <b>0.249</b>    <b>0.469</b>     <b>0.240</b>
</pre>

## CodaLab Leaderboards

- validation - https://codalab.lisn.upsaclay.fr/competitions/22964
- test - https://codalab.lisn.upsaclay.fr/competitions/23218
