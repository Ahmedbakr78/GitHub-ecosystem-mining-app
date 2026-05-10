# GitHub Repository Mining - Data Mining Report

Generated: 2026-05-10T22:15:33.369904+00:00

## Objective

This project mines GitHub repository data to discover software development trends, influential technologies, common technology stacks, and semantically similar projects.

## Dataset

- Repositories analyzed: 5,049
- Unique technologies/topics: 10,470
- Technology graph edges exported: 1,500
- Data source distribution: {'github_api': 5049}
- Embedding method: BERT/NumPy BERT-Tiny: phob0s/bert-tiny

## Preprocessing

Repository descriptions, README snippets, languages, and topics were cleaned and normalized. Topics were merged with programming languages to create a transaction-style `tech_stack` for each repository.

## Association Rule Mining

Apriori was applied to repository technology stacks. The pipeline exports support, support count, confidence, lift, conviction, and leverage.

| Antecedent | Consequent | Support | Confidence | Lift |
|---|---|---:|---:|---:|
| deep-reinforcement-learning | reinforcement-learning | 0.01208 | 0.64894 | 12.89952 |
| data-analysis | data-science + machine-learning | 0.01485 | 0.8427 | 8.40865 |
| data-analysis + machine-learning | data-science | 0.01485 | 0.85227 | 8.22777 |
| data-analysis | data-science | 0.01485 | 0.8427 | 8.13533 |
| keras + machine-learning | tensorflow | 0.01743 | 0.74576 | 6.42552 |
| keras | deep-learning + tensorflow | 0.02416 | 0.56744 | 6.1481 |
| deep-learning + tensorflow | keras | 0.02416 | 0.2618 | 6.1481 |
| keras | machine-learning + tensorflow | 0.01743 | 0.4093 | 5.52558 |
| keras | python + tensorflow | 0.01822 | 0.42791 | 5.38779 |
| keras | tensorflow | 0.02654 | 0.62326 | 5.37 |
| keras + python | tensorflow | 0.01822 | 0.62162 | 5.35592 |
| deep-learning + keras | tensorflow | 0.02416 | 0.61929 | 5.33582 |

## Link Analysis

Technologies are nodes. Weighted edges represent co-occurrence inside repository tech stacks. PageRank identifies central ecosystem technologies, while HITS identifies hub and authority roles.

| Rank | Technology | PageRank | Authority | Hub | Repositories |
|---:|---|---:|---:|---:|---:|
| 1 | machine-learning | 0.04558222 | 0.4555015 | 0.4555015 | 3110 |
| 2 | python | 0.04354603 | 0.50902881 | 0.50902881 | 3081 |
| 3 | deep-learning | 0.04324727 | 0.4928393 | 0.4928393 | 3157 |
| 4 | pytorch | 0.01650566 | 0.26666517 | 0.26666517 | 1104 |
| 5 | artificial-intelligence | 0.01548367 | 0.201389 | 0.201389 | 976 |
| 6 | jupyter notebook | 0.01089159 | 0.15715649 | 0.15715649 | 810 |
| 7 | computer-vision | 0.01063691 | 0.16456304 | 0.16456304 | 703 |
| 8 | tensorflow | 0.00877297 | 0.1496584 | 0.1496584 | 586 |
| 9 | data-science | 0.00858805 | 0.12836654 | 0.12836654 | 523 |
| 10 | natural-language-processing | 0.00776842 | 0.12512277 | 0.12512277 | 518 |
| 11 | neural-network | 0.00569921 | 0.09192347 | 0.09192347 | 357 |
| 12 | large-language-model | 0.00431743 | 0.05707241 | 0.05707241 | 249 |

## Semantic/BERT Analysis

Repository text is classified into 13 categories. The default pipeline uses real pretrained BERT-Tiny embeddings through a lightweight NumPy inference path. A SentenceTransformer model can also be supplied with `--semantic-model` when PyTorch is available.

- Average classification confidence: 0.5599
- Largest categories: [('AI/ML Infrastructure', 3760), ('Generative AI', 523), ('Data Science', 309), ('Natural Language Processing', 124), ('Computer Vision', 99)]

## Key Insights

- machine-learning is the most central technology by PageRank, appearing in 3110 repositories.
- The strongest association rule is deep-reinforcement-learning -> reinforcement-learning with lift 12.89952 and confidence 0.64894.
- AI/ML Infrastructure is the largest semantic category with 3760 repositories.

## Decision Support

The mined rules and graph rankings can be used to recommend technology stacks, identify trending repository domains, and find similar projects for learning or reuse.
