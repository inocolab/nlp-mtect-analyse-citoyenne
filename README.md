## Exemples d'utilisation
### Analyse quantitative (avec détection des commentaires non recevables)

```
import pandas as pd
from data_preprocessor import DataPreprocessor
from quantitative_analyser import QuantitativeAnalysisResponse, QuantitativeAnalyser

data_preprocessor = DataPreprocessor("Exemple 1/data.csv")
df: pd.DataFrame = data_preprocessor.preprocess()

quantitative_analyser: QuantitativeAnalyser = QuantitativeAnalyser()
quantitative_analysis_response: QuantitativeAnalysisResponse = quantitative_analyser(df["whole_text"].to_list())
```

### Analyse qualitative

```
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from data_preprocessor import DataPreprocessor
from reduce_chain import ReduceChain

data_preprocessor = DataPreprocessor("Exemple 1/data.csv")
df: pd.DataFrame = data_preprocessor.preprocess().head(10)

reduce_chain_pos = ReduceChain("favorables")
reduce_chain_neg = ReduceChain("défavorables")


def run_reduce_chain(chain):
    return chain.run(df)


with ThreadPoolExecutor() as executor:
    future_pos = executor.submit(run_reduce_chain, reduce_chain_pos)
    future_neg = executor.submit(run_reduce_chain, reduce_chain_neg)

result_pos = future_pos.result()
result_neg = future_neg.result()
```

### Déploiement de l'endpoint Sagemaker

```
from sagemaker_endpoint import SagemakerEndpoint
sagemaker_endpoint = SagemakerEndpoint()
predictor = sagemaker_endpoint.deploy(
    "unitary/multilingual-toxic-xlm-roberta",
    "text-classification",
    "AmazonSageMaker-ExecutionRole-20240705T144754"
)
```

## résultats de l'analyse quantitative

En testant notre classifieur sur les 100 premiers éléments de l'exemple des cétacés, on obtient les métriques suivantes:

précision: [0.93442623, 0.6]

rappel: [0.87692308, 0.75]

f-beta score: [0.9047619 , 0.66666667]

support: [65, 16] (on a rejeté les commentaires non retenus)

le script est disponible dans classifier_metrics.py