import pandas as pd
from data_preprocessor import DataPreprocessor


def run_quantitative_analyser():
    from quantitative_analyser import QuantitativeAnalysisResponse, QuantitativeAnalyser

    data_preprocessor = DataPreprocessor("Exemple 1/data.csv")
    data_preprocessor.preprocess()
    data = data_preprocessor.extract_title_and_text()
    # print(data)
    # data = [
    #     "Clôtures - textes abscons. Les textes, arrêtés ou présentation, sont tout aussi abscons l'un que l'autre. On ne voit pas plus après lecture où est la vraie finalité. Le fait que le conseil de la chasse et de la faune sauvage soit le seul organisme officiellement consulté et qu'il ait donné un avis favorable à l'unanimité laisse le citoyen non chasseur, la majorité, plus que suspicieux. "
    # ]
    quantitative_analyser: QuantitativeAnalyser = QuantitativeAnalyser()
    quantitative_analysis_response: QuantitativeAnalysisResponse = quantitative_analyser(data)
    print(quantitative_analysis_response.negative_comments_total)
    print(quantitative_analysis_response.positive_comments_total)
    print(quantitative_analysis_response.rejected_comments_indexes)


def run_qualitative_analyser():
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd
    # from data_preprocessor import DataPreprocessor
    from reduce_chain import ReduceChain

    data_preprocessor = DataPreprocessor("Exemple 1/data_ia.csv")
    data_preprocessor.preprocess()
    data = data_preprocessor.extract_title_and_text()

    # reduce_chain_pos = ReduceChain("favorables")
    reduce_chain_neg = ReduceChain("défavorables", profile="mtcet-nlp", token_max=2048)

    def run_reduce_chain(chain):
        return chain.run(data)

    # result_pos = run_reduce_chain(reduce_chain_pos)
    result_neg = run_reduce_chain(reduce_chain_neg)
    print(type(result_neg),result_neg)
    # with ThreadPoolExecutor() as executor:
    #     future_pos = executor.submit(run_reduce_chain, reduce_chain_pos)
    #     future_neg = executor.submit(run_reduce_chain, reduce_chain_neg)

    # result_pos = future_pos.result()
    # result_neg = future_neg.result()

if __name__ == "__main__":
    run_qualitative_analyser()
    # run_quantitative_analyser()