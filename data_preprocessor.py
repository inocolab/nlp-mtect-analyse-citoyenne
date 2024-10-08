from csv import DictReader

import pandas as pd
from io import StringIO
import re


class DataPreprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
    def extract_title_and_text(self):
        return [
            row["Titre"].strip().replace(".", "") + ". " + row["Texte"].lstrip()
            for row in self.dict_reader
        ]
    def preprocess(self) -> DictReader:
        with open(self.file_path, "r", encoding="latin1") as f:
            csv_lines = f.readlines()
            csv_lines = self._clean_special_characters_html(csv_lines)
            csv_lines = self._clean_misplaced_commas(csv_lines)
            csv_lines = self._remove_last_header(csv_lines)
            csv_lines = self._clean_double_quotes_inside_fields(csv_lines)

            self.dict_reader = DictReader(StringIO("\n".join(csv_lines)))

        return self.dict_reader

    def _preprocess_dataframe(self, df: pd.DataFrame):
        df["Titre"] = df["Titre"].fillna('')
        df["Texte"] = df["Texte"].fillna('')
        df["whole_text"] = df["Titre"] + ". " + df["Texte"]
        df['whole_text'] = df['whole_text'].astype("string")

    def _clean_special_characters_html(self, csv_lines):
        return [
            csv_line.replace("&#8217\";", "'").replace("&#8217;\"", "'").replace("&#8217;", "'").replace("&#8230;", "…")
            for csv_line in csv_lines]

    def _clean_misplaced_commas(self, csv_lines):
        csv_lines_to_return = [csv_lines[0].replace('"objet,"', '""objet"","')]
        csv_lines_to_return.extend(
            [csv_line.replace('"article,"', '""article"","').replace('"",publie"""', '""publie"""') for csv_line in
             csv_lines[1:]])
        return csv_lines_to_return

    def _remove_last_header(self, csv_lines):
        return [csv_line[0:csv_line.rfind(',')] for csv_line in csv_lines]

    def _clean_double_quotes_inside_fields(self, csv_lines):
        return [re.sub('(?<!")"(?!")', "", csv_line).replace('""', '"') for csv_line in csv_lines]
