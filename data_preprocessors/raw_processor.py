import pandas as pd
import glob
from configs.raw_data_sources import concept_vocab, hospital_folders


class RawDataProcessor:

    def __init__(self):

        self.concept_df = pd.read_csv(concept_vocab)[
            ["concept_id", "concept_name"]]

        self.hospitals_df = pd.DataFrame()

        for hospital_folder in hospital_folders:
            hospital_csvs = glob.glob(hospital_folder + "*")
            hospital_df = pd.DataFrame()

            for csv in hospital_csvs:
                hospital_df = pd.concat(
                    [hospital_df,
                     pd.read_csv(csv)
                     ]
                )
            self.hospitals_df = pd.concat(
                [self.hospitals_df,
                 hospital_df
                 ]
            )

    def join_source_target(self):

        self.hospitals_df = self.hospitals_df[[
            "source_vocabulary_id", "source_code_description", "target_concept_id"]]

        source_2_target = self.hospitals_df.merge(
            self.concept_df, how='outer', left_on="target_concept_id", right_on="concept_id")

        source_2_target.dropna(inplace=True)

        source_2_target.loc[source_2_target['concept_name']
                            != 'No matching concept', :]

        sources = source_2_target["source_code_description"].tolist()
        targets = source_2_target["concept_name"].tolist()

        return sources, targets

    @staticmethod
    def _prepare_4_encoding(sources, targets):
        sources = [("query: " + i) for i in sources]
        targets = [("query: " + i) for i in targets]

        return sources, targets
