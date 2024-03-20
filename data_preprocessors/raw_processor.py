import pandas as pd
import glob
from configs.raw_data_sources import concept_vocab, hospital_folders


class RawDataProcessor:

    def __init__(self, vocab_file=concept_vocab, hospital_folders=hospital_folders):

        concept_df = pd.read_csv(vocab_file, low_memory=False)
        self.set_dtype = concept_df['concept_id'].dtype
        self.target_dict = dict(
            zip(concept_df['concept_id'], concept_df['concept_name']))

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
            "source_code_description", "target_concept_id"]]

        source_2_target = self.hospitals_df.copy()
        source_2_target.loc[:, 'concept_name'] = source_2_target['target_concept_id'].astype(
            self.set_dtype).map(self.target_dict)
        source_2_target.dropna(inplace=True, ignore_index=True)
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
