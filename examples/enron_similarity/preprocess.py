import re
from src.beam import resource
import pandas as pd


def download_data_from_kaggle(output_dir):
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Initialize the API
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files('wcukierski/enron-email-dataset', path=output_dir, unzip=True)


def get_data(output_dir):

    download_data_from_kaggle(output_dir)
    df = resource(output_dir).joinpath('emails.csv').read()

    # from pandarallel import pandarallel
    # pandarallel.initialize(progress_bar=True, nb_workers=16)

    pattern1 = r"Message-ID: (?P<message_id>[\w\W]*)\nDate: (?P<date>[\w\W]*)\nFrom: (?P<from>[\w\W]*)\nTo: (?P<to>[\w\W]*)\nSubject: (?P<subject>[\w\W]*)\nMime-Version: (?P<mime_version>[\w\W]*)\nContent-Type: (?P<content_type>[\w\W]*)" \
               r"\nContent-Transfer-Encoding: (?P<content_encoding>[\w\W]*)\nX-From: (?P<x_from>[\w\W]*)\nX-To: (?P<x_to>[\w\W]*)\nX-cc: (?P<x_cc>[\w\W]*)\nX-bcc: (?P<x_bcc>[\w\W]*)\nX-Folder: (?P<x_folder>[\w\W]*)\nX-Origin: (?P<x_origin>[\w\W]*)" \
               r"\nX-FileName: (?P<x_filename>.*)\n(?P<body>[\w\W]*)"

    def parse_re1(txt):
        try:
            m = re.match(pattern1, txt)
            if m is None:
                return None
            return m.groupdict()
        except Exception as e:
            return None

    # parsed = df['message'].parallel_apply(parse_re1)
    parsed = df['message'].apply(parse_re1)

    pattern2 = r"Message-ID: (?P<message_id>[\w\W]*)\nDate: (?P<date>[\w\W]*)\nFrom: (?P<from>[\w\W]*)\nSubject: (?P<subject>[\w\W]*)\nMime-Version: (?P<mime_version>[\w\W]*)\nContent-Type: (?P<content_type>[\w\W]*)" \
               r"\nContent-Transfer-Encoding: (?P<content_encoding>[\w\W]*)\nX-From: (?P<x_from>[\w\W]*)\nX-To: (?P<x_to>[\w\W]*)\nX-cc: (?P<x_cc>[\w\W]*)\nX-bcc: (?P<x_bcc>[\w\W]*)\nX-Folder: (?P<x_folder>[\w\W]*)\nX-Origin: (?P<x_origin>[\w\W]*)" \
               r"\nX-FileName: (?P<x_filename>.*)\n(?P<body>[\w\W]*)"

    def parse_re2(txt):

        m = re.match(pattern2, txt)
        if m is None:
            return None
        return m.groupdict()

    df_else = df.loc[parsed.isna()]['message'].apply(parse_re2)

    parsed[parsed.isna()] = df_else

    df_parsed = pd.DataFrame(parsed.tolist())

    df_parsed = df_parsed.drop_duplicates(subset=['date', 'from', 'to', 'subject', 'body'])

    resource(output_dir).joinpath('emails.parquet').write(df_parsed)


def main():
    output_dir = '/home/mlspeech/elads/data/enron/data'
    get_data(output_dir)


if __name__ == '__main__':
    main()