import os
import gdown


if __name__ == "__main__":
    file_id = "1atsqGopHmGYSbOm0MWfVtSt8MkwQzahv"
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, "query_data/query_data.zip", quiet=False)
    os.system("unzip query_data/query_data.zip -d query_data")
    os.remove("query_data/query_data.zip")
