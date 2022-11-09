import os
import gdown


if __name__ == "__main__":
    file_id = "1s_zg6g6Z19P8GM-vV-uLMe4wUY3YksHR"
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, "model_features/model_features.zip", quiet=False)
    os.system("unzip model_features/model_features.zip -d model_features")
    os.remove("model_features/model_features.zip")
