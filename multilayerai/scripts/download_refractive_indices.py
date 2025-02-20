import requests
import tqdm
import os


uris = [
    "https://refractiveindex.info/tmp/database/data-nk/main/Ge/Burnett.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Al/Rakic.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/C/Querry-DixonKS2.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Si/Shkondin.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/GaAs/Skauli.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Ag/Yang.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/SiO2/Kischkat.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/SiO/Hass.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Cr/Rakic-BB.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/BaF2/Li.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/AgCl/Tilton.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/NaI/Li.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Au/Olmon-ev.csv",
]


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(current_dir, "../../assets/refractive_indices")
    os.makedirs(target_dir, exist_ok=True)
    for uri in tqdm.tqdm(uris, desc="Downloading refractive index CSVs"):
        _, subdirectory, filename = uri.rsplit("/", 2)
        output_filename = f"{subdirectory}_{filename}"
        r = requests.get(uri, allow_redirects=True)
        with open(os.path.join(target_dir, f"{output_filename}"), "wb") as f:
            f.write(r.content)
