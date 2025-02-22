import requests
import tqdm
import os


uris = [
    "https://refractiveindex.info/tmp/database/data-nk/main/Ge/Burnett.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Al/Rakic.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Si/Shkondin.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Ag/Yang.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/SiO2/Kischkat.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Al2O3/Kischkat.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/AlN/Kischkat.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/HfO2/Franta.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Au/Olmon-ev.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/MgF2/Franta.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Si3N4/Kischkat.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Ta2O5/Franta.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/TiN/Beliaev-ALD.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/TiO2/Kischkat.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/ZnSe/Querry.csv",
    "https://refractiveindex.info/tmp/database/data-nk/main/Si3N4/Kischkat.csv",
    "https://refractiveindex.info/tmp/database/data/glass/optical/LZOS%20K108/nk/Bassarab.csv",
]


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(current_dir, "../../assets/refractive_indices")
    os.makedirs(target_dir, exist_ok=True)
    for uri in tqdm.tqdm(uris, desc="Downloading refractive index CSVs"):
        _, subdirectory, filename = uri.replace("/nk/", "/").rsplit("/", 2)
        output_filename = f"{subdirectory}_{filename}"
        r = requests.get(uri, allow_redirects=True)
        with open(os.path.join(target_dir, f"{output_filename}"), "wb") as f:
            f.write(r.content)
