# Instalacija vritualnog okruženja za razvoj programa

U datotekama *"environment.yml"* i *"setup.py"* nalaze se konfiguracije i paketi koji će se instalirati za potrebe kolegija. 

Jedan od načina kreiranja virtualnog okruženja je pomoću **"conda"** sistemskog menađžera paketa. Isti se može instalirati prateći upute na sljedećoj [poveznici.](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html). Jednom instalirana, u terminalu mogu se pokrenuti iduće naredbe kako bi se stvorilo virtualno okruženje:

```
conda env create -f environment.yml
conda activate SDSR-SU
```

Izlazak iz virtualnog okruženja moguć je putem naredbe:

```
conda deactivate SDSR-SU
```

Dok je ažuriranje sa novim paketima moguće putem idućih naredbi nakon što se pozicioniramo u direktorij *Environment*:

```
git pull
pip install -e .
```
---

