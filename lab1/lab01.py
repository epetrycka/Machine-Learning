# %% [markdown]
# Pobranie danych i wizualizacja:

# %%
import requests
import tarfile
import gzip
import os
import pandas as pd

url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
file_name = "housing.tzg"
file_csv = "housing.csv"
file_gz = "housing.csv.gz"

response = requests.get(url)
if not os.path.exists(file_name):
    with open (file_name, "wb") as file:
        file.write(response.content)
    with tarfile.open(file_name, 'r:gz') as tar:
        tar.extractall(filter='data')
    with open (file_csv, 'rb') as file:
        b_compressed_data = gzip.compress(file.read())
    with open (file_gz, 'wb') as file:
        file.write(b_compressed_data)

if os.path.exists(file_name):
    os.remove(file_name)
    os.remove(file_csv)

df = pd.read_csv(file_gz)
print(df.head(2))

# %% [markdown]
# Testowanie metod Pandas:

# %%
df.info()

# %%
df.value_counts()

# %%
df.describe()

# %% [markdown]
# Wizualizacja:

# %%
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots(figsize=(20,15))
df.hist(bins=50, ax=ax1)
fig1.savefig(fname="obraz1.png")

# %%
fig2, ax2 = plt.subplots(figsize=(7,4))
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, ax=ax2)
fig2.savefig(fname="obraz2.png")

# %%
fig3, ax3 = plt.subplots(figsize=(7,4))
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, ax=ax3, colorbar=True, s=df["population"]/100, label="population", c="median_house_value", cmap=plt.get_cmap('viridis'))
fig3.savefig(fname="obraz3.png")

# %% [markdown]
# Analiza, macierz korelacji:

# %%
df_encoded = pd.get_dummies(df, columns=["ocean_proximity"], dtype=float)
print(df_encoded.head(2))

# %%
corr_matrix = df_encoded.corr()["median_house_value"].sort_values(ascending=False)

korelacja = corr_matrix.reset_index().rename(columns={"index": "atrybut", "median_house_value": "wspolczynnik_korelacji"})
korelacja.to_csv("korelacja.csv", index=False)

# %%
import seaborn as sns

sns.pairplot(df_encoded)

# %% [markdown]
# Przygotowanie do uczenia:

# %%
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df_encoded, test_size=0.2, random_state=42)
print(len(train_data), len(test_data))
print("Train data : \n", train_data.head(10), "\n")
print("Test data : \n", test_data.head(10))

# %% [markdown]
# Macierze korelacji zbiorów:

# %%
print("Macierz korelacji całego zbioru: \n", korelacja)

corr_train = train_data.corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={"index":"atrybut","median_house_value":"wspolczynnik korelacji"})
print("Macierz korelacji zbioru uczącego: \n", corr_train)

corr_test = test_data.corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={"index":"atrybut","median_house_value":"wspolczynnik korelacji"})
print("Macierz korelacji zbioru testującego: \n", corr_test)

# %% [markdown]
# Wyniki są bardzo podobne. Oznacza to, że zbiory są do siebie zbliżone pod względem częstości występowania atrybutów (mają podobne rozkłady) i ich wzajemnych zależności.

# %%
import pickle

train_file_pickle = "train_set.pkl"
test_file_pickle = "test_set.pkl"

if not os.path.exists(train_file_pickle):
    with open(train_file_pickle, "wb") as file:
        pickle.dump(train_data, file)

if not os.path.exists(test_file_pickle):
    with open(test_file_pickle, "wb") as file:
        pickle.dump(test_data, file)


