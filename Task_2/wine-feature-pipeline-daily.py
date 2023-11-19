import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
   def f():
       g()


def generate_wine(wine, quality):
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd

    wine_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/wine.csv")
    wine_df = wine_df.drop('total sulfur dioxide', axis=1)
    wine_df.columns = [col.replace(' ', '_').lower() for col in wine_df.columns]
    column_order = list(wine_df.columns)
    df = wine_df[wine_df.type == wine]
    df = df[df.quality == quality].drop(['type', 'quality'], axis=1)
    sampled_data = df.sample(n=2)
    mean_data = pd.DataFrame(sampled_data.mean()).T
    mean_data['type'] = wine
    mean_data['quality'] = int(quality)
    generated_df = mean_data.reindex(columns=column_order)

    return generated_df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine
    """
    import random

    # randomly pick one of these 2 and write it to the feature store
    import random
    pick_random_wine = random.uniform(0,2)
    if pick_random_wine >= 1:
        wine = "red"
        random_quality = random.randint(3,8)
    else:
        wine = "white"
        random_quality = random.randint(3,9)
    
    wine_df = generate_wine(wine, random_quality)
    print("%s wine with a quality of %d added" %(wine,random_quality))

    return wine_df


def g():
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name="wine",version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("wine_daily")
        with stub.run():
            f()
