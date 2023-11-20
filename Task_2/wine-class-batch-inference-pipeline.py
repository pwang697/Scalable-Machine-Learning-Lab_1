import os
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

#split one column as targets
def feature_target_split(df, column_name):
    features = df.drop(column_name, axis=1).reset_index(drop=True)
    targets = df[column_name].reset_index(drop=True)
    return features, targets

# One-hot encode the categorical feature
def one_hot_encoder(df, column_name):
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(drop='first',sparse=False)
    encoded_features = encoder.fit_transform(df[[column_name]])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([column_name]))
    encoded_df = pd.concat([df.drop(column_name, axis=1), encoded_df], axis=1)
    return encoded_df

def find_thresholds(train, test, column_name):
    import pandas as pd

    df = pd.concat([train, test], axis=0)
    column_list = list(df[column_name])
    column_list.sort()
    lower_threshold_idx = int(len(column_list)/3)
    upper_threshold_idx = lower_threshold_idx * 2
    lower_threshold = column_list[lower_threshold_idx]
    upper_threshold = column_list[upper_threshold_idx]
    return lower_threshold, upper_threshold

def quality2class(series, lower_threshold, upper_threshold):
    import pandas as pd

    quality_list = list(series)
    class_list = []
    for i in range(len(quality_list)):
        if quality_list[i] < lower_threshold:
            class_list.append('poor')
        elif quality_list[i] > upper_threshold:
            class_list.append('excellent')
        else:
            class_list.append('fair')
    class_series= pd.Series(class_list, name = 'class')

    return class_series

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests
    import numpy as np

    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    model = mr.get_model("wine_class_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_class_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()
    batch_data = batch_data.sample(n=100).reset_index(drop=True)
    batch_data, batch_label = feature_target_split(batch_data, 'quality')
    batch_data = one_hot_encoder(batch_data, 'type')
    
    y_pred = model.predict(batch_data)
    offset = 1
    wine = y_pred[y_pred.size-offset]
    print("wine predicted: " + wine)

    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read()
    lower_threshold, upper_threshold = find_thresholds(df[:100], df[100:], 'quality')
    label = batch_label.iloc[-offset]
    actual_class = quality2class([label], lower_threshold, upper_threshold)[0]
    print("wine class actual: " + actual_class)
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_class_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="wine quality Prediction/Outcome Monitoring"
                                                )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine],
        'label': [actual_class],
        'datetime': [now],
        }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})

    history_df = monitor_fg.read(read_options={"use_hive": True})
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our wine_predictions feature group has examples of all 3 classes
    print("Number of different wine class predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 3:
        results = confusion_matrix(labels, predictions)

        # Normalise confusion matrix
        results_norm = results.astype('float') / results.sum(axis=1)[:, np.newaxis]

        # Create the confusion matrix as a figure, we will later store it as a PNG image file
        df_cm = pd.DataFrame(results_norm, ['True poor', 'True fair', 'True excellent'],
                            ['Pred poor', 'Pred fair', 'Pred excellent'])

        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api = project.get_dataset_api()
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 3 different wine class predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 3 different wine class predictions")


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

