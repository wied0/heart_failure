import os
import zipfile
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import PIL.Image as im
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


file_title = 'heart_failure_clinical_records_dataset.csv'

header = st.container()     # creates section width of page
row = st.container()
dataset = st.container()
data_exploration = st.container()
interactive = st.container()
modelling = st.container()
slot = st.container()
_image = im.open('Heart-diseases.jpg')


@st.cache
def get_data():
    if file_title not in os.listdir():
        Database(file_title)
    df_ = pd.read_csv(file_title)
    return df_


with header:
    st.title("Heart failure prediction")
    st.write("""In deze blogpost onderzoeken we de voorspelbaarheid van hartfalen en realties tussen veschillende
                waardes gemeten in het bloed van patienten. De dataset is afkomsting van Kaggle en word geimporteerd door middels van 
                de Kaggle API..""")
    with row:
        st.image(_image)

with dataset:
    st.header('Importeren van de data')
    st.write('''Voor het importeren van de data is gebruik gemaakt vaan de kaggle API. Hiervoor moeten eerst een paar 
                voorbereidingen voor gedaan worden. Allereerst mot je via je Kaggle account een API-key opvragen. Daarna
                moet je het kaggle.json file in een directory bij je hom directory opslaan. Dit gaat met de volgende code 
                voor de mac:
                ''')
    st.text('cd ~')
    st.text('mkdir .kaggle')
    st.text('mv Downloads/kaggle.json ~/.kaggle')
    st.text('pip install kaggle')
    st.write("Hierna kan gebruik gemaakt worden van de kaggle API. Wanneer een call voor een dataset gemaakt word levert kaggle"
             "het bestaand in een zip file. Met de onderstaande code word de data geimporteerd als zip file en geopend als csv.")

    with st.echo():
        class Database:
            def __init__(self, api_command):
                self.dataset = None
                self.filenames = None

                os.system(api_command)
                self.name = api_command.split('/')
                self.Open()
                os.remove(self.name[1] + ".zip")

            def Open(self):
                self.dataset = []
                with zipfile.ZipFile(self.name[-1] + ".zip", "r") as zip:
                    filenames = zip.namelist()
                    zip.extractall()
                for i in filenames:
                    self.dataset.append(pd.read_csv(i))
                self.filenames = filenames
                return self


        # Database("kaggle datasets download -d andrewmvd/heart-failure-clinical-data")
        df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

with data_exploration:
    st.header("Data verkennen")
    st.write("""Allereerst crontroleren we de data op missende waardes.""")
    with st.echo():
        print(df.isnull().sum())  # Geen nan values
        print(df.info())
        print(df.describe())

    # create figure
    fig = go.Figure()

    for j in df.columns:
        fig.add_trace(go.Box(y=df[j], name=j))

    dropdown_buttons = [
        {
            'label': 'All-data', 'method': 'update',
            'args': [{'visible': [True, True, True, True, True, True, True, True, True, True, True, True, True]},
                     ]},
        {
            'label': 'age', 'method': 'update',
            'args': [
                {'visible': [True, False, False, False, False, False, False, False, False, False, False, False, False]},
                ]},

        {
            'label': 'creatinine_phosphokinase', 'method': 'update',
            'args': [
                {'visible': [False, False, True, False, False, False, False, False, False, False, False, False, False]},
                ]},

        {
            'label': 'ejection_fraction', 'method': 'update',
            'args': [
                {'visible': [False, False, False, False, True, False, False, False, False, False, False, False, False]},
                ]},

        {
            'label': 'platelets', 'method': 'update',
            'args': [
                {'visible': [False, False, False, False, False, False, True, False, False, False, False, False, False]},
                ]},
        {
            'label': 'serum_creatinine', 'method': 'update',
            'args': [
                {'visible': [False, False, False, False, False, False, False, True, False, False, False, False, False]},
                ]},
        {
            'label': 'serum_sodium', 'method': 'update',
            'args': [
                {'visible': [False, False, False, False, False, False, False, False, True, False, False, False, False]},
                ]},

        {
            'label': 'time', 'method': 'update',
            'args': [
                {'visible': [False, False, False, False, False, False, False, False, False, False, False, True, False]},
                ]}

    ]
    fig.update_layout({
        'updatemenus': [{
            'type': "dropdown",
            'showactive': True,
            'active': 0,
            'buttons': dropdown_buttons}]
    }, title='Boxplots van de dataset')

    st.write(fig)
    st.write("""Om te kijken of er uitschieters zijn hebben we van alle variabelen een boxplot gemaakt. Maar dit gehele overzicht heeft een paar problemen! 
    De platelets data maakt gebruik van veel hogere getallen, en sommige variabelen zijn binaire van waarde, oftewel de waardes zijn alleen een 0 of een 1. 
    Daar kunnen we geen boxplots van maken. Daarom hebben we voor elke variabele die niet binair is apart een boxplot gemaakt die je kan selecteren met het 
    drop-dowm menu aan de zijkant van deze grafiek. Zo kunnen we voor elk van die variabelen goed kijken of er uitschieters zijn.""")
    df_c_melt = df.corr()[12:13].melt()[:12]
    fig = px.bar(df_c_melt, x='variable', y='value', color='value')
    fig.update_yaxes(title='Correlatie')
    st.write(fig)
    st.write("""Hier hebben we de correlatie tussen verscheidene variabelen en DEATH_EVENTS weergegeven. Om enigzins goed te kunnen voorspellen willen we gebruik maken van variabelen met een sterke correlatie. Dit kan een positieve of negatieve correlatie zijn.
                Uit deze grafiek kunnen we halen dat 'age', 'ejection_fraction', 'serum_creatine', 'serum_sodium' en 'time' het sterkste correleren, en waar we dus het beste mee kunnen voorspellen.""")

    fig = px.scatter(df,
                     x="serum_creatinine",
                     y="ejection_fraction",
                     animation_frame="sex",
                     animation_group="age",
                     size='time', color="DEATH_EVENT", hover_name="age",
                     log_x=False, size_max=25, range_x=[0, 10], range_y=[10, 100])

    fig["layout"].pop("updatemenus")
    st.write(fig)
    st.write("De slider vertelt iets over de death event van de man en de vrouw. De waardes uit de bovenstaande "
                     "correlatie grafiek. De slider is interactief, het is dus mogelijk om te filteren op man of vrouw. "
                     "Hier wordt weergegeven wat de death event is per ziekte per sex.")
with modelling:
    st.header('Modelleren')
    st.write(
        "Om een voorspellend model te maken voor de data maken we gebruik van een kNN algorithme. Het kNN algorithme "
        "neemt de 'k' dichtsbijzijnde waarnemingen en voorspelt de klasse afhankelijk van de waargenomen klassen van de "
        "'k' dichtst bijzijnde buren. In dit verslag proberen we de optimale hyperparameter 'k' te vinden, maar allereesrt "
        "preparen we de dataset met wat transformaties via de 'standerscalar' methode en splitsen de dataset in test en train.")
    with st.echo():
        scaler = StandardScaler()
        scaler.fit(df.drop('DEATH_EVENT', axis=1))
        scaled_features = scaler.transform(df.drop(['DEATH_EVENT'], axis=1))

        # Spitten in X_train, X_test, y_train, y_test
        X = pd.DataFrame(scaled_features, columns=df.columns[:-1])
        y = df['DEATH_EVENT']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    st.write("Nadat de datasetes zijn voorbereid en gepslitst kunnen we een model trainen. Voor het schatten van de optimale"
             "waarde voor paramer k geruiken we de volgende code.")

    with st.echo():
        error_rate = np.array([])
        for i in range(1, 40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rate = np.append(error_rate, np.mean(pred_i != y_test))

        error_rate_df = pd.DataFrame(error_rate)

    error_rate_df.columns = ['error']
    fig = px.line(error_rate_df, x=error_rate_df.index, y='error', markers=True,
                  title='Error van voorspellingen',
                  labels={
                      'index': "k"
                  })
    st.write(fig)
    knn = KNeighborsClassifier(n_neighbors=np.argmin(error_rate))
    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    st.write(confusion_matrix(y_test, pred))

    st.write(classification_report(y_test, pred))


with slot:
    st.header("Slot")
    st.write(
        "Dit was onze blog wat betreft hartfalen. Tijdens deze casus hebben we geleerd hoe we interactieve visualisaties moeten maken. "
        "Vervolgens hebben we via Streamlit onze data online kunnen zetten zodat het zichtbaat is voor iedereen. Wij hopen jullie hiermee "
        "genoeg geïnformeerd te hebben. ~Groep 18")

def niets():
    """
        leeftijd = st.sidebar.number_input("Vul leeftijd in: ",  min_value=10, max_value=100)
    anaemia = st.sidebar.selectbox("Een toename in rode bloedcellen?", "Kies?", ('Ja', 'Nee'))

    Dit is wat on afgemaakte code voor een sidebar input.
    creatinine_phosphokinase = st.sidebar.number_input("Vul leeftijd in: ", "")
    diabetes = st.sidebar.number_input("Vul leeftijd in: ", "")
    ejection_fraction = st.sidebar.number_input("Vul leeftijd in: ", "")
    high_blood_pressure = st.sidebar.number_input("Vul leeftijd in: ", "")
    platelets = st.sidebar.number_input("Vul leeftijd in: ", "")
    serum_creatinine = st.sidebar.number_input("Vul leeftijd in: ", "")
    serum_sodium = st.sidebar.number_input("Vul leeftijd in: ", "")
    smoking = st.sidebar.number_input("Vul leeftijd in: ", "")
    """