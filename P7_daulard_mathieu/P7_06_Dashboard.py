import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors




def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data, ranges):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)

    x1, x2 = ranges[0]
    d = data[0]

    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1

    sdata = [d]

    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1

        sdata.append((d - y1) / (y2 - y1) * (x2 - x1) + x1)

    return sdata


class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                    n_ordinate_levels=6):
        angles = np.arange(0, 360, (360. / len(variables)))

        axes = [fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True,
                                label="axes{}".format(i))
                for i in range(len(variables))]

        axes[0].set_thetagrids(angles, labels=[])

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                                num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x, 2))
                            for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1]  # hack to invert grid
                # gridlabels aren't reversed
            gridlabel[0] = ""  # clean up origin
            ax.set_rgrids(grid, labels=gridlabel, angle=angles[i])
            # ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])

        ticks = angles
        ax.set_xticks(np.deg2rad(ticks))  # crée les axes suivant les angles, en radians
        ticklabels = variables
        ax.set_xticklabels(ticklabels, fontsize=10)  # définit les labels

        angles1 = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
        angles1[np.cos(angles1) < 0] = angles1[np.cos(angles1) < 0] + np.pi
        angles1 = np.rad2deg(angles1)
        labels = []
        for label, angle in zip(ax.get_xticklabels(), angles1):
            x, y = label.get_position()
            lab = ax.text(x, y - .5, label.get_text(), transform=label.get_transform(),
                            ha=label.get_ha(), va=label.get_va())
            lab.set_rotation(angle)
            lab.set_fontsize(16)
            lab.set_fontweight('bold')
            labels.append(lab)
        ax.set_xticklabels([])

        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)








# -- Set page config
apptitle = 'Solvabilité financière'

st.set_page_config(page_title=apptitle, layout='wide', page_icon=":eyeglasses:")


st.title('Vérification solvabilité financière client')

with st.expander("Description de l'application"):
     st.write("""
         Dashbord permettant de vérifier la solvabilité financière d'un client pour l'octroie d'un prêt. 
         Nous pouvons comparer ce client à des clients similaires solvables ou non solvables, ainsi
         que vérifier l'impact des caractéristiques des clients sur leur solvabilité.
     """)

def open_test_data(url):
    return open(url, 'rb')

def load_model(df_scaled):
    infile = open('LightGBMModel.joblib', 'rb') 
    lgbm = joblib.load(infile) 
    infile.close()
    result = pd.Series(lgbm.predict_proba(df_scaled)[:, 1])
    result = result.apply(lambda x: np.where(x > 0.44, "Non Solvable", "Solvable"))
    return result

def nearest(df_scaled, df, Customer_id, solvable = True):
    if solvable:
        index_near = df[df['Solvabilite'] == 'Solvable'].index.tolist()
        index_near.append(Customer_id)
        df_scaled = df_scaled.loc[index_near,:]
    else:
        index_near = df[df['Solvabilite'] == 'Non Solvable'].index.tolist()
        index_near.append(Customer_id)
        df_scaled = df_scaled.loc[index_near,:]
    df_scaled = df_scaled.drop_duplicates()
    NN = NearestNeighbors(n_neighbors = 4)
    NN = NN.fit(df_scaled.fillna(df_scaled.mean()))
    result = []
    for i in NN.kneighbors(df_scaled.fillna(df_scaled.mean()).loc[Customer_id,:].to_numpy().reshape([1,-1]))[1] :
        i = i[1:]
        for j in i :
            result.append(df_scaled.index[j])
        result = list(map(int, result))
    return result

@st.cache
def df_data():
    with open_test_data('DATA.csv') as f:
        df = pd.read_csv(f)
        #df = df.drop('Unnamed: 0', axis = 1)
        df.set_index('SK_ID_CURR',inplace = True)
        df[['Revenus', 'Montant_credit', 'Cout_annuel_credit', 'Valeur_bien_finance',
       'Annuite/revenus']] = df[['Revenus', 'Montant_credit', 'Cout_annuel_credit', 'Valeur_bien_finance',
       'Annuite/revenus']].round(2)
        df['Solvabilite'] = load_model(df_data_scaled()).values
    return df

@st.cache
def df_data_scaled():
    with open_test_data('DATA_SCALED.csv') as f:
        df_scaled = pd.read_csv(f)
        #df_scaled = df_scaled.drop('Unnamed: 0', axis = 1) 
        df_scaled.set_index('SK_ID_CURR',inplace = True)
    return df_scaled


Customer_id = st.selectbox("Entrez l'identifiant du client que vous souhaitez analyser", df_data().index)
Customer_id = int(Customer_id)



if df_data().loc[Customer_id, 'Solvabilite'] == "Solvable" :
    st.success("Le client est SOLVABLE")
else:
    st.error("Le client est NON-SOLVABLE")

def adaptation_client(Customer_id):
    df = df_data()
    near_solvable = nearest(df_data_scaled(), df, Customer_id, True)
    near_nsolvable = nearest(df_data_scaled(), df, Customer_id, False)
    df_adapt = df.loc[near_solvable,:]
    df_adapt['Nearest_status']  = "Voisins solvables"
    df_adapt = pd.concat([df_adapt, df.loc[near_nsolvable,:]])
    df_adapt['Nearest_status']  = df_adapt['Nearest_status'].fillna("Voisins non solvables")
    df_adapt = pd.concat([df_adapt, pd.DataFrame(df.loc[Customer_id,:].to_numpy().reshape([1,-1]), index=[Customer_id], columns=df_adapt.columns[:-1])])
    df_adapt['Nearest_status']  = df_adapt['Nearest_status'].fillna("Cible")
    df_adapt = pd.concat([df_adapt, df.drop(df_adapt.index)])
    df_adapt['Nearest_status']  = df_adapt['Nearest_status'].fillna("Tous")
    return df_adapt
st.header("--- Informations client cible ---")


col1, col2, col3 = st.columns(3)
col1.metric("Age :",  int(adaptation_client(Customer_id).loc[Customer_id, "Age"]))
col2.metric("Sex :", adaptation_client(Customer_id).loc[Customer_id, "Genre"])
col3.metric("Nombre d'enfant :", adaptation_client(Customer_id).loc[Customer_id, "Enfants"])
col1b, col2b, col3b = st.columns(3)
col1b.metric("Statut marital :",  adaptation_client(Customer_id).loc[Customer_id, "Statut_Marital"])
col2b.metric("Anciennete emploi actuel :", adaptation_client(Customer_id).loc[Customer_id, "Anciennete_emploi_actuel"])
col3b.metric("Niveau éducation :", adaptation_client(Customer_id).loc[Customer_id, "Education"])
col1c, col2c, col3c = st.columns(3)
col1c.metric("Type de credit :",  adaptation_client(Customer_id).loc[Customer_id, "Type_credit"])
col2c.metric("Revenus du foyer :", int(adaptation_client(Customer_id).loc[Customer_id, "Revenus"]))
col3c.metric("Part coût prêt sur revenus :", '{0:1.2f}%'.format(adaptation_client(Customer_id).loc[Customer_id, "Annuite/revenus"]*100))


l_radio = adaptation_client(Customer_id)['Nearest_status'].unique().tolist()
l_radio.remove('Cible')
radio_choice = st.radio('A quel groupe comparer le client cible ?', l_radio)
del l_radio

st.dataframe(adaptation_client(Customer_id)[adaptation_client(Customer_id)['Nearest_status']==radio_choice])

fig_col1, fig_col2 = st.columns(2)

mask_categ = ['Type_credit',
 'Genre',
 'Voiture',
 'Proprietaire',
 'Enfants',
 'Taille_foyer',
 'Statut_Marital',
 'Education']

Categ = fig_col1.selectbox('Choisissez la variable pour comparer la population sur le critère de solvabilité', df_data()[mask_categ].columns)

del mask_categ

data_fig = adaptation_client(Customer_id).groupby(['Solvabilite', Categ], as_index=False).agg({'Cout_annuel_credit':'count'})
for i in data_fig['Solvabilite'].unique():
    data_fig.loc[data_fig['Solvabilite'] == i, 'Pourcentage'] = data_fig.loc[data_fig['Solvabilite'] == i, 'Cout_annuel_credit'].apply(lambda x: '{0:1.2f}%'.format(x/data_fig.loc[data_fig['Solvabilite'] == i, 'Cout_annuel_credit'].sum()*100))

fig_col1, fig_col2 = st.columns(2)

fig = px.bar(data_fig, x = 'Solvabilite', y = 'Cout_annuel_credit', color = Categ, text=data_fig['Pourcentage'], labels={
                     "Cout_annuel_credit": "Nombre individus"})

fig_col1.plotly_chart(fig, use_container_width=True)


# data
categories = ['Revenus',
 'Montant_credit',
 'Cout_annuel_credit',
 'Valeur_bien_finance', 'Nearest_status']


data_fig = adaptation_client(Customer_id)[categories].groupby('Nearest_status').mean()

categories = ['Revenus',
 'Montant_credit',
 'Cout_annuel_credit',
 'Valeur_bien_finance']




data_ex = data_fig.iloc[0]


ranges = []
for i in data_fig.columns:
    ranges.append((int(data_fig[i].min()*0.80), int(data_fig[i].max()*1.20)))


# plotting
fig2 = plt.figure(figsize=(7, 7))
radar = ComplexRadar(fig2, categories, ranges)
radar.plot(data_fig.iloc[0,:], label='Notre client')
#radar.fill(data_ex, alpha=0.2)

radar.plot(data_fig.iloc[3,:],
            label='Moyenne des clients similaires sans défaut de paiement',
            color='g')
radar.plot(data_fig.iloc[2,:],
            label='Moyenne des clients similaires avec défaut de paiement',
            color='r')

fig2.legend(bbox_to_anchor=(1.7, 1), fontsize = 'large')




fig_col2.pyplot(fig2, use_container_width=True)
