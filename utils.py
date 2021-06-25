import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def oneHotEncodeData(data_df):
    # Make sure names are similar
    data_df['t1_playerid'] = data_df['t1_playerid'].str.lower().str.strip().str.replace(" ","_")
    data_df['t2_playerid'] = data_df['t2_playerid'].str.lower().str.strip().replace(" ","_")
    data_df['t1p1_player'] = data_df['t1p1_player'].str.lower().str.strip().replace(" ","_")
    data_df['t1p2_player'] = data_df['t1p2_player'].str.lower().str.strip().replace(" ","_")
    data_df['t1p3_player'] = data_df['t1p3_player'].str.lower().str.strip().replace(" ","_")
    data_df['t1p4_player'] = data_df['t1p4_player'].str.lower().str.strip().replace(" ","_")
    data_df['t1p5_player'] = data_df['t1p5_player'].str.lower().str.strip().replace(" ","_")
    data_df['t2p1_player'] = data_df['t2p1_player'].str.lower().str.strip().replace(" ","_")
    data_df['t2p2_player'] = data_df['t2p2_player'].str.lower().str.strip().replace(" ","_")
    data_df['t2p3_player'] = data_df['t2p3_player'].str.lower().str.strip().replace(" ","_")
    data_df['t2p4_player'] = data_df['t2p4_player'].str.lower().str.strip().replace(" ","_")
    data_df['t2p5_player'] = data_df['t2p5_player'].str.lower().str.strip().replace(" ","_")

    data_df['t1p1_champion'] = data_df['t1p1_champion'].str.lower().str.strip().replace(" ","_")
    data_df['t1p2_champion'] = data_df['t1p2_champion'].str.lower().str.strip().replace(" ","_")
    data_df['t1p3_champion'] = data_df['t1p3_champion'].str.lower().str.strip().replace(" ","_")
    data_df['t1p4_champion'] = data_df['t1p4_champion'].str.lower().str.strip().replace(" ","_")
    data_df['t1p5_champion'] = data_df['t1p5_champion'].str.lower().str.strip().replace(" ","_")
    data_df['t2p1_champion'] = data_df['t2p1_champion'].str.lower().str.strip().replace(" ","_")
    data_df['t2p2_champion'] = data_df['t2p2_champion'].str.lower().str.strip().replace(" ","_")
    data_df['t2p3_champion'] = data_df['t2p3_champion'].str.lower().str.strip().replace(" ","_")
    data_df['t2p4_champion'] = data_df['t2p4_champion'].str.lower().str.strip().replace(" ","_")
    data_df['t2p5_champion'] = data_df['t2p5_champion'].str.lower().str.strip().replace(" ","_")

    data_df['t1_ban1'] = data_df['t1_ban1'].str.lower().str.strip().replace(" ","_")    
    data_df['t1_ban2'] = data_df['t1_ban2'].str.lower().str.strip().replace(" ","_")
    data_df['t1_ban3'] = data_df['t1_ban3'].str.lower().str.strip().replace(" ","_")
    data_df['t1_ban4'] = data_df['t1_ban4'].str.lower().str.strip().replace(" ","_")
    data_df['t1_ban5'] = data_df['t1_ban5'].str.lower().str.strip().replace(" ","_")
    data_df['t2_ban1'] = data_df['t2_ban1'].str.lower().str.strip().replace(" ","_")
    data_df['t2_ban2'] = data_df['t2_ban2'].str.lower().str.strip().replace(" ","_")
    data_df['t2_ban3'] = data_df['t2_ban3'].str.lower().str.strip().replace(" ","_")
    data_df['t2_ban4'] = data_df['t2_ban4'].str.lower().str.strip().replace(" ","_")
    data_df['t2_ban5'] = data_df['t2_ban5'].str.lower().str.strip().replace(" ","_")

    categorical_columns = ['t1_playerid','t2_playerid','t1p1_player','t1p2_player','t1p3_player','t1p4_player',
    't1p5_player','t2p1_player','t2p2_player','t2p3_player','t2p4_player','t2p5_player',
    't1p1_champion','t1p2_champion','t1p3_champion','t1p4_champion',
    't1p5_champion','t2p1_champion','t2p2_champion','t2p3_champion','t2p4_champion','t2p5_champion',
    't1_ban1','t1_ban2','t1_ban3','t1_ban4','t1_ban5','t2_ban1','t2_ban2','t2_ban3','t2_ban4','t2_ban5',]
    dum_df = pd.get_dummies(data_df, columns=categorical_columns, prefix=categorical_columns)
    return dum_df
    
def piecharts(data_df):
    bans = pd.Series(data_df['t1_ban1'])
    bans.append(data_df['t1_ban2'])
    bans.append(data_df['t1_ban3'])

    unique_bans = bans.unique()
    ban_count = []

    for i in unique_bans:
        count = 0
        for a in data_df['t1_ban1']:
            if(a == i):
                count += 1
    
        for b in data_df['t1_ban2']:
            if(b == i):
                count += 1
    
        for c in data_df['t1_ban3']:
            if(c == i):
                count += 1

        ban_count.append(count)

    ban_count_series = pd.Series(ban_count)
    ban_count_series.index = unique_bans

    plt.figure(figsize=(12,7))
    ban_count_series.sort_values(ascending=False)[:10].plot(kind='pie', autopct='%1.1f%%')
    plt.title('Top 10 Banned Champions')
    plt.ylabel('Champions')
    plt.show()

    picks = pd.Series(data_df['t1p1_champion'])
    picks.append(data_df['t1p2_champion'])
    picks.append(data_df['t1p3_champion'])
    picks.append(data_df['t1p4_champion'])
    picks.append(data_df['t1p5_champion'])

    unique_picks = picks.unique()
    pick_count = []

    for i in unique_picks:
        count = 0
        for a in data_df['t1_ban1']:
            if(a == i):
                count += 1
    
        for b in data_df['t1_ban2']:
            if(b == i):
                count += 1
    
        for c in data_df['t1_ban3']:
            if(c == i):
                count += 1
        pick_count.append(count)

    pick_count_series = pd.Series(pick_count)
    pick_count_series.index = unique_picks

    plt.figure(figsize=(12,7))
    pick_count_series.sort_values(ascending=False)[:10].plot(kind='pie', autopct='%1.1f%%')
    plt.title('Top 10 Picked Champions')
    plt.ylabel('Champions')
    plt.show()
    
def bargraphs(data_df):
    total_dragons = data_df.groupby(["t1_playerid"]).t1_dragons.sum() + data_df.groupby(["t2_playerid"]).t2_dragons.sum()
    total_dragons.sort_values(ascending=False)[:10].plot(kind='barh')
    plt.title('Teams Top 10 Dragon Count')
    plt.ylabel('Teams')
    plt.show()

    total_heralds = data_df.groupby(["t1_playerid"]).t1_heralds.sum() + data_df.groupby(["t2_playerid"]).t2_heralds.sum()
    total_heralds.sort_values(ascending=False)[:10].plot(kind='barh')
    plt.title('Teams Top 10 Heralds Count')
    plt.ylabel('Teams')
    plt.show()

    total_barons = data_df.groupby(["t1_playerid"]).t1_barons.sum() + data_df.groupby(["t2_playerid"]).t2_barons.sum()
    total_barons.sort_values(ascending=False)[:10].plot(kind='barh')
    plt.title('Teams Top 10 Barons Count')
    plt.ylabel('Teams')
    plt.show()


def bargraphs2(data_df):
    wins = data_df[data_df['t2_result'] == 1]['t2_playerid'].value_counts() + data_df[data_df['t1_result'] == 1]['t1_playerid'].value_counts()
    wins.sort_values(ascending=False)[:10].plot(kind='barh')
    plt.title("Number of games won")
    plt.show()


def bargraphs3(data_df):
    wins = data_df[data_df['t2_result'] == 1]['t2_playerid'].value_counts() + data_df[data_df['t1_result'] == 1]['t1_playerid'].value_counts()
    losses = data_df[data_df['t2_result'] == 0]['t2_playerid'].value_counts() + data_df[data_df['t1_result'] == 0]['t1_playerid'].value_counts()

    ratio = wins / (losses + wins)
    plt.title("Win/loss ratio")
    ratio.sort_values(ascending=False)[:15].plot(kind='barh')

    
def rolling_average(data_df, t1_count_name, t1_objective, t1_avg_objective, t2_count_name, t2_objective, t2_avg_objective):
    cummsum(t1_count_name, 't1_playerid', t1_objective, data_df)
    cummsum(t2_count_name, 't2_playerid', t2_objective, data_df)

    data_df['t1_gamecount'] = data_df.groupby('t1_playerid').cumcount()
    data_df[t1_avg_objective] = data_df[t1_count_name]/data_df['t1_gamecount']
    data_df[t1_avg_objective] = data_df[t1_avg_objective].fillna(0)

    data_df['t2_gamecount'] = data_df.groupby('t2_playerid').cumcount()
    data_df[t2_avg_objective] = data_df[t2_count_name]/data_df['t2_gamecount']
    data_df[t2_avg_objective] = data_df[t2_avg_objective].fillna(0)

    data_df[t1_avg_objective]= data_df[t1_avg_objective].round(2)
    data_df[t2_avg_objective]= data_df[t2_avg_objective].round(2)
    return data_df


def cummsum(sum_feature, player, player_stats, data):
    data[sum_feature] = data.groupby(player)[player_stats].cumsum(axis=0)
    data[sum_feature] = data.groupby(player)[sum_feature].shift(1) #lag by 1 so theres only info from previous matches
    data[sum_feature].fillna(0,inplace=True) 
    return data

def rep(new_col, og_col, data):
    data[new_col] = data[og_col].replace([0],1)
    return data

def kda (player_kda, player_kills, player_assists, player_deaths, data):
    data[player_kda] = (data[player_kills] + data[player_assists])/data[player_deaths]
    data[player_kda] = data[player_kda].round(2)
    return data
    
def buildLrModel(X_train, Y_train, feature_names):
    logistic = LogisticRegression()

    log_model = GridSearchCV(logistic, {
        'C': [1,10,100],
        'max_iter': [25,50,100],
        'solver' : ['liblinear','saga'],
        'tol' : [0.1,0.2,0.3]
    })

    log_model.fit(X_train, Y_train)
    
    print(log_model.best_estimator_)
    return log_model

def buildNeuralModel(X_train,Y_train,feature_names):
    feature_count = len(feature_names)
    neural_model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[feature_count]), 
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    ])

    neural_model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    EPOCHS = 50

    neural_model.fit(
        X_train,
        Y_train,
        batch_size=32,
    epochs=EPOCHS,
    )
    return neural_model

def buildRandomForestModel(X_train,Y_train,feature_names):
    random_forest= RandomForestClassifier()

    random_forest_model = GridSearchCV(random_forest, {
        'n_estimators': [10,100,200],
        'max_depth': [1,2,5,10],
    })

    random_forest_model.fit(X_train, Y_train)
    
    return random_forest_model
    

def addWinRate(data_df,dum_df):
    winMap = {}
    for item in dum_df.columns:
        if 't1_playerid' in item:
            winMap[item] = {'wins':[],'totalGames':[]}
        if 't2_playerid' in item:
            winMap[item] = {'wins':[],'totalGames':[]}

    data_df['t1_games_won_so_far'] = 0
    data_df['t1__games_played_so_far'] = 0
    data_df['t2_games_won_so_far'] = 0
    data_df['t2__games_played_so_far'] = 0
    for team, values in winMap.items():
        team_df = data_df[data_df[team] == 1]
        idx = 0
        for index, row in team_df.iterrows():
            result = 0
            if 't1_playerid' in team:
                result = row['t1_result']
            else:
                result = row['t2_result']
            laggedIdx = idx
            if idx == 0:
                values['wins'].append(result)
                values['totalGames'].append(1)
                if 't1_playerid' in team:
                    data_df.loc[index,'t1_games_won_so_far'] = 0
                    data_df.loc[index,'t1_games_played_so_far'] = 0
                else:
                    data_df.loc[index,'t2_games_won_so_far'] = 0
                    data_df.loc[index,'t2_games_played_so_far'] = 0
            else:
                values['wins'].append(values['wins'][idx - 1] + result)
                values['totalGames'].append(values['totalGames'][idx - 1] + 1)
                if 't1_playerid' in team:
                    data_df.loc[index,'t1_games_won_so_far'] = values['wins'][idx - 1]
                    data_df.loc[index,'t1_games_played_so_far'] = values['totalGames'][idx - 1]
                else:
                    data_df.loc[index,'t2_games_won_so_far'] = values['wins'][idx - 1]
                    data_df.loc[index,'t2_games_played_so_far'] = values['totalGames'][idx - 1]
        
            idx = idx + 1

    data_df['t1_winrate'] = data_df['t1_games_won_so_far'] / data_df['t1_games_played_so_far']
    data_df['t2_winrate'] = data_df['t2_games_won_so_far'] / data_df['t2_games_played_so_far']

    data_df['t1_winrate'] = data_df['t1_winrate'].fillna(0)
    data_df['t2_winrate'] = data_df['t2_winrate'].fillna(0)
    return data_df