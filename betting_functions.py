#!/usr/bin/env python
# coding: utf-8

# In[44]:
import requests
import pandas as pd
import sqlite3
from datetime import datetime
from OddsJamClient import OddsJamClient

def load_api_key(path):
    with open(path, 'r') as file:
        return file.readline().strip()

# Convert American odds to decimal odds
def american_to_decimal(odds):
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1

# Calculate implied probability from decimal odds
def implied_probability(decimal_odds):
    return 1 / decimal_odds

# Adjust implied probability to account for the vig
def adjust_for_vig(implied_prob, vig_reduction=0.02):
    return implied_prob / (1 + vig_reduction)

# Calculate expected value (EV)
def calculate_ev(true_prob, odds, stake=100):
    decimal_odds = american_to_decimal(odds)
    profit_if_win = (decimal_odds - 1) * stake
    loss_prob = 1 - true_prob
    ev = (true_prob * profit_if_win) - (loss_prob * stake)
    return ev

# Fetch sports markets dynamically for a specific sport and league
def fetch_sports_markets(api_key, sport, league, sportsbook=['Pinnacle']):
    url = "https://api.opticodds.com/api/v3/markets"
    params = {
        'sport': sport,
        'league': league,
        'key': api_key
    }
    if sportsbook:
        params['sportsbook'] = sportsbook
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data['data'])
    else:
        print(f"Failed to fetch markets: {response.status_code} - {response.text}")
        return pd.DataFrame()

# Get today's game IDs dynamically
def get_todays_game_ids(api_key, league):
    Client = OddsJamClient(api_key)
    Client.UseV2()
    GamesResponse = Client.GetGames(league=league)
    Games = GamesResponse.Games
    games_data = [{'game_id': game.id, 'start_date': game.start_date} for game in Games]
    games_df = pd.DataFrame(games_data)
    games_df['start_date'] = pd.to_datetime(games_df['start_date'])
    return games_df['game_id'].tolist()

# Fetch game data dynamically and filter based on player or game markets
def fetch_game_data(game_ids, api_key, market_type='game', sport='baseball', league='MLB', sportsbooks=['Pinnacle', 'Caesars'], include_player_name=True):
    markets_df = fetch_sports_markets(api_key, sport, league, sportsbooks)

    if market_type == 'player':
        markets = markets_df[markets_df['name'].str.contains('Player', case=False)]['name'].tolist()
    elif market_type == 'game':
        markets = markets_df[~markets_df['name'].str.contains('Player', case=False)]['name'].tolist()
    else:
        print(f"Unknown market type: {market_type}")
        return pd.DataFrame()

    url = "https://api-external.oddsjam.com/api/v2/game-odds"
    all_data = []
    
    for chunk in [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]:
        for sportsbook in sportsbooks:
            params = {
                'key': api_key,
                'sportsbook': sportsbook,
                'game_id': chunk,
                'market_name': markets
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json().get('data', [])
                all_data.extend(data)
            else:
                print(f"Error fetching data: {response.status_code} - {response.text}")
    
    rows = []
    for game_data in all_data:
        home_team = game_data.get('home_team', 'Unknown')
        away_team = game_data.get('away_team', 'Unknown')
        odds_list = game_data.get('odds', [])
        
        for item in odds_list:
            row = {
                'Game ID': game_data.get('id', 'Unknown'),
                "Game Name": f"{home_team} vs {away_team}",
                "Bet Name": item.get('name', None),
                'Market Name': item.get('market_name', ''),
                'Sportsbook': item.get('sports_book_name', sportsbook),
                'line': item.get('bet_points', None),
                'Odds': item.get('price', None),
            }
            if include_player_name and market_type == 'player':
                row['Player Name'] = item.get('selection', 'Unknown')
            rows.append(row)
    
    return pd.DataFrame(rows)

# Find plus EV bets by merging sportsbook data and calculating EV
def find_plus_ev_bets(df, threshold=5):
    caesars_df = df[df['Sportsbook'] == 'Caesars']
    pinnacle_df = df[df['Sportsbook'] == 'Pinnacle']
    betmgm_df = df[df['Sportsbook'] == 'BetMGM'].rename(columns={'Odds': 'Odds_betmgm'})
    
    merged_df = pd.merge(caesars_df, pinnacle_df, on=['Game ID', 'Bet Name', 'Market Name'], suffixes=('_caesars', '_pinnacle'))
    merged_df = pd.merge(merged_df, betmgm_df, on=['Game ID', 'Bet Name', 'Market Name'], how='left')
    
    merged_df['decimal_odds_caesars'] = merged_df['Odds_caesars'].apply(american_to_decimal)
    merged_df['decimal_odds_pinnacle'] = merged_df['Odds_pinnacle'].apply(american_to_decimal)
    merged_df['decimal_odds_betmgm'] = merged_df['Odds_betmgm'].apply(american_to_decimal)
    
    merged_df['implied_prob_caesars'] = merged_df['decimal_odds_caesars'].apply(implied_probability)
    merged_df['implied_prob_pinnacle'] = merged_df['decimal_odds_pinnacle'].apply(implied_probability)
    
    merged_df['true_prob_pinnacle'] = merged_df['implied_prob_pinnacle'].apply(adjust_for_vig)
    
    merged_df['EV_caesars'] = merged_df.apply(lambda row: calculate_ev(row['true_prob_pinnacle'], row['Odds_caesars']), axis=1)
    merged_df['EV_betmgm'] = merged_df.apply(lambda row: calculate_ev(row['true_prob_pinnacle'], row['Odds_betmgm']), axis=1)
    
    positive_ev_bets = merged_df[(merged_df['EV_caesars'] > threshold) | (merged_df['EV_betmgm'] > threshold)]
    return positive_ev_bets

# Function to save the DataFrame to a SQL database using sqlite3
def save_to_sql(df, league, conn, table_prefix='betting_data'):
    table_name = f"{table_prefix}_{league}"
    df.to_sql(table_name, conn, if_exists='append', index=False)

# Example main logic
def get_ev_bets(api_key, sport, league, threshold=5):
    game_ids = get_todays_game_ids(api_key, league)
    
    player_props_df = fetch_game_data(game_ids, api_key, market_type='player', sport=sport, league=league, sportsbooks=['Pinnacle', 'Caesars'])
    player_ev_bets = find_plus_ev_bets(player_props_df, threshold=threshold)
    
    game_props_df = fetch_game_data(game_ids, api_key, market_type='game', sport=sport, league=league, sportsbooks=['Pinnacle', 'Caesars'])
    game_ev_bets = find_plus_ev_bets(game_props_df, threshold=threshold)
    
    final_ev_df = pd.concat([player_ev_bets, game_ev_bets], ignore_index=True)
    
    return final_ev_df
