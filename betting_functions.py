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

# Adjust implied probability to account for the vig (adjusting for FanDuel higher vig)
def adjust_for_vig(implied_prob, sportsbook, vig_reduction=0.02):
    """
    Adjust for the vig based on the sportsbook. FanDuel typically has a higher vig.
    """
    if sportsbook == 'FanDuel':
        vig_reduction = 0.06  # Adjust for FanDuel's higher vig (-113 on both sides)
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
def calculate_avg_odds(df):
    # Convert odds to decimal if necessary and calculate avg_odds
    df['decimal_odds_pinnacle'] = df['Odds_pinnacle'].apply(american_to_decimal)
    df['decimal_odds_circa'] = df['Odds_circa'].apply(american_to_decimal)

    # Calculate the average odds
    df['avg_odds'] = df[['decimal_odds_pinnacle', 'decimal_odds_circa']].mean(axis=1)
    return df

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
def fetch_game_data(game_ids, api_key, market_type='game', sport='baseball', league='MLB', sportsbooks=['Pinnacle', 'Circa Sports', 'Caesars'], include_player_name=True):
    markets_df = fetch_sports_markets(api_key, sport, league, sportsbooks)
    if market_type == 'player':
        markets = markets_df[markets_df['name'].str.contains('Player', case=False)]['name'].tolist()
    elif market_type == 'game':
        markets = markets_df[~markets_df['name'].str.contains('Player', case=False)]['name'].tolist()
    else:
        print(f"Unknown market type: {market_type}")
        return pd.DataFrame()
    print(markets)
    print(game_ids)
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
def update_table_schema(conn, table_name, df):
    cursor = conn.cursor()
    
    # Check if the table exists
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = [info[1] for info in cursor.fetchall()]

    # Add missing columns if they don't exist
    for col in df.columns:
        if col not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN '{col}' TEXT")
                print(f"Added missing column: {col}")
            except sqlite3.OperationalError as e:
                print(f"Error adding column {col}: {e}")
    
    conn.commit()

def find_plus_ev_bets(df, sportsbook, odds_threshold=10, threshold=5, league=None):
    """
    Modify the function to include FanDuel for NBA and other sportsbooks for other leagues.
    """
    if league == 'NBA':
        # For NBA, use FanDuel as the true probability source
        fanduel_df = df[df['Sportsbook'] == 'FanDuel'].rename(columns={'Odds': 'Odds_fanduel'})
        user_sportsbook_df = df[df['Sportsbook'] == sportsbook]
        
        # Merge with FanDuel data
        merged_df = pd.merge(user_sportsbook_df, fanduel_df, on=['Game ID', 'Bet Name', 'Market Name'], suffixes=('_user', '_fanduel'))
        # Convert odds to decimal and calculate implied probabilities
        merged_df = fetch_true_probabilities_fanduel(merged_df)

        # Adjust for vig
        merged_df['true_prob_avg'] = merged_df['implied_prob_fanduel'].apply(adjust_for_vig)

    else:
        # For non-NBA leagues, use Pinnacle and Circa as true probability sources
        user_sportsbook_df = df[df['Sportsbook'] == sportsbook]
        pinnacle_df = df[df['Sportsbook'] == 'Pinnacle']
        circa_df = df[df['Sportsbook'] == 'Circa Sports'].rename(columns={'Odds': 'Odds_circa'})
        
        # Merge user data with Pinnacle and Circa
        merged_df = pd.merge(user_sportsbook_df, pinnacle_df, on=['Game ID', 'Bet Name', 'Market Name'], suffixes=('_user', '_pinnacle'))
        merged_df = pd.merge(merged_df, circa_df, on=['Game ID', 'Bet Name', 'Market Name'], how='left')

        # Convert odds to decimal
        merged_df['decimal_odds_user'] = merged_df['Odds_user'].apply(american_to_decimal)
        merged_df['decimal_odds_pinnacle'] = merged_df['Odds_pinnacle'].apply(american_to_decimal)
        merged_df['decimal_odds_circa'] = merged_df['Odds_circa'].apply(american_to_decimal)

        # Calculate implied probabilities
        merged_df['implied_prob_pinnacle'] = merged_df['decimal_odds_pinnacle'].apply(implied_probability)
        merged_df['implied_prob_circa'] = merged_df['decimal_odds_circa'].apply(implied_probability)

        # Average the true probabilities from Pinnacle and Circa
        merged_df['true_prob_avg'] = merged_df[['implied_prob_pinnacle', 'implied_prob_circa']].mean(axis=1)
        merged_df['true_prob_avg'] = merged_df['true_prob_avg'].apply(adjust_for_vig)

    # Filter out odds that exceed the threshold
    merged_df = merged_df[merged_df['decimal_odds_user'] <= odds_threshold]

    # Calculate EV for user-specified sportsbook
    merged_df['EV_user'] = merged_df.apply(lambda row: calculate_ev(row['true_prob_avg'], row['Odds_user']), axis=1)
    
    # Filter out bets that have EV above the threshold
    positive_ev_bets = merged_df[merged_df['EV_user'] > threshold]
    return positive_ev_bets

def save_to_sql(df, league, conn, table_prefix='betting_data'):
    table_name = f"{table_prefix}_{league}"
    # Add 'Odds_circa' and 'avg_odds' if they're not already in the DataFrame
    if 'Odds_circa' not in df.columns:
        df['Odds_circa'] = None  # Assign None or fetch the appropriate values

    # Ensure 'avg_odds' is calculated before saving
    if 'avg_odds' not in df.columns:
        df = calculate_avg_odds(df)

    # Ensure table schema matches the DataFrame
    update_table_schema(conn, table_name, df)


    # Now, save the DataFrame to the SQLite database
    df.to_sql(table_name, conn, if_exists='append', index=False)

# Function to calculate true probabilities using FanDuel for NBA with higher vig adjustment
def fetch_true_probabilities_fanduel(df):
    """
    Fetch FanDuel odds and calculate the implied probabilities for NBA markets,
    adjusting for the higher vig at FanDuel.
    """
    df['decimal_odds_fanduel'] = df['Odds_fanduel'].apply(american_to_decimal)

    # Calculate implied probabilities for FanDuel and adjust for higher vig
    df['implied_prob_fanduel'] = df['decimal_odds_fanduel'].apply(implied_probability)
    df['true_prob_avg'] = df['implied_prob_fanduel'].apply(lambda x: adjust_for_vig(x, 'FanDuel'))

    return df

def find_plus_ev_bets(df, sportsbook, odds_threshold=10, threshold=5, league=None):
    """
    Modify the function to include FanDuel for NBA and other sportsbooks for other leagues.
    """
    if league == 'test':
        # For NBA, use FanDuel as the true probability source
        fanduel_df = df[df['Sportsbook'] == 'FanDuel'].rename(columns={'Odds': 'Odds_fanduel'})
        user_sportsbook_df = df[df['Sportsbook'] == sportsbook]

        # Merge with FanDuel data
        merged_df = pd.merge(user_sportsbook_df, fanduel_df, on=['Game ID', 'Bet Name', 'Market Name'], suffixes=('_user', '_fanduel'))
        merged_df.rename({'Odds':'Odds_user'},axis=1,inplace=True)
        # Ensure decimal odds are calculated for user odds
        merged_df['decimal_odds_user'] = merged_df['Odds_user'].apply(american_to_decimal)
        merged_df['decimal_odds_fanduel'] = merged_df['Odds_fanduel'].apply(american_to_decimal)

        # Calculate implied probabilities for FanDuel and adjust for vig
        merged_df['implied_prob_fanduel'] = merged_df['decimal_odds_fanduel'].apply(implied_probability)
        merged_df['true_prob_avg'] = merged_df['implied_prob_fanduel'].apply(lambda x: adjust_for_vig(x, 'FanDuel'))

    else:
        # For non-NBA leagues, use Pinnacle and Circa as true probability sources
        user_sportsbook_df = df[df['Sportsbook'] == sportsbook]
        pinnacle_df = df[df['Sportsbook'] == 'Pinnacle']
        circa_df = df[df['Sportsbook'] == 'Circa Sports'].rename(columns={'Odds': 'Odds_circa'})
        
        # Merge user data with Pinnacle and Circa
        merged_df = pd.merge(user_sportsbook_df, pinnacle_df, on=['Game ID', 'Bet Name', 'Market Name'], suffixes=('_user', '_pinnacle'))
        merged_df = pd.merge(merged_df, circa_df, on=['Game ID', 'Bet Name', 'Market Name'], how='left')

        # Convert odds to decimal
        merged_df['decimal_odds_user'] = merged_df['Odds_user'].apply(american_to_decimal)
        merged_df['decimal_odds_pinnacle'] = merged_df['Odds_pinnacle'].apply(american_to_decimal)
        merged_df['decimal_odds_circa'] = merged_df['Odds_circa'].apply(american_to_decimal)

        # Calculate implied probabilities
        merged_df['implied_prob_pinnacle'] = merged_df['decimal_odds_pinnacle'].apply(implied_probability)
        merged_df['implied_prob_circa'] = merged_df['decimal_odds_circa'].apply(implied_probability)

        # Average the true probabilities from Pinnacle and Circa
        merged_df['true_prob_avg'] = merged_df[['implied_prob_pinnacle', 'implied_prob_circa']].mean(axis=1)
        merged_df['true_prob_avg'] = merged_df['true_prob_avg'].apply(lambda x: adjust_for_vig(x, 'Pinnacle'))

    # Filter out odds that exceed the threshold
    merged_df = merged_df[merged_df['decimal_odds_user'] <= odds_threshold]

    # Calculate EV for user-specified sportsbook
    merged_df['EV_user'] = merged_df.apply(lambda row: calculate_ev(row['true_prob_avg'], row['Odds_user']), axis=1)
    
    # Filter out bets that have EV above the threshold
    positive_ev_bets = merged_df[merged_df['EV_user'] > threshold]
    
    return positive_ev_bets

# Main function to fetch EV bets, with FanDuel included for NBA
def get_ev_bets(api_key, sport, league, threshold=5, odds_threshold=10, sportsbook='Caesars'):
    game_ids = get_todays_game_ids(api_key, league)

    player_props_df = fetch_game_data(game_ids, api_key, market_type='player', sport=sport, league=league, sportsbooks=['Pinnacle', 'Circa Sports', sportsbook, 'FanDuel'])
    player_ev_bets = find_plus_ev_bets(player_props_df, sportsbook, odds_threshold=odds_threshold, threshold=threshold, league=league)

    game_props_df = fetch_game_data(game_ids, api_key, market_type='game', sport=sport, league=league, sportsbooks=['Pinnacle', 'Circa Sports', sportsbook, 'FanDuel'])
    game_ev_bets = find_plus_ev_bets(game_props_df, sportsbook, odds_threshold=odds_threshold, threshold=threshold, league=league)

    final_ev_df = pd.concat([player_ev_bets, game_ev_bets], ignore_index=True)

    return final_ev_df
