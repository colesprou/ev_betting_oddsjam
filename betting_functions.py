#!/usr/bin/env python
# coding: utf-8

# In[44]:
import requests
import pandas as pd
import sqlite3
from datetime import datetime
from OddsJamClient import OddsJamClient
import os
from datetime import datetime, timezone, timedelta


api_key = os.getenv('API_KEY')
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
def fetch_sports_markets(api_key, sport, league, sportsbook=['Pinnacle','BookMaker']):
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

def get_todays_game_ids(api_key, league, is_live='false'):
    endpoint = "https://api.opticodds.com/api/v3/fixtures"

    # Current date and time
    today = datetime.now(timezone.utc)
    future_date = today + timedelta(hours=48)

    # Parameters
    params = {
        "key": api_key,
        "league": league,
        "start_date_after": today,
        "start_date_before": future_date
    }

    # Make the request
    response = requests.get(endpoint, params=params)

    # Check response
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    api_response = response.json()
    game_data = {
        game['id']: f"{game['home_team_display']} vs {game['away_team_display']}"
        for game in api_response['data']
    }
    return game_data


# Fetch game data dynamically and filter based on player or game markets
def fetch_game_data(game_ids, api_key, market_type='game', sport='baseball', league='MLB', sportsbooks=None, include_player_name=True, is_live='false'):
    # Validate inputs
    if sportsbooks is None:
        sportsbooks = ['Pinnacle', 'FanDuel', 'DraftKings']

    # Fetch game names
    game_data_dict = get_todays_game_ids(api_key, league, is_live=is_live)
    if not game_data_dict:
        print("No game data found.")
        return pd.DataFrame()

    # Fetch markets for the given sport and league
    markets_df = fetch_sports_markets(api_key, sport, league, sportsbooks)
    if market_type == 'player':
        markets = markets_df[markets_df['name'].str.contains('Player', case=False)]['name'].tolist()
    elif market_type == 'game':
        markets = markets_df[~markets_df['name'].str.contains('Player', case=False)]['name'].tolist()
    else:
        print(f"Unknown market type: {market_type}")
        return pd.DataFrame()

    url = "https://api.opticodds.com/api/v3/fixtures/odds"
    all_data = []  # Collect all data across sportsbooks and chunks

    for chunk in [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]:
        for sportsbook in sportsbooks:
            params = {
                'key': api_key,
                'sportsbook': sportsbook,
                'fixture_id': chunk,
                'market_name': markets
            }
            if is_live != 'false':
                params['status'] = 'live'

            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json().get('data', [])
                for game_data in data:
                    # Add sportsbook info to each record
                    for item in game_data.get('odds', []):
                        all_data.append({
                            'Game ID': game_data.get('id', 'Unknown'),
                            'Game Name': game_data_dict.get(game_data.get('id', 'Unknown'), 'Unknown Game'),
                            'Bet Name': item.get('name', None),
                            'Market Name': item.get('market', ''),
                            'Sportsbook': sportsbook,
                            'line': item.get('points', None),
                            'Odds': item.get('price', None),
                            **({'Player Name': item.get('selection', 'Unknown')} if include_player_name and market_type == 'player' else {})
                        })
            else:
                print(f"Error fetching data for sportsbook {sportsbook}: {response.status_code} - {response.text}")

    return pd.DataFrame(all_data)
def fetch_closing_line_game_data(game_ids, api_key, market_type='game', sport='basketball', league='NBA', 
                                 sportsbooks=['FanDuel', 'BetOnline', 'Caesars'], include_player_name=True, is_live='false'):
    """
    Fetches game data for FanDuel and specified sportsbooks, suitable for closing-line calculations.
    """
    # Fetch FanDuel markets and additional sportsbook markets
    markets_df = fetch_fanduel_markets(api_key, sport, league) if 'FanDuel' in sportsbooks else pd.DataFrame()
    other_markets_df = fetch_sports_markets(api_key, sport, league, [s for s in sportsbooks if s != 'FanDuel'])
    
    # Combine FanDuel markets with other markets
    combined_markets_df = pd.concat([markets_df, other_markets_df], ignore_index=True)

    # Filter based on market type
    if market_type == 'player':
        markets = combined_markets_df[combined_markets_df['name'].str.contains('Player', case=False)]['name'].tolist()
    elif market_type == 'game':
        markets = combined_markets_df[~combined_markets_df['name'].str.contains('Player', case=False)]['name'].tolist()
    else:
        print(f"Unknown market type: {market_type}")
        return pd.DataFrame()
    
    url = "https://api-external.oddsjam.com/api/v2/game-odds"
    all_data = []
    
    # Fetch odds data in chunks
    for chunk in [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]:
        for sportsbook in sportsbooks:
            params = {
                'key': api_key,
                'sportsbook': sportsbook,
                'game_id': chunk,
                'market_name': markets,
                'is_main':True
            }
            if is_live == 'true':
                params['status'] = 'live'
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json().get('data', [])
                all_data.extend(data)
            else:
                print(f"Error fetching data: {response.status_code} - {response.text}")
    
    # Parse the response data
    rows = []
    for game_data in all_data:
        home_team = game_data.get('home_team', 'Unknown')
        away_team = game_data.get('away_team', 'Unknown')
        start_date = game_data.get('start_date', 'Unknown')  # Ensure start_date is captured
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
                'start_date': start_date  # Add start_date to each row
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
def get_player_ev_bets(api_key, sport, league, sportsbook='Caesars', is_live='false'):
    # Retrieve game IDs for today's games
    game_ids = get_todays_game_ids(api_key, league, is_live=is_live)
    
    # Fetch player props
    player_props_df = fetch_game_data(
        list(game_ids.keys()), api_key, market_type='player', sport=sport, league=league,
        sportsbooks=['Pinnacle','BetOnline','Blue Book','DraftKings',"Bookmaker","Caesars", "Circa Vegas",'Novig',"Prophet X",sportsbook], is_live=is_live
    )
    player_props_df.to_csv('DEBUG_PLAYER_PROPS.csv', index=False)

    # Calculate EV for player props
    #player_ev_bets = find_plus_ev_bets(player_props_df, sportsbook, odds_threshold=odds_threshold, threshold=threshold)
    
    # Save results for debugging
    #player_ev_bets.to_csv('DEBUG_PLAYER_PROPS_AFTER_EV.csv', index=False)
    
    return player_props_df
def get_game_ev_bets(api_key, sport, league, sportsbook='Caesars',is_live='false'):
    # Retrieve game IDs for today's games
    game_ids = get_todays_game_ids(api_key, league, is_live=is_live)
    
    # Fetch game props
    game_props_df = fetch_game_data(
        list(game_ids.keys()), api_key, market_type='game', sport=sport, league=league,
        sportsbooks=['Pinnacle','BetOnline','Blue Book','DraftKings',"Bookmaker", "Caesars","Circa Vegas","Betcris","Novig",'Prophet X',sportsbook], is_live=is_live
    )
    game_props_df.to_csv('DEBUG_GAME_PROPS.csv', index=False)

    # Calculate EV for game props
    #game_ev_bets = find_plus_ev_bets(game_props_df, sportsbook, odds_threshold=odds_threshold, threshold=threshold)
    # Save results for debugging
    #game_ev_bets.to_csv('DEBUG_GAME_PROPS_AFTER_EV.csv', index=False)

    return game_props_df
def get_ev_bets(api_key, sport, league, odds_threshold=10, sportsbook='Caesars', is_live='false', threshold=0.01):
    # Get player and game EV bets separately
    player_ev_bets = get_player_ev_bets(
        api_key, sport, league, 
        sportsbook=sportsbook, is_live=is_live
    )
    print("Player EV Bets Retrieved:", player_ev_bets)  # Debugging
    
    game_ev_bets = get_game_ev_bets(
        api_key, sport, league,
        sportsbook=sportsbook, is_live=is_live
    )
    print("Game EV Bets Retrieved:", game_ev_bets)  # Debugging

    # Standardize column names for both datasets
    common_columns = [
        "Game ID", "Game Name", "line", "Market Name", "Bet Name A", "Bet Name B",
        "Odds A User", "Odds B User", "Odds A Pinnacle", "Odds B Pinnacle",
        "Market_Type", "Player Name", "Decimal Odds A", "Decimal Odds B",
        "Implied Probability A", "Implied Probability B", "Total Implied Probability",
        "True Probability A", "True Probability B", "EV A", "EV B"
    ]

    # Exclude Player Name if not present in game_ev_bets
    game_columns = [col for col in common_columns if col in game_ev_bets.columns]
    player_columns = [col for col in common_columns if col in player_ev_bets.columns]

    game_ev_bets = game_ev_bets[game_columns]
    player_ev_bets = player_ev_bets[player_columns]

    # Combine the results
    final_ev_df = pd.concat([player_ev_bets, game_ev_bets], ignore_index=True)
    final_ev_df.to_csv('testing_ev_combined.csv', index=False)  # Debugging: save combined EV bets
    
    return final_ev_df

def get_ev_bets_combined(api_key, sport, league, threshold=-20, odds_threshold=10, sportsbook='Caesars', is_live='false'):
    # Get player EV bets
    if not league == 'NCAAB':
        player_ev_bets = get_player_ev_bets(api_key, sport, league, threshold, odds_threshold, sportsbook, is_live)
    
    # Get game EV bets
    game_ev_bets = get_game_ev_bets(api_key, sport, league, threshold, odds_threshold, sportsbook, is_live)
    
    # Combine both
    if not league == 'NCAAB':
        final_ev_df = pd.concat([player_ev_bets, game_ev_bets], ignore_index=True)
    else:
        final_ev_df = game_ev_bets
    final_ev_df.to_csv('final_ev_bets_combined.csv', index=False)
    
    return final_ev_df



def find_plus_ev_bets(df, sportsbook, odds_threshold=10, threshold=0.05):
    print("Columns in the DataFrame:", df.columns)

    def pair_over_under_with_market_name(df):
        totals = df[df["Market Name"].str.contains("Total Points", na=False)]
        over_bets = totals[totals["Bet Name_user"].str.contains("Over", na=False)].copy()
        under_bets = totals[totals["Bet Name_user"].str.contains("Under", na=False)].copy()
        over_bets["line"] = over_bets["line"].astype(float)
        under_bets["line"] = under_bets["line"].astype(float)
        paired_totals = pd.merge(
            over_bets,
            under_bets,
            on=["Game ID", "line"],
            suffixes=("_over", "_under"),
            how="inner"
        )
        return paired_totals[[
            "Game ID", "Game Name_user_over", "line", "Market Name_over",
            "Bet Name_user_over", "Bet Name_user_under",
            "Odds_user_over", "Odds_user_under",
            "Odds_pinnacle_over", "Odds_pinnacle_under"
        ]].rename(columns={
            "Game Name_user_over": "Game Name",
            "Market Name_over": "Market Name",
            "Bet Name_user_over": "Over Bet Name",
            "Bet Name_user_under": "Under Bet Name",
            "Odds_user_over": "Over Odds_user",
            "Odds_user_under": "Under Odds_user",
            "Odds_pinnacle_over": "Over Odds_pinnacle",
            "Odds_pinnacle_under": "Under Odds_pinnacle"
        })

    def add_market_context(df):
        df["Market Context"] = df["Market Name"].apply(
            lambda x: "1st Quarter Total" if "1st Quarter" in x else
                      "1st Half Total" if "1st Half" in x else
                      "Game Total" if "Total Points" in x else
                      "Unknown"
        )
        return df

    def pair_point_spreads_with_absolute_matching(df):
        spreads = df[df["Market Name"].str.contains("Point Spread", na=False)]
        spreads["abs_line"] = spreads["line"].abs()
        positive_spreads = spreads[spreads["Bet Name_user"].str.contains("\+", na=False)].copy()
        negative_spreads = spreads[spreads["Bet Name_user"].str.contains("-", na=False)].copy()
        paired_spreads = pd.merge(
            positive_spreads,
            negative_spreads,
            on=["Game ID", "abs_line"],
            suffixes=("_positive", "_negative"),
            how="inner"
        )
        return paired_spreads[[
            "Game ID", "Game Name_user_positive", "line_positive",
            "Bet Name_user_positive", "Bet Name_user_negative",
            "Odds_user_positive", "Odds_user_negative",
            "Odds_pinnacle_positive", "Odds_pinnacle_negative"
        ]].rename(columns={
            "Game Name_user_positive": "Game Name",
            "line_positive": "Line",
            "Bet Name_user_positive": "Positive Bet Name",
            "Bet Name_user_negative": "Negative Bet Name",
            "Odds_user_positive": "Positive Odds_user",
            "Odds_user_negative": "Negative Odds_user",
            "Odds_pinnacle_positive": "Positive Odds_pinnacle",
            "Odds_pinnacle_negative": "Negative Odds_pinnacle"
        })

    def pair_player_props(df):
        props = df[df["Market Name"].str.contains("Player", na=False)]
        over_bets = props[props["Bet Name_user"].str.contains("Over", na=False)].copy()
        under_bets = props[props["Bet Name_user"].str.contains("Under", na=False)].copy()
        print()
        paired_props = pd.merge(
            over_bets,
            under_bets,
            on=["Game ID", "Player Name", "line"],
            suffixes=("_over", "_under"),
            how="inner"
        )
        return paired_props[[
            "Game ID", "Game Name_user_over", "Player Name", "line", "Market Name_over",
            "Bet Name_user_over", "Bet Name_user_under",
            "Odds_user_over", "Odds_user_under",
            "Odds_pinnacle_over", "Odds_pinnacle_under"
        ]].rename(columns={
            "Game Name_user_over": "Game Name",
            "Market Name_over": "Market Name",
            "Bet Name_user_over": "Over Bet Name",
            "Bet Name_user_under": "Under Bet Name",
            "Odds_user_over": "Over Odds_user",
            "Odds_user_under": "Under Odds_user",
            "Odds_pinnacle_over": "Over Odds_pinnacle",
            "Odds_pinnacle_under": "Under Odds_pinnacle"
        })

    def normalize_columns(df, market_type):
        return df.rename(
            columns={
                "Over Bet Name": "Bet Name A",
                "Under Bet Name": "Bet Name B",
                "Positive Bet Name": "Bet Name A",
                "Negative Bet Name": "Bet Name B",
                "Over Odds_user": "Odds A User",
                "Under Odds_user": "Odds B User",
                "Positive Odds_user": "Odds A User",
                "Negative Odds_user": "Odds B User",
                "Over Odds_pinnacle": "Odds A Pinnacle",
                "Under Odds_pinnacle": "Odds B Pinnacle",
                "Positive Odds_pinnacle": "Odds A Pinnacle",
                "Negative Odds_pinnacle": "Odds B Pinnacle"
            }
        ).assign(Market_Type=market_type)

    # Process the data
    user_sportsbook_df = df[df['Sportsbook'] == sportsbook]
    pinnacle_df = df[df['Sportsbook'] == 'Pinnacle']

    merge_keys = ['Game ID', 'Market Name', 'line']
    if 'Player Name' in user_sportsbook_df.columns:
        merge_keys.append('Player Name')

    merged_df = pd.merge(
        user_sportsbook_df, 
        pinnacle_df, 
        on=merge_keys, 
        suffixes=('_user', '_pinnacle')
    ).dropna(subset=['Odds_user', 'Odds_pinnacle'])

    # Process each market type
    paired_totals = pair_over_under_with_market_name(merged_df)
    paired_spreads = pair_point_spreads_with_absolute_matching(merged_df)
    # Process player props only if Player Name exists in both datasets
    paired_props = pd.DataFrame()
    if 'Player Name' in merge_keys:
        print("Processing player props...")
        paired_props = pair_player_props(merged_df)
        print("Paired player props:", paired_props.head())  # Debugging
    # Normalize and combine
    normalized_totals = normalize_columns(paired_totals, "Totals")
    normalized_spreads = normalize_columns(paired_spreads, "Point Spreads")
    normalized_props = pd.DataFrame()
    if not paired_props.empty:
        normalized_props = normalize_columns(paired_props, "Player Props")

    combined_markets = pd.concat(
        [normalized_totals, normalized_spreads, normalized_props],
        ignore_index=True, sort=False
    )

    print("Combined markets:", combined_markets.head())  # Debugging

    # Calculate EV for all markets
    final_df = calculate_ev_for_midpoint(combined_markets)
    final_df.to_csv('before_ev_midpoint.csv')
    plus_ev_bets = final_df[
        (final_df["EV A"] > threshold) | (final_df["EV B"] > threshold)
    ]
    plus_ev_bets.to_csv('new_midpoint_df.csv', index=False)
    return plus_ev_bets



def calculate_ev_for_midpoint(df):
    """
    Calculates midpoint implied probability, true probability, and expected value (EV) for each row.
    """
    def american_to_decimal(odds):
        """Convert American odds to decimal odds."""
        if pd.isnull(odds):  # Handle missing odds
            return None
        if odds > 0:
            return (odds / 100) + 1
        elif odds < 0:
            return (100 / abs(odds)) + 1
        else:
            raise ValueError("Odds cannot be zero")

    def implied_probability(decimal_odds):
        """Convert decimal odds to implied probability."""
        if decimal_odds is None:  # Handle missing odds
            return None
        return 1 / decimal_odds

    def calculate_ev_row(true_prob, decimal_user_odds):
        """Calculate EV using true probability and decimal user odds."""
        if true_prob is None or decimal_user_odds is None:
            return None
        payout = decimal_user_odds - 1  # Net payout
        return (true_prob * payout) - (1 - true_prob)

    # Calculate midpoint odds for Pinnacle
    df["Midpoint Odds Pinnacle"] = (df["Odds A Pinnacle"] + df["Odds B Pinnacle"]) / 2

    # Convert midpoint odds to implied probability (True Probability)
    df["True Probability"] = df["Midpoint Odds Pinnacle"].apply(
        lambda x: implied_probability(american_to_decimal(x))
    )

    # Calculate EV for User Odds
    df["EV A"] = df.apply(
        lambda row: calculate_ev_row(
            row["True Probability"], american_to_decimal(row["Odds A User"])
        ),
        axis=1,
    )
    df["EV B"] = df.apply(
        lambda row: calculate_ev_row(
            row["True Probability"], american_to_decimal(row["Odds B User"])
        ),
        axis=1,
    )

    return df






























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

def get_ev_bet_old(api_key, sport, league, threshold=-20, odds_threshold=10, sportsbook='Caesars', is_live='false'):
    # Retrieve game IDs for today's games
    game_ids = get_todays_game_ids(api_key, league, is_live=is_live)
    
    # Fetch player props, now including the is_live parameter
    player_props_df = fetch_game_data(game_ids, api_key, market_type='player', sport=sport, league=league, 
                                      sportsbooks=['Pinnacle', 'Circa Sports', 'FanDuel',sportsbook], is_live=is_live)
    player_props_df.to_csv('debug_player_props.csv', index=False)
    # Find EV bets for player props
    player_ev_bets = find_plus_ev_bets(player_props_df, sportsbook, odds_threshold=odds_threshold, threshold=threshold)
    # Fetch game props, also passing the is_live parameter
    game_props_df = fetch_game_data(game_ids, api_key, market_type='game', sport=sport, league=league, 
                                    sportsbooks=['Pinnacle', 'Circa Sports','FanDuel', sportsbook], is_live=is_live)
    player_ev_bets.to_csv('player_props_after_ev.csv')
    # Find EV bets for game props
    game_ev_bets = find_plus_ev_bets(game_props_df, sportsbook, odds_threshold=odds_threshold, threshold=threshold)
    
    # Combine the player and game EV bets
    final_ev_df = pd.concat([player_ev_bets, game_ev_bets], ignore_index=True)
    final_ev_df.to_csv('testing_ev.csv')
    return final_ev_df
def calculate_midpoint(prob_a, prob_b):
    """Calculate midpoint true probability given both sides."""
    if prob_a is not None and prob_b is not None:
        return (prob_a + prob_b) / 2  # Average both sides
    else:
        return None  # Return None if either side is missing

def get_ev_bets_fanduel_pinnacle(api_key, sport, league, threshold=0, odds_threshold=10, sportsbook='Caesars', is_live='false'):
    # Retrieve game IDs for today's games
    game_ids = get_todays_game_ids(api_key, league, is_live=is_live)
    
    # Fetch player props, now including the is_live parameter
    player_props_df = fetch_game_data(game_ids, api_key, market_type='player', sport=sport, league=league, 
                                      sportsbooks=['Pinnacle',  'FanDuel',sportsbook], is_live=is_live)
    #player_props_df.to_csv('player_props_nba_test.csv')
    # Find EV bets for player props
    player_ev_bets = find_plus_ev_bets_fanduel_pinnacle(player_props_df, sportsbook, odds_threshold=odds_threshold, threshold=threshold)
    #print(player_ev_bets.to_csv('player_props_nba_test_after_ev.csv'))
    # Fetch game props, also passing the is_live parameter
    game_props_df = fetch_game_data(game_ids, api_key, market_type='game', sport=sport, league=league, 
                                    sportsbooks=['Pinnacle', 'FanDuel', sportsbook], is_live=is_live)
    #game_props_df.to_csv('game_props_nba_test.csv')
    # Find EV bets for game props
    game_ev_bets = find_plus_ev_bets_fanduel_pinnacle(game_props_df, sportsbook, odds_threshold=odds_threshold, threshold=threshold)
    #game_ev_bets.to_csv('game_props_nba_test_after_ev.csv')
    # Combine the player and game EV bets
    final_ev_df = pd.concat([player_ev_bets, game_ev_bets], ignore_index=True)
    print(final_ev_df)
    return final_ev_df
def get_ev_bets_all(api_key, sport, league, threshold=0, odds_threshold=10000, sportsbook='Caesars', is_live='false'):
    # Retrieve game IDs for today's games
    game_ids = get_todays_game_ids(api_key, league, is_live=is_live)
    
    # Fetch player props, now including the is_live parameter
    player_props_df = fetch_game_data(game_ids, api_key, market_type='player', sport=sport, league=league, 
                                      sportsbooks=['Pinnacle','FanDuel','Circa Sports','BetOnline','DraftKings','BookMaker',
                                                   'Caesars','BetMGM','Hard Rock','PointsBet','William Hill','WynnBET','BetUS','BetRivers',
                                                   '888sport'], is_live=is_live)
    #player_props_df.to_csv('player_props_nba_test.csv')
    # Find EV bets for player props
    #player_ev_bets = find_plus_ev_bets_fanduel_pinnacle(player_props_df, sportsbook, odds_threshold=odds_threshold, threshold=threshold)
    #print(player_ev_bets.to_csv('player_props_nba_test_after_ev.csv'))
    # Fetch game props, also passing the is_live parameter
    game_props_df = fetch_game_data(game_ids, api_key, market_type='game', sport=sport, league=league, 
                                    sportsbooks=['Pinnacle','FanDuel','Circa Sports','BetOnline','DraftKings','BookMaker',
                                                   'Caesars','BetMGM','Hard Rock','PointsBet','William Hill','WynnBET','BetUS','BetRivers',
                                                   '888sport'], is_live=is_live)
    #game_props_df.to_csv('game_props_nba_test.csv')
    # Find EV bets for game props
    #game_ev_bets = find_plus_ev_bets_fanduel_pinnacle(game_props_df, sportsbook, odds_threshold=odds_threshold, threshold=threshold)
    #game_ev_bets.to_csv('game_props_nba_test_after_ev.csv')
    # Combine the player and game EV bets
    final_ev_df = pd.concat([player_props_df, game_props_df], ignore_index=True)
    return final_ev_df

def find_plus_ev_bets_fanduel_pinnacle(df, sportsbook, odds_threshold=10, threshold=-5):
    print(df.columns)

    # Helper functions
    def american_to_decimal(odds):
        """Convert American odds to decimal odds."""
        if odds > 0:
            return 1 + odds / 100
        else:
            return 1 + 100 / abs(odds)

    def implied_probability(decimal_odds):
        """Calculate implied probability from decimal odds."""
        return 1 / decimal_odds

    def calculate_true_probabilities(odds_list):
        """Calculate true probabilities for a list of American odds."""
        decimal_odds = [american_to_decimal(o) for o in odds_list]
        implied_probs = [implied_probability(d) for d in decimal_odds]
        total_implied = sum(implied_probs)  # Sum of implied probabilities (with vig)
        return [prob / total_implied for prob in implied_probs]  # Adjusted true probabilities

    # Filter dataframes for user sportsbook, Pinnacle, and FanDuel
    user_sportsbook_df = df[df['Sportsbook'] == sportsbook]
    pinnacle_df = df[df['Sportsbook'] == 'Pinnacle']
    fanduel_df = df[df['Sportsbook'] == 'FanDuel'].rename(columns={'Odds': 'Odds_fanduel'})  # Rename Odds column for FanDuel

    # Merge dataframes on shared identifiers
    merged_df = pd.merge(user_sportsbook_df, pinnacle_df, on=['Game ID', 'Bet Name', 'Market Name'], suffixes=('_user', '_pinnacle'))
    merged_df = pd.merge(merged_df, fanduel_df, on=['Game ID', 'Bet Name', 'Market Name'], how='left')

    # Drop rows with missing odds
    merged_df = merged_df.dropna(subset=['Odds_pinnacle', 'Odds_fanduel'])

    # Group by game and market to calculate true probabilities
    grouped = merged_df.groupby(['Game ID', 'Market Name'])

    # Initialize columns for true probabilities
    merged_df['true_prob_pinnacle'] = None
    merged_df['true_prob_fanduel'] = None

    # Calculate true probabilities for each group
    for name, group in grouped:
        if len(group) < 2:  # Skip groups without both sides of the bet
            continue

        # Calculate Pinnacle true probabilities
        pinnacle_probs = calculate_true_probabilities(group['Odds_pinnacle'].tolist())
        merged_df.loc[group.index, 'true_prob_pinnacle'] = pinnacle_probs

        # Calculate FanDuel true probabilities
        fanduel_probs = calculate_true_probabilities(group['Odds_fanduel'].tolist())
        merged_df.loc[group.index, 'true_prob_fanduel'] = fanduel_probs

    # Ensure columns are numeric
    merged_df['true_prob_pinnacle'] = pd.to_numeric(merged_df['true_prob_pinnacle'], errors='coerce')
    merged_df['true_prob_fanduel'] = pd.to_numeric(merged_df['true_prob_fanduel'], errors='coerce')

    # Average Pinnacle and FanDuel true probabilities for the final true probability
    merged_df['true_prob_avg'] = merged_df[['true_prob_pinnacle', 'true_prob_fanduel']].mean(axis=1)

    # Convert user odds to decimal and add a column
    merged_df['decimal_odds_user'] = merged_df['Odds_user'].apply(american_to_decimal)

    # Filter bets based on odds threshold
    merged_df = merged_df[merged_df['decimal_odds_user'] <= odds_threshold]

    # Calculate EV for user odds
    def calculate_ev(true_prob, odds_user):
        """Calculate expected value for a given true probability and user odds."""
        decimal_odds_user = american_to_decimal(odds_user)
        return (true_prob * (decimal_odds_user - 1)) - (1 - true_prob)

    merged_df['EV_user'] = merged_df.apply(
        lambda row: calculate_ev(row['true_prob_avg'], row['Odds_user']), axis=1
    )

    # Save intermediate results for debugging
    merged_df.to_csv('debug_fanduel_pinnacle_fixed.csv', index=False)
    
    # Filter for positive EV bets
    positive_ev_bets = merged_df[merged_df['EV_user'] > threshold]

    return positive_ev_bets

def calculate_midpoint_true_prob(df):
    """
    Calculates the midpoint-based true probability from FanDuel odds.
    """
    # Convert odds to decimal for FanDuel
    df['decimal_odds_user'] = df['Odds_user'].apply(american_to_decimal)
    df['decimal_odds_fanduel'] = df['Odds_fanduel'].apply(american_to_decimal)
    
    # Calculate the midpoint for FanDuel odds
    df['midpoint_odds'] = (df['decimal_odds_fanduel'] + df['decimal_odds_user']) / 2

    # Convert midpoint to true probability
    df['true_prob_midpoint'] = 1 / df['midpoint_odds']
    
    return df

def find_plus_ev_bets_fanduel_midpoint(df, sportsbooks, odds_threshold=10, threshold=5):
    """
    Finds +EV bets by comparing specified sportsbooks with FanDuel's midpoint-based true probability.
    """
    # Filter for user-specified sportsbooks and FanDuel
    user_sportsbook_df = df[df['Sportsbook'].isin(sportsbooks)].rename(columns={'Odds': 'Odds_user'})
    fanduel_df = df[df['Sportsbook'] == 'FanDuel'].rename(columns={'Odds': 'Odds_fanduel'})
    
    # Merge user sportsbook data with FanDuel data
    merged_df = pd.merge(user_sportsbook_df, fanduel_df, on=['Game ID', 'Bet Name', 'Market Name'], suffixes=('_user', '_fanduel'))
    
    # Ensure required columns are present
    if 'Odds_user' not in merged_df.columns or 'Odds_fanduel' not in merged_df.columns:
        print("Error: Required 'Odds' columns are missing after merging.")
        return pd.DataFrame()  # Return an empty DataFrame if critical columns are missing

    # Convert odds to decimal format and calculate midpoint
    merged_df['decimal_odds_user'] = merged_df['Odds_user'].apply(american_to_decimal)
    merged_df['decimal_odds_fanduel'] = merged_df['Odds_fanduel'].apply(american_to_decimal)
    merged_df['midpoint_odds'] = (merged_df['decimal_odds_fanduel'] + merged_df['decimal_odds_user']) / 2
    merged_df['true_prob_midpoint'] = 1 / merged_df['midpoint_odds']
    
    # Calculate EV for user-specified sportsbooks using midpoint probability
    merged_df['EV_user'] = merged_df.apply(lambda row: calculate_ev(row['true_prob_midpoint'], row['Odds_user']), axis=1)
    
    # Apply odds threshold and EV filter
    filtered_df = merged_df[(merged_df['decimal_odds_user'] <= odds_threshold) & (merged_df['EV_user'] > threshold)]

    # Select and order only the required columns
    columns_to_display = [
        'Game Name_user', 'Bet Name', 'Market Name', 'Sportsbook_user', 'line_user', 'Odds_user',
        'Sportsbook_fanduel', 'line_fanduel', 'Odds_fanduel', 'start_date_fanduel',
        'decimal_odds_user', 'decimal_odds_fanduel', 'midpoint_odds', 'true_prob_midpoint', 'EV_user'
    ]
    filtered_df = filtered_df[columns_to_display]
    
    return filtered_df
def fetch_fanduel_markets(api_key, sport, league):
    """
    Fetches markets specifically for FanDuel for a given sport and league.
    
    Parameters:
    - api_key (str): Your API key for authentication.
    - sport (str): The sport (e.g., 'basketball').
    - league (str): The league (e.g., 'NBA').

    Returns:
    - DataFrame: DataFrame containing markets available for FanDuel.
    """
    url = "https://api.opticodds.com/api/v3/markets"
    params = {
        'sport': sport,
        'league': league,
        'key': api_key,
        'sportsbook': ['FanDuel']
    }
    headers = {'Authorization': f'Bearer {api_key}'}
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        
        if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
            fanduel_markets_df = pd.DataFrame(data['data'])
            
            # Ensure the 'name' column is present
            if 'name' in fanduel_markets_df.columns:
                return fanduel_markets_df
            else:
                print("Warning: 'name' column is missing in FanDuel markets data.")
                return pd.DataFrame()
        else:
            print("Unexpected structure in FanDuel API response:", data)
            return pd.DataFrame()
    else:
        print(f"Failed to fetch FanDuel markets: {response.status_code} - {response.text}")
        return pd.DataFrame()


def adjust_for_vig_circa(implied_prob):
    """
    Adjusts implied probability specifically for Circa's -115/-115 vig.
    
    Parameters:
    - implied_prob (float): The raw implied probability from Circa's odds.
    
    Returns:
    - float: Adjusted probability removing the vig.
    """
    total_implied_prob = implied_probability(1.91) + implied_probability(1.91)  # 1.87 is the decimal odds for -115
    return implied_prob / total_implied_prob

def find_plus_ev_bets_circa(df, sportsbook, odds_threshold=10, threshold=0):
    """ 
    Finds +EV bets using Circa Sports as the reference for 'sharp' odds, adjusted for their fixed vig.
    
    Parameters:
    - df (DataFrame): DataFrame containing odds data from multiple sportsbooks.
    - sportsbook (str): The sportsbook to compare against Circa Sports.
    - odds_threshold (float): Maximum decimal odds to consider for bets.
    - threshold (float): Minimum EV threshold for bets to qualify as +EV.
    
    Returns:
    - DataFrame: DataFrame with bets that have positive EV when compared with Circa Sports.
    """
    
    # Filter DataFrames for the specified sportsbook and Circa Sports
    user_sportsbook_df = df[df['Sportsbook'] == sportsbook].rename(columns={'Odds':'Odds_user'})
    circa_df = df[df['Sportsbook'] == 'Circa Sports'].rename(columns={'Odds': 'Odds_circa'})
    
    # Merge user sportsbook odds with Circa odds
    merged_df = pd.merge(user_sportsbook_df, circa_df, on=['Game ID', 'Bet Name', 'Market Name'], suffixes=('_user', '_circa'))
    
    # Convert odds to decimal format
    merged_df['decimal_odds_user'] = merged_df['Odds_user'].apply(american_to_decimal)
    merged_df['decimal_odds_circa'] = merged_df['Odds_circa'].apply(american_to_decimal)
    
    # Calculate implied probabilities
    merged_df['implied_prob_circa'] = merged_df['decimal_odds_circa'].apply(implied_probability)
    
    # Adjust Circa’s implied probability using the specific vig adjustment for Circa
    merged_df['true_prob_circa'] = merged_df['implied_prob_circa'].apply(adjust_for_vig_circa)
    
    # Filter out odds exceeding the threshold
    merged_df = merged_df[merged_df['decimal_odds_user'] <= odds_threshold]
    
    # Calculate EV for user-specified sportsbook using Circa as the sharp sportsbook
    merged_df['EV_user'] = merged_df.apply(lambda row: calculate_ev(row['true_prob_circa'], row['Odds_user']), axis=1)
    
    # Filter for positive EV bets above the threshold
    positive_ev_bets = merged_df[merged_df['EV_user'] > threshold]
    
    return positive_ev_bets

def fetch_game_data_circa(game_ids, api_key, user_sportsbook, market_type='game', sport='baseball', league='MLB', include_player_name=True, is_live='false'):
    # Set Circa Sports and user-specified sportsbook
    sportsbooks = ['Circa Sports', user_sportsbook]
    markets_df = fetch_sports_markets(api_key, sport, league, sportsbook=['Circa Sports'])
    
    if market_type == 'player':
        markets = markets_df['name'].tolist()
    elif market_type == 'game':
        markets = markets_df['name'].tolist()
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
                'market_name': markets,
            }
            if is_live == 'true':
                params['status'] = 'live'
            
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