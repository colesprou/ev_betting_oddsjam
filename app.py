from flask import Flask, request, render_template
import os
import pandas as pd
import sqlite3
from betting_functions import get_ev_bets, save_to_sql, load_api_key, fetch_game_data, get_todays_game_ids, find_plus_ev_bets_fanduel_midpoint,fetch_closing_line_game_data,fetch_game_data_circa,find_plus_ev_bets_circa,get_ev_bets_fanduel_pinnacle
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve form data
        sport = request.form['sport']
        league = request.form['league']
        odds_threshold = float(request.form['odds_threshold'])
        sportsbook = request.form['sportsbook']
        is_live = 'true' if 'is_live' in request.form and request.form['is_live'] == 'on' else 'false'  # Handle live games checkbox

        # Load API key from environment variable
        api_key = os.getenv('API_KEY')
        print(api_key)
        
        # Fetch the EV bets
        final_ev_df = get_ev_bets(api_key, sport, league, odds_threshold=odds_threshold, sportsbook=sportsbook, is_live=is_live)
        print(final_ev_df.columns)
        
        # Filter positive EV A and EV B
        ev_a_df = final_ev_df[
            final_ev_df['EV A'] > 0
        ][[
            'Game ID', 'Game Name', 'Market Name', 'Bet Name A', 'line', 
            'Odds A User', 'Odds A Pinnacle', 'EV A'
        ]].rename(columns={
            'Bet Name A': 'Bet Name',
            'Odds A User': 'User Odds',
            'Odds A Pinnacle': 'Pinnacle Odds',
            'EV A': 'EV'
        }).sort_values('EV', ascending=False)

        ev_b_df = final_ev_df[
            final_ev_df['EV B'] > 0
        ][[
            'Game ID', 'Game Name', 'Market Name', 'Bet Name B', 'line', 
            'Odds B User', 'Odds B Pinnacle', 'EV B'
        ]].rename(columns={
            'Bet Name B': 'Bet Name',
            'Odds B User': 'User Odds',
            'Odds B Pinnacle': 'Pinnacle Odds',
            'EV B': 'EV'
        }).sort_values('EV', ascending=False)

        # Convert DataFrames to HTML tables
        table_a_html = ev_a_df.to_html(index=False, classes='table table-striped', table_id="ev_a_table")
        table_b_html = ev_b_df.to_html(index=False, classes='table table-striped', table_id="ev_b_table")

        # Pass both tables to the template
        return render_template('index.html', table_a_html=table_a_html, table_b_html=table_b_html)
    
    else:
        # For GET request, just show the form without tables
        return render_template('index.html', table_a_html=None, table_b_html=None)
@app.route('/fanduel-pinnacle', methods=['GET', 'POST'])
def fanduel_pinnacle():
    if request.method == 'POST':
        # Retrieve form data
        sport = request.form['sport']
        league = request.form['league']
        odds_threshold = float(request.form['odds_threshold'])
        sportsbook = request.form['sportsbook']
        is_live = 'true' if 'is_live' in request.form and request.form['is_live'] == 'on' else 'false'
        
        # Load API key from environment variable
        api_key = os.getenv('API_KEY')
        
        # Fetch the EV bets specific to FanDuel and Pinnacle
        final_ev_df = get_ev_bets_fanduel_pinnacle(api_key, sport, league, odds_threshold=odds_threshold, sportsbook=sportsbook, is_live=is_live)
        
        # Filter DataFrame for display and apply odds threshold
        fdf = final_ev_df[['Game ID', 'Game Name_user', 'Bet Name', 'Market Name', 'line_user', 'Odds_user', 'Odds_pinnacle', 'Odds_fanduel', 'EV_user']].sort_values('EV_user', ascending=False)
        fdf = fdf[abs(fdf['Odds_user']) < odds_threshold]
        
        # Convert DataFrame to HTML table for display
        table_html = fdf.to_html(index=False, classes='table table-striped')
        return render_template('fanduel_pinnacle.html', table_html=table_html)
    
    else:
        # For GET request, just show the form without the table
        return render_template('fanduel_pinnacle.html', table_html=None)

@app.route('/closing-line', methods=['GET', 'POST'])
def closing_line():
    if request.method == 'POST':
        # Retrieve form data
        sport = request.form['sport']
        league = request.form['league']
        odds_threshold = float(request.form['odds_threshold'])
        sportsbook_list = request.form.getlist('sportsbook')
        time_window = int(request.form['time_window'])  # Time window in minutes

        # Load API key from environment variable
        api_key = os.getenv('API_KEY')

        # Get current time in UTC and calculate the end time
        current_time = datetime.now(pytz.UTC)  # Set current_time to UTC timezone
        end_time = current_time + timedelta(minutes=time_window)

        # Fetch today's games and filter based on start time
        game_ids = get_todays_game_ids(api_key, league)
        games_df = fetch_closing_line_game_data(game_ids, api_key, sport=sport, league=league, sportsbooks=sportsbook_list + ['FanDuel'])

        # Check if 'start_date' exists in games_df
        if 'start_date' not in games_df.columns:
            print("Error: 'start_date' column is missing in games data.")
            return render_template('closing_line.html', table_html="Error: Could not retrieve game start dates.")

        # Convert 'start_date' to datetime and ensure it's timezone-aware
        games_df['start_date'] = pd.to_datetime(games_df['start_date'], errors='coerce').dt.tz_convert('UTC')
        
        # Filter games that are starting within the time window
        closing_games_df = games_df[(games_df['start_date'] >= current_time) & (games_df['start_date'] <= end_time)]

        # Calculate +EV bets compared to FanDuel’s closing line using midpoint probability
        ev_bets_fanduel_midpoint = find_plus_ev_bets_fanduel_midpoint(closing_games_df, sportsbook_list, odds_threshold=odds_threshold)

        # Reapply the odds threshold filter if needed
        ev_bets_fanduel_midpoint = ev_bets_fanduel_midpoint[abs(ev_bets_fanduel_midpoint['Odds_user']) < odds_threshold]

        # Select only the desired columns for display
        columns_to_display = [
            'Game Name_user', 'Bet Name', 'Market Name', 'Sportsbook_user', 'line_user', 'Odds_user',
             'line_fanduel', 'Odds_fanduel', 
              'EV_user'
        ]
        ev_bets_fanduel_midpoint = ev_bets_fanduel_midpoint[columns_to_display]
        ev_bets_fanduel_midpoint = ev_bets_fanduel_midpoint.sort_values('EV_user',ascending=False)
        # Convert DataFrame to HTML table for display
        table_html = ev_bets_fanduel_midpoint.to_html(index=False, classes='table table-striped')

        return render_template('closing_line.html', table_html=table_html)
    else:
        # For GET request, just show the form without the table
        return render_template('closing_line.html', table_html=None)
@app.route('/circa', methods=['GET', 'POST'])
def circa_index():
    if request.method == 'POST':
        # Retrieve form data
        sport = request.form['sport']
        league = request.form['league']
        user_sportsbook = request.form['user_sportsbook']
        odds_threshold = float(request.form['odds_threshold'])
        is_live = 'true' if 'is_live' in request.form and request.form['is_live'] == 'on' else 'false'
        
        # Load API key from environment variable
        api_key = os.getenv('API_KEY')
        
        # Get game IDs for today's games
        game_ids = get_todays_game_ids(api_key, league, is_live=is_live)
        print(game_ids)
        # Fetch data for both Circa and the user-specified sportsbook
        combined_df = fetch_game_data_circa(game_ids, api_key, user_sportsbook=user_sportsbook, sport=sport, league=league, is_live=is_live)
        combined_df.to_csv('test_circa.csv')
        # Calculate +EV bets using Circa as the reference for sharp odds
        positive_ev_bets = find_plus_ev_bets_circa(combined_df, user_sportsbook, odds_threshold=odds_threshold, threshold=0)
        print(positive_ev_bets)
        # Filter for display and sort by EV
        display_df = positive_ev_bets[['Game ID', 'Game Name_user', 'Bet Name', 'Market Name', 'line_user', 'Odds_user', 'Odds_circa', 'EV_user']].sort_values('EV_user', ascending=False)
        
        # Convert DataFrame to HTML table
        table_html = display_df.to_html(index=False, classes='table table-striped')
        return render_template('circa_index.html', table_html=table_html)
    
    else:
        # For GET request, just show the form without the table
        return render_template('circa_index.html', table_html=None)

if __name__ == "__main__":
    app.run(debug=True)
