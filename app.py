from flask import Flask, request, render_template
import os
import pandas as pd
import sqlite3
from betting_functions import get_ev_bets, save_to_sql, load_api_key

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve the form data
        sport = request.form['sport']
        league = request.form['league']
        odds_threshold = float(request.form['odds_threshold'])
        sportsbook = request.form['sportsbook']
        
        # Load API key from file
        api_key = load_api_key('api_key.txt')
        
        # Fetch the EV bets
        final_ev_df = get_ev_bets(api_key, sport, league, odds_threshold=odds_threshold, sportsbook=sportsbook)
        
        # Connect to SQLite database and save the data
        conn = sqlite3.connect('betting_data.db')
        save_to_sql(final_ev_df, league, conn)
        conn.close()
        
        # Filter DataFrame for display and ensure odds threshold is applied
        fdf = final_ev_df[['Game ID','Game Name_user','Bet Name','Market Name','line_user','Odds_user','Odds_pinnacle','Odds_circa','avg_odds','EV_user']].sort_values('EV_user', ascending=False)
        fdf = fdf[abs(fdf['Odds_user'] < odds_threshold)]
        
        # Convert DataFrame to HTML table for display
        table_html = fdf.to_html(index=False, classes='table table-striped')
        return render_template('index.html', table_html=table_html)
    
    else:
        # When the method is GET, just show the form without the table
        return render_template('index.html', table_html=None)

if __name__ == "__main__":
    app.run(debug=True)
