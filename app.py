from flask import Flask, request, render_template
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
        
        # Load API key from file
        api_key = load_api_key('api_key.txt')
        
        # Fetch the EV bets
        final_ev_df = get_ev_bets(api_key, sport, league)
        
        # Connect to SQLite database and save the data
        conn = sqlite3.connect('betting_data.db')
        save_to_sql(final_ev_df, league, conn)
        conn.close()
        fdf = (final_ev_df[['Game ID','Game Name_caesars','Bet Name','Market Name','line_caesars','Odds_caesars','Odds_pinnacle','EV_caesars']]).sort_values('EV_caesars',ascending=False)

        fdf = fdf[abs(fdf['Odds_caesars'] < 1000)]
        # Convert DataFrame to HTML table for display
        table_html = fdf.to_html(index=False, classes='table table-striped')
        return render_template('index.html', table_html=table_html)
    else:
        # When the method is GET, just show the form without the table
        return render_template('index.html', table_html=None)

if __name__ == "__main__":
    app.run(debug=True)





