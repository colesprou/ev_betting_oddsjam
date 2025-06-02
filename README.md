# Sports Betting Analytics App

This Flask application calculates and displays positive expected value (EV) bets for sports betting, focusing specifically on getting odds from sharp sportsbooks and using a sharp "market" based approach to finding a True Probability. Once the True Prob is calculated we then can automatically pull +EV bets on the sportsbook the user wants to bet on.

## Features

- **Real-time EV Calculation:** Dynamically calculates the expected value of bets
- **API Integration:** Utilizes the OddsJam API for fetching up-to-date odds, ensuring accuracy in calculations.

## Technologies Used

- **Python 3.8+**
- **Flask**: Serves as the web framework for the application.
- **Pandas**: Handles data manipulation and analysis.
- **SQLite3**: Used for storing fetched data for further processing.

## Project Structure
