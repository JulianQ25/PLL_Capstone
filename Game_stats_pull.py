# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:31:47 2024

@author: jules
"""
#Import libraries
from bs4 import BeautifulSoup
import re
import pandas as pd

# HTML content of the table for Offense
Game_stats = """<table class="css-6jg5m7"><thead><tr><th align="left"></th><th aria-label="Scores" class="">S</th><th aria-label="1Pt Goals" class="">1G</th><th aria-label="2Pt Goals" class="">2G</th><th aria-label="Assists" class="">A</th><th aria-label="Turnovers" class="">TO</th><th aria-label="Shots" class="">Sh</th><th aria-label="Shot %" class="">Sh%</th><th aria-label="2Pt Shots" class="">2Pt Sh</th><th aria-label="2Pt Shot %" class="">2Pt Sh%</th><th aria-label="Shots On Goal" class="">SOG</th><th aria-label="Shots On Goal %" class="">SOG%</th><th aria-label="Time Of Possession" class="">POSS</th><th aria-label="Time Of Possession %" class="">POSS%</th><th aria-label="Touches" class="">TCH</th><th aria-label="Total Passes" class="">PAS</th><th aria-label="Caused Turnovers" class="">CT</th><th aria-label="Groundballs" class="">GB</th><th aria-label="Faceoff %" class="">FO %</th><th aria-label="Faceoffs Won" class="">FOW</th><th aria-label="Faceoffs Lost" class="">FOL</th><th aria-label="Saves" class="">Sv</th><th aria-label="Save %" class="">Sv %</th><th aria-label="Scores Against" class="">SA</th></tr></thead><tbody><tr><td><div class="css-14oqyo1"><div class="default"><div><div color="#DEDEE3" class="css-vl1jdw"><img src="https://premierlacrosseleague.com/wp-content/uploads/2020/02/500x500_PLL_Teams_square_whipsnakes.png" alt="WHP Logo" class="css-8ie5iz"><div class="teamName">Whipsnakes</div></div></div></div></div></td><td>8</td><td>6</td><td>1</td><td>5</td><td>15</td><td>48</td><td>14.6%</td><td>3</td><td>33.3%</td><td>18</td><td>37.5%</td><td>0m 0s</td><td>0%</td><td>0</td><td>0</td><td>5</td><td>21</td><td>55%</td><td>11</td><td>9</td><td>14</td><td>58.3%</td><td>11</td></tr><tr><td><div class="css-14oqyo1"><div class="default"><div><div color="#DEDEE3" class="css-vl1jdw"><img src="https://premierlacrosseleague.com/wp-content/uploads/2020/02/500x500_PLL_Teams_square_archers.png" alt="ARC Logo" class="css-8ie5iz"><div class="teamName">Archers</div></div></div></div></div></td><td>11</td><td>9</td><td>1</td><td>6</td><td>15</td><td>36</td><td>27.8%</td><td>6</td><td>16.7%</td><td>24</td><td>66.7%</td><td>0m 0s</td><td>0%</td><td>0</td><td>0</td><td>0</td><td>28</td><td>45%</td><td>9</td><td>11</td><td>11</td><td>61.1%</td><td>8</td></tr></tbody></table>
"""

# Parse the HTML content
soup = BeautifulSoup(Game_stats, 'html.parser')

# Find the table element
table = soup.find('table', class_='css-6jg5m7')

headers=[header.get_text(strip=True) for header in table.find_all('th')]

data=[]
if table:
    rows=table.find_all('tr')
    for row in rows:
        cells=row.find_all('td')
        row_data=[cell.get_text(strip=True) for cell in cells]
        data.append(row_data)

print(' | '.join(headers))

for row_data in data:
    print(' | '.join(row_data))


#Add Column headers
game_stats=pd.DataFrame(data,columns=headers)
game_stats


#Drop column 0
game_stats=game_stats.drop(0)
game_stats


#Converting percents into float
percent_columns=game_stats.columns[game_stats.columns.str.contains('%')]

#Strip % and convert to numeric
for col in percent_columns:
    game_stats[col]=game_stats[col].str.replace('%','').astype(float)
game_stats

# Adding Win Column
game_stats['Win']= [0,1]

#Adding Game and Season
game_stats['Game']= 10
game_stats['Season']=2019

print(game_stats)

#Appending to current Data set
if 'C:/Users/jules/OneDrive/Documents/A&M Capstone/All_Teams_Stats.csv':
    # Append data to existing CSV file
    game_stats.to_csv("All_Teams_Stats.csv",mode='a', header=False, index=False)
else:
    # Create a new CSV file
    game_stats.to_csv("All_Teams_Stats.csv",index=False)
