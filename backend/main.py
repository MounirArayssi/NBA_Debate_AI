from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from datetime import datetime
from nba_api.stats.endpoints import leagueleaders, leaguestandings, leaguedashplayerstats
from nba_api.stats.static import players
import time

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DebateRequest(BaseModel):
    topic: str
    messages: list
    tone: str  # "trash", "analyst", "stats", or default

# 🏀 Standings
def fetch_standings():
    context = "Top Teams in 2024–25 Season:\n"
    try:
        standings = leaguestandings.LeagueStandings(season="2024-25")
        df = standings.get_data_frames()[0]
        top = df.sort_values("WinPCT", ascending=False).head(5)
        for _, row in top.iterrows():
            context += f"{row['TeamName']}: {row['WINS']}-{row['LOSSES']} (Win%: {round(row['WinPCT'], 3)})\n"
        return context
    except Exception as e:
        return f"⚠️ Error fetching standings: {e}"

# 📊 Top scorers
def fetch_top_scorers():
    try:
        leaders = leagueleaders.LeagueLeaders(stat_category_abbreviation="PTS", season="2024-25", season_type_all_star="Regular Season")
        df = leaders.get_data_frames()[0].head(5)
        return "Top Scorers:\n" + "\n".join(f"{row['PLAYER']} — {row['PTS']} PPG" for _, row in df.iterrows())
    except Exception as e:
        return f"⚠️ Error fetching scoring leaders: {e}"

# 📊 PTS, AST, REB leaders
def fetch_all_around_stats():
    try:
        df = leaguedashplayerstats.LeagueDashPlayerStats(season="2024-25").get_data_frames()[0]
        top = df.sort_values("PTS", ascending=False).head(5)
        context = "All-Around Leaders:\n"
        for _, row in top.iterrows():
            context += f"{row['PLAYER_NAME']}: {row['PTS']} PPG, {row['AST']} AST, {row['REB']} REB\n"
        return context
    except Exception as e:
        return f"⚠️ Error fetching all-around stats: {e}"

# 🔍 Individual player stats
def get_player_stats(name):
    try:
        match = players.find_players_by_full_name(name)
        if not match:
            return f"⚠️ Player '{name}' not found."
        pid = match[0]['id']
        df = leaguedashplayerstats.LeagueDashPlayerStats(season="2024-25").get_data_frames()[0]
        row = df[df['PLAYER_ID'] == pid]
        if row.empty:
            return f"⚠️ No 2024–25 stats available for {name}."
        row = row.iloc[0]
        return (
            f"{name}'s 2024–25 stats:\n"
            f"{row['PTS']} PPG, {row['AST']} AST, {row['REB']} REB\n"
            f"FG%: {round(row['FG_PCT'] * 100, 1)}%, 3P%: {round(row['FG3_PCT'] * 100, 1)}%, FT%: {round(row['FT_PCT'] * 100, 1)}%\n"
            f"Team: {row['TEAM_ABBREVIATION']}"
        )
    except Exception as e:
        return f"⚠️ Error fetching stats for {name}: {e}"

@app.post("/debate")
async def debate(request: DebateRequest):
    try:
        # Tone
        if request.tone == "trash":
            style = (
                "You're a bold, sarcastic NBA fan who loves trash-talking bad takes. "
                "You never hold back, and your responses are packed with confidence, jokes, and bold claims."
            )
        elif request.tone == "analyst":
            style = (
                "You're a professional NBA analyst. You break down arguments with matchups, team performance, and player efficiency. "
                "Your tone is calm, logical, and grounded in facts."
            )
        elif request.tone == "stats":
            style = (
                "You're a data-driven NBA nerd who relies on numbers, not emotion. You argue with stats like PPG, FG%, and Win Shares."
            )
        else:
            style = (
                "You're a regular NBA fan just debating a friend. You speak casually, but with solid knowledge and opinions."
            )

        # Pull context
        standings = fetch_standings()
        scorers = fetch_top_scorers()
        all_around = fetch_all_around_stats()

        # Try extracting one player from the topic (optional: improve with NLP)
        player_focus = ""
        possible_names = ["Shai Gilgeous-Alexander", "Nikola Jokic", "Joel Embiid", "Jayson Tatum", "Giannis Antetokounmpo", "Anthony Edwards", "LeBron James"]
        for name in possible_names:
            if name.lower() in request.topic.lower():
                player_focus = get_player_stats(name)
                break

        system_prompt = (
            f"{style}\n\n"
            f"The user wants to debate the topic: \"{request.topic}\"\n\n"
            "Use the following real NBA data from the 2024–25 season to help form your arguments and rebuttals:\n\n"
            f"{standings}\n\n{scorers}\n\n{all_around}\n\n{player_focus}\n\n"
            "Reference the topic even if the user only says 'he' or 'they'. Always respond like you're in a real conversation."
        )

        messages = [{"role": "system", "content": system_prompt}] + request.messages

        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        print("🔥 Backend Error:", e)
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
