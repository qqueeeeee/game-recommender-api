from fastapi import FastAPI, Query, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df_games = pd.read_csv("steam.csv")

GENRE_COL = "genres"    
TAG_COL = "categories" 
NAME_COL = "name"
APPID_COL = "appid"
REVIEW_COL = "positive_ratings"  
POPULARITY_COL = "owners"

for col in [GENRE_COL, TAG_COL]:
    df_games[col] = df_games[col].fillna("").str.lower()

def parse_owners(raw):
    if pd.isna(raw): return 0
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            val = raw.split('-')[0].strip().replace(',', '')
            return int(val)
        except Exception:
            return 0
    return 0

df_games["owners_num"] = df_games[POPULARITY_COL].apply(parse_owners)

class SteamGame(BaseModel):
    appid: int
    name: str = ""
    playtime_forever: int = 0

class RecRequest(BaseModel):
    games: List[SteamGame]

SKIP_WORDS = ["test", "demo", "beta", "tutorial", "alpha", "soundtrack", "PTB", "benchmark"]
def is_good_game(name):
    name_low = str(name).lower()
    return not any(word in name_low for word in SKIP_WORDS)

@app.post("/recommend")
def recommend(request: RecRequest):
    user_games = pd.DataFrame([g.dict() for g in request.games])
    if user_games.empty:
        return {"recommendations": []}

    user_top = user_games.sort_values("playtime_forever", ascending=False).head(5)
    profile_appids = user_top["appid"].tolist()
    user_games_meta = df_games[df_games[APPID_COL].isin(profile_appids)]
    user_profile_text = (
        " ".join(user_games_meta[GENRE_COL].values) + " " +
        " ".join(user_games_meta[TAG_COL].values)
    )

    candidate_games = df_games[~df_games[APPID_COL].isin(user_games[APPID_COL])]
    candidate_games = candidate_games[candidate_games[NAME_COL].apply(is_good_game)].copy()

    candidate_texts = (
        candidate_games[GENRE_COL].fillna("") + " " + candidate_games[TAG_COL].fillna("")
    ).tolist()
    corpus = [user_profile_text] + candidate_texts

    vec = CountVectorizer(token_pattern=r"[a-zA-Z0-9_\-+]+", min_df=1)
    X = vec.fit_transform(corpus)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    candidate_games["score"] = sims

    top_games = (
        candidate_games
        .sort_values(["score", "owners_num"], ascending=[False, False])
        .head(10)
    )

    result = top_games[[APPID_COL, NAME_COL, GENRE_COL, "score", "owners_num"]].rename(
        columns={"owners_num": "owners"}).to_dict(orient="records")
    return {"recommendations": result}

@app.get("/price")
def get_price(appid: int, cc: str = "US"):
    import logging
    steam_api_url = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc={cc}"
    try:
        resp = requests.get(steam_api_url)
        data = resp.json()
        if not data.get(str(appid)) or not data[str(appid)].get("success"):
            return {"price": None}
        price_obj = data[str(appid)]["data"].get("price_overview")
        if price_obj:
            return {"price": (price_obj["final"] / 100), "currency": price_obj["currency"]}
        if data[str(appid)]["data"].get("is_free"):
            return {"price": 0, "currency": "USD"}
        return {"price": None}
    except Exception as e:
        logging.exception("Error in get_price endpoint")
        return {"price": None, "error": str(e)}

@app.post("/steam")
def get_steam_data(payload: dict):
    steam_id = payload.get("steamId")
    # 1. Get owned games
    games_url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
    games_params = {
        "key": STEAM_API_KEY,
        "steamid": steam_id,
        "include_appinfo": True,
        "include_played_free_games": True,
    }
    games_resp = requests.get(games_url, params=games_params)
    games_data = games_resp.json().get("response", {})

    # 2. Get player summary (username/avatar)
    profile_url = f"https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/"
    profile_params = {
        "key": STEAM_API_KEY,
        "steamids": steam_id,
    }
    profile_resp = requests.get(profile_url, params=profile_params)
    profile_data = profile_resp.json().get("response", {})

    # Format the response
    return {
        "response": {
            "games": games_data.get("games", []),
            "players": profile_data.get("players", []),
        }
    }
