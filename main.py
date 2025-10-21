#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:02:57 2025

@author: aaronbrace
"""

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
import os
import pandas as pd
from dotenv import dotenv_values
import openai
from pinecone import Pinecone

# ---- Load environment variables ----
if os.getenv("RAILWAY_ENVIRONMENT"):
    # Railway (production)
    openai.api_key = os.getenv("OPENAPIKEY")
    pineconeKey = os.getenv('PINECONE')
    
    
else:
    # Local dev
    config = dotenv_values("/Users/aaronbrace/Desktop/interestApi/.env")
    openai.api_key = config['OPENAIKEY']
    pineconeKey = config['PINECONE']
    
    

# ---- Init Pinecone ----
pc = Pinecone(api_key=pineconeKey)
index_name = "interests"
index = pc.Index(index_name)

# ---- FastAPI app ----
app = FastAPI(title="Meta Semantic Interest Streamer API", version="2.0.0")


# ---- Middleware: block direct Railway calls ----
@app.middleware("http")
async def block_direct_railway(request: Request, call_next):
    app_env = os.getenv("APP_ENV", "development")
    host = request.headers.get("host", "")

    if app_env == "production" and "up.railway.app" in host:
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "code": 403,
                    "message": "Direct Railway access is forbidden. Please use the RapidAPI endpoint."
                }
            },
        )

    return await call_next(request)


# ---- Semantic Search Function ----
def semantic_search(query_text: str, top_k: int = 10, threshold: float = 0.35):
    try:
        # Step 1: Use GPT to generate normalized key terms
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Provide a simple and succinct set of key terms of what the following term "
                        f"refers to in a comma separated list, as if this was an interest to target audiences "
                        f"with on Facebook for ad targeting: {query_text}. "
                        f"This is not a marketing pitch, but a factual set of keywords"
                    )
                }
            ],
        )

        descriptionK = completion.choices[0].message.content.strip()

        # Step 2: Generate embedding for query
        resp = openai.embeddings.create(
            input=[descriptionK],
            model="text-embedding-3-small"
        )
        query_vec = resp.data[0].embedding

        # Step 3: Query Pinecone index
        res = index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True
        )

        # Step 4: Format & filter
        results = [
            {
                "id": match["id"],
                "attribute": match["metadata"].get("attribute"),
                "similarity": match["score"]
            }
            for match in res["matches"]
            if match["score"] >= threshold
        ]

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search error: {str(e)}")


# ---- Endpoint: semantic search interests ----
@app.get("/interests")
def search_interests(q: str = Query(..., description="Search term")):
    """
    Example: GET /interests?q=fitness
    Returns semantically matched interests from Pinecone.
    """
    return semantic_search(q, top_k=50)
