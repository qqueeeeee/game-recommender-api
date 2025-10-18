import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const json = await req.json();
  const pyApiUrl = "https://game-recommender-api-production-71d1.up.railway.app/recommend";
  const response = await fetch(pyApiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(json),
  });
  const recs = await response.json();
  return NextResponse.json(recs);
}
