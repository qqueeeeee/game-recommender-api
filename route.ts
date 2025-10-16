import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const json = await req.json();
  const pyApiUrl = "http://localhost:8000/recommend";
  const response = await fetch(pyApiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(json),
  });
  const recs = await response.json();
  return NextResponse.json(recs);
}
