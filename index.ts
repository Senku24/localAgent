import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { GoogleGenAI, type Content, type FunctionDeclaration, type Part } from "@google/genai";

const execFileAsync = promisify(execFile);

// Cheapest Gemini text model available to new users (ai.google.dev/pricing, Nov 2026):
const MODEL = "gemini-3.1-flash-lite";
const MAX_TOOL_ROUNDS = 5;

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  throw new Error("GEMINI_API_KEY is missing. Add it to .env");
}

const ai = new GoogleGenAI({ apiKey });

// ---- Tools: declarations the model sees ----
const tools: FunctionDeclaration[] = [
  {
    name: "get_weather",
    description: "Get the current weather for a city.",
    parametersJsonSchema: {
      type: "object",
      properties: {
        city: { type: "string", description: "City name, e.g. 'London'" },
      },
      required: ["city"],
    },
  },
  {
    name: "run_bash",
    description: "Run a bash shell command on the local machine and return stdout/stderr.",
    parametersJsonSchema: {
      type: "object",
      properties: {
        command: { type: "string", description: "The bash command to run" },
      },
      required: ["command"],
    },
  },
];

// ---- Tool implementations ----
async function getWeather(city: string): Promise<unknown> {
  const geoRes = await fetch(
    `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`,
  );
  const geo = (await geoRes.json()) as { results?: { latitude: number; longitude: number; name: string; country: string }[] };
  const place = geo.results?.[0];
  if (!place) return { error: `City not found: ${city}` };

  const wxRes = await fetch(  // gives weather
    `https://api.open-meteo.com/v1/forecast?latitude=${place.latitude}&longitude=${place.longitude}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code&timezone=auto`,
  );
  const wx = (await wxRes.json()) as { current: Record<string, number | string> };
  return { location: `${place.name}, ${place.country}`, ...wx.current };
}

async function runBash(command: string): Promise<unknown> {
  try {
    const { stdout, stderr } = await execFileAsync("bash", ["-c", command], {
      timeout: 15_000,
      maxBuffer: 1024 * 1024,
    });
    return { stdout, stderr };
  } catch (err) {
    const e = err as { stdout?: string; stderr?: string; message: string };
    return { error: e.message, stdout: e.stdout ?? "", stderr: e.stderr ?? "" };
  }
}

async function callTool(name: string, args: Record<string, unknown>): Promise<unknown> {
  switch (name) {
    case "get_weather":
      return getWeather(String(args.city));
    case "run_bash":
      return runBash(String(args.command));
    default:
      return { error: `Unknown tool: ${name}` };
  }
}

// ---- Agent loop: call model, run requested tools, feed results back ----
async function chat(prompt: string): Promise<string> {
  const contents: Content[] = [{ role: "user", parts: [{ text: prompt }] }];

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    const response = await ai.models.generateContent({
      model: MODEL,
      contents,
      config: { tools: [{ functionDeclarations: tools }] },
    });

    const calls = response.functionCalls ?? [];
    if (calls.length === 0) return response.text ?? "";

    // Keep the model's turn (including its function-call parts) in history.
    const modelParts = response.candidates?.[0]?.content?.parts ?? [];
    contents.push({ role: "model", parts: modelParts });

    const resultParts: Part[] = [];
    for (const call of calls) {
      const name = call.name ?? "";
      const args = (call.args ?? {}) as Record<string, unknown>;
      console.error(`[tool] ${name}(${JSON.stringify(args)})`);
      const result = await callTool(name, args);
      resultParts.push({ functionResponse: { name, response: { result } } });
    }
    contents.push({ role: "user", parts: resultParts });
  }

  return "Stopped: too many tool-call rounds.";
}

async function main() {
  const prompt = process.argv.slice(2).join(" ");
  console.log(await chat(prompt));
}

main().catch((err) => {
  console.error("Gemini request failed:", err);
  process.exit(1);
});
