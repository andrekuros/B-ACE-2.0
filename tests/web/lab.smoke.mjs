// Headless browser smoke test for the B-ACE Lab.
// Serves web/dist on a free port, loads every tab, runs the self-test, and fails on
// any page error or failed diagnostic. Run: node tests/web/lab.smoke.mjs
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = fileURLToPath(new URL("../../web/dist/", import.meta.url));
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".wasm": "application/wasm", ".json": "application/json" };
const server = createServer(async (req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^([/\\])+/, "") || "index.html";
  if (path.startsWith("..")) return res.writeHead(403).end();
  try {
    const body = await readFile(join(root, path));
    res.writeHead(200, { "content-type": types[extname(path)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((ok) => server.listen(0, ok));
const url = `http://127.0.0.1:${server.address().port}/`;

const errors = [];
const browser = await chromium.launch(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {});
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => { if (m.type() === "error" && !/fonts\.g|cdn\.jsdelivr|favicon/.test(m.text())) errors.push(`console: ${m.text()}`); });
const step = (name) => console.log(`  ok  ${name}`);

try {
  await page.goto(url, { waitUntil: "load" });
  await page.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 30000 });
  step("engine loaded");

  // Live episode runs, draws, and logs a launch.
  await page.selectOption("#preset", "b1_vs_striker");
  await page.locator('#speed [data-speed="400"]').click();
  await page.waitForFunction(() => window.bace.sim.ended && window.bace.sim.cursor === window.bace.sim.frames.length - 1, null, { timeout: 60000 });
  const kinds = await page.evaluate(() => window.bace.sim.frames.flatMap((f) => f.events.map((e) => e.kind)));
  if (!kinds.includes("launch")) throw new Error("no launch event in the striker preset");
  step(`simulate: ${kinds.length} events`);

  // Every preset starts.
  for (const id of await page.$$eval("#preset option", (o) => o.map((x) => x.value).filter((v) => v !== "custom"))) {
    await page.selectOption("#preset", id);
    await page.waitForTimeout(150);
  }
  step("all presets start");

  await page.evaluate(() => window.bace.go("evaluate"));
  await page.fill("#eval-eps", "4");
  await page.click("#eval-run");
  await page.waitForFunction(() => /episodes in/.test(document.getElementById("eval-status").textContent), null, { timeout: 120000 });
  if ((await page.$$("#eval-matrix td[data-k]")).length < 4) throw new Error("evaluation matrix is empty");
  step("evaluate");

  await page.evaluate(() => window.bace.go("wez"));
  await page.fill("#wez-ranges", "10, 30, 50");
  await page.fill("#wez-alts", "25000");
  await page.click("#wez-run");
  await page.waitForFunction(() => /Done in/.test(document.getElementById("wez-status").textContent), null, { timeout: 60000 });
  step("wez");

  await page.evaluate(() => window.bace.go("fsm"));
  await page.fill("#fsm-pop", "4"); await page.fill("#fsm-gen", "2"); await page.fill("#fsm-eps", "1"); await page.fill("#fsm-max", "100");
  await page.click("#fsm-run");
  await page.waitForFunction(() => !document.getElementById("fsm-run").disabled && /\(\d/.test(document.getElementById("fsm-status").textContent), null, { timeout: 60000 });
  step("fsm search");

  await page.evaluate(() => window.bace.go("diagnostics"));
  await page.click("#diag-run");
  await page.waitForFunction(() => document.body.dataset.diagnostics, null, { timeout: 120000 });
  const fails = await page.$$eval("td.res.fail", (els) => els.map((e) => e.closest("tr").querySelector("b").textContent));
  console.log(`  ..  diagnostics: ${await page.textContent("#diag-status")}`);
  if (fails.length) throw new Error(`diagnostics failed: ${fails.join(", ")}`);
  step("diagnostics");

  await page.evaluate(() => window.bace.go("replay"));
  step("replay tab");
  if (errors.length) throw new Error(errors.join("\n"));
  console.log("Lab smoke test passed");
} catch (e) {
  console.error(e.message);
  if (errors.length) console.error(errors.join("\n"));
  process.exitCode = 1;
} finally {
  await browser.close();
  server.close();
}
