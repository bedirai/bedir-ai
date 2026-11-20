// Run a live Bedir debate from Node.js 18+ (or a browser) and re-check every quote's hash.
//
//   node examples/javascript/debate.mjs examples/documents/en-commercial-lease.txt \
//     "Can the tenant end the lease early without paying the penalty?"

import { readFile } from "node:fs/promises";

const [file, question, lang = "en"] = process.argv.slice(2);
if (!file || !question) {
  console.error('usage: node debate.mjs <document.txt> "<yes-or-no question>" [en|tr|de]');
  process.exit(1);
}
const document = (await readFile(file, "utf8")).trim();

const sha256 = async (text) => {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
};

// A verified claim must point at an exact span of the document whose hash matches.
const recheck = async (claim) =>
  !claim.verified || (document.slice(claim.start, claim.end) === claim.quote && (await sha256(claim.quote)) === claim.sha256);

const res = await fetch("https://bedirsavasi.com/api/debate", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ lang, question, document }),
});
if (!res.ok) {
  console.error(`HTTP ${res.status}: ${await res.text()}`);
  process.exit(2);
}

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";
let failures = 0;
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  let nl;
  while ((nl = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, nl).trim();
    buffer = buffer.slice(nl + 1);
    if (!line) continue;
    const event = JSON.parse(line);
    if (event.type === "sides") {
      for (const key of ["for", "against"]) {
        console.log(`\n${key.toUpperCase()}: ${event[key].thesis}`);
        for (const claim of event[key].claims) {
          const ok = await recheck(claim);
          failures += ok ? 0 : 1;
          console.log(`  ${claim.verified ? "✓" : "✕"} ${claim.text}\n    “${claim.quote}”${ok ? "" : "  <- local check failed"}`);
        }
      }
    } else if (event.type === "rebuttals") {
      for (const key of ["for", "against"]) {
        const reb = event[key];
        if (!reb) continue;
        const ok = await recheck(reb);
        failures += ok ? 0 : 1;
        console.log(`\n${key.toUpperCase()} rebuts: ${reb.text}`);
      }
    } else if (event.type === "verdict") {
      console.log(`\nMizan: FOR ${event.p_for}% · AGAINST ${100 - event.p_for}%\n${event.verdict}`);
    } else if (event.type === "done") {
      console.log(`\n${event.verified} of ${event.total} quotes verified by the server; ${failures} local check failures.`);
    } else if (event.type === "error") {
      console.error(`debate failed: ${event.code}`);
      process.exit(2);
    }
  }
}
process.exit(failures ? 1 : 0);
