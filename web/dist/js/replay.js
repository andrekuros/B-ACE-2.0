// Replay tab: open episode files or server recordings.
import { $, toast } from "./util.js";
import { parseEpisode } from "./episode.js";

export class ReplayTab {
  constructor(onLoad) {
    this.onLoad = onLoad;
    const drop = $("drop");
    $("file-input").onchange = (e) => this._file(e.target.files[0]);
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); this._file(e.dataTransfer.files[0]); });
    $("runs-refresh").onclick = () => this.refresh();
  }
  show() { if (!this._loaded) { this._loaded = true; this.refresh(); } }

  async _file(file) {
    if (!file) return;
    try { this.onLoad(parseEpisode(await file.text())); } catch (e) { toast(`Could not open ${file.name}: ${e.message}`); }
  }

  async refresh() {
    try {
      const res = await fetch("api/runs");
      if (!res.ok) throw new Error(String(res.status));
      const runs = await res.json();
      runs.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
      $("runs-status").textContent = runs.length ? `${runs.length} recordings in the server’s runs folder.` : "The server’s runs folder is empty.";
      $("run-list").innerHTML = runs.map((r) => `<li data-id="${r.run_id}"><span>${new Date(r.created_at).toLocaleString()}</span><span>${r.config?.blue?.num_agents ?? "?"}v${r.config?.red?.num_agents ?? "?"}, ${r.config?.blue?.behavior} vs ${r.config?.red?.behavior}</span><span>${r.end.replaceAll("_", " ")}, ${r.total_steps} decisions</span></li>`).join("");
      $("run-list").querySelectorAll("li").forEach((li) => (li.onclick = () => this._remote(li.dataset.id)));
    } catch {
      $("runs-status").textContent = "No local server here. Run `cargo run -p bace-server` to browse recordings from Python and the server.";
      $("run-list").innerHTML = "";
    }
  }

  async _remote(id) {
    try {
      const res = await fetch(`api/runs/${id}`);
      if (!res.ok) throw new Error(await res.text());
      this.onLoad(parseEpisode(await res.json()));
    } catch (e) { toast(`Could not load recording: ${e.message}`); }
  }
}
