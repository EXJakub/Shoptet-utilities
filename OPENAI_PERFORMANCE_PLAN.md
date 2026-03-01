# OpenAI performance plan (CZ→SK translator)

Aktuální stav (~50 překladů/min) je nejspíš limitovaný hlavně počtem HTTP requestů na OpenAI na 1 buňku.

## Největší bottlenecky v kódu

1. `app.py` volá `translate_cell(...)` po jedné buňce v cyklu `_process_batch`, takže i pro krátké texty často vzniká 1 API call na buňku.
2. V `translator/openai_provider.py` je režim bez batch API realizovaný jako paralelní jednotlivé requesty (`_translate_parallel_async`), což je robustní, ale má vyšší overhead než skutečné dávkování.
3. Nízké defaulty v UI (`per_run_cells`, batch vypnutý) zpomalují throughput při běžném použití.

## Co jsem změnil teď (quick wins)

- Zapnul jsem **batch API jako default** v UI.
- Zvýšil jsem default `per_run_cells` z 40 na 120.
- Zvýšil jsem default `max_parallel_requests` z 8 na 16 (relevantní při vypnutém batch režimu).

Tyto změny jsou bezpečné, okamžité a měly by zvýšit propustnost bez zásahu do validační logiky HTML.

## Doporučené další kroky (priorita)

### P1: Mikro-batching napříč více buňkami v `_process_batch`

- Před překladem si pro dávku (`per_run_cells`) připravit všechny textové segmenty.
- Udělat lookup do cache pro všechny segmenty.
- Misses poslat v několika větších requestech (`translate_texts` nad listem segmentů), ne po buňkách.
- Výsledek složit zpět do buněk.

**Dopad:** typicky 2–5× méně requestů, často největší boost.

### P1: Deduplikace misses před voláním OpenAI

- Pokud se stejný segment v dávce opakuje (typicky parametrické popisy), překládat ho jen jednou.

**Dopad:** méně tokenů i requestů, levnější i rychlejší běh.

### P2: Adaptivní velikost batchů

- Dynamicky volit počet segmentů v requestu podle součtu znaků/tokenů.
- Např. cíl ~5k–15k znaků/request, aby se minimalizoval overhead, ale bez timeoutů.

### P2: Persistovat provider mezi reruny Streamlitu

- Využít `st.session_state` pro provider instance (místo častého vytváření klienta v `_process_batch`).

**Dopad:** menší overhead, hlavně při dlouhých bězích.

### P3: Oddělit HTML a plain-text pipeline

- Plain text překládat maximálně dávkově.
- HTML buňky nechat ve stávající bezpečné validaci, ale i tam dávkovat textové nody po více buňkách.

---

## Jak měřit zlepšení

Doporučené metriky při stejném datasetu:

- cells/min
- OpenAI requests/min
- average chars/request
- cache hit rate
- chybovost validací (href/structure)

Cíl: posun z ~50 cells/min alespoň na 120–200 cells/min na stejném modelu.
