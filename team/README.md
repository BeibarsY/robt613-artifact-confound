# Team folder — what is where / Что где лежит

## Start here / С чего начать

1. Read the overview / прочитай обзор: `00_OVERVIEW_EN.md` (EN) · `00_ОБЩЕЕ_для_всех.md` (RU)
2. Read your personal file / прочитай свой файл (table below / таблица ниже)
3. Rehearse the talk / репетируй выступление: `SCRIPT_defense_EN.md` (we speak this / выступаем по нему) · `СЦЕНАРИЙ_защиты.md` (RU translation / перевод)

## Files in this folder / Файлы в этой папке

| Person | Role | English | Русский |
| --- | --- | --- | --- |
| Everyone | Project overview, glossary, common questions | `00_OVERVIEW_EN.md` | `00_ОБЩЕЕ_для_всех.md` |
| Chidiadi Bethel Mba | Data and reproducibility (stage 1, Colab) | `Bethel_EN.md` | `Bethel.md` |
| Sungkar Bolat | Signal processing (stages 2–3) | `Sungkar_EN.md` | `Sungkar.md` |
| Beibars Ybraiakhyn | Decoding (stages 5–6), presents stage 4 | `Beibars_EN.md` | `Beibars.md` |
| Ramazan Zypayev | Evaluation and reporting (stages 4, 7, report, slides) | `Ramazan_EN.md` | `Ramazan.md` |
| Everyone | Full 12-minute talk script | `SCRIPT_defense_EN.md` | `СЦЕНАРИЙ_защиты.md` |

## Talk order / Порядок выступления (12 min)

| Speaker | Slides | Time |
| --- | --- | --- |
| Bethel | 1–3: title, question and hypotheses, data audit | 3:05 |
| Sungkar | 4–5: filtering, artifact cleaning | 2:35 |
| Beibars | 6–7: eye signal before any classifier, features and evaluation | 2:50 |
| Ramazan | 8–11: results, exploratory analysis, limitations, conclusion | 2:35 |

## Elsewhere in the project / Остальное в проекте

| What | Where | Note |
| --- | --- | --- |
| **Slides / слайды** | `slide_pdf/slides/slides.tex` | LaTeX source. No PDF yet: compile in Overleaf together with `slide_pdf/report/` (Ramazan's task 5) |
| **Report / отчёт** | `slide_pdf/report/main.tex`, `slide_pdf/report/refs.bib`, `slide_pdf/report/numbers.tex`, `slide_pdf/report/figures/` | Compile in Overleaf. Numbers come from `numbers.tex`, never typed by hand |
| **Notebook for submission / ноутбук** | `notebooks/final.ipynb` | Run on Colab (Bethel's task 3) |
| **All figures / все рисунки** | `results/figures/` | `fig_3...` = stage 3, etc. |
| **All numbers / все числа** | `results/tables/` | |
| **Every decision and why / все решения** | `NOTES.md` | Best file for preparing answers |
| **Original plan / исходный план** | `PLAN.md` | |
| **Who did what / кто что сделал** | `CONTRIBUTIONS.md` | Add a dated line after any work or review |
| **Dataset paper / статья о датасете** | https://www.bbci.de/competition/iv/desc_2a.pdf | Not stored in the repo (third-party PDF) / не хранится в репозитории |
