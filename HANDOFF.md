# HANDOFF: CSC 201 Final Project documentation (BOFO)

Student: Jaymar Tablo. Deadline: before Nov 14, 2026. Documentation only, no slides. BOFO app + Supabase database. Exactly 15 pages including the cover. **No colors (black, white, gray only).**

## Status
- `CSC201_Final_Project_BOFO_Documentation.docx` renders at **15 pages** (LibreOffice, Calibri->Carlito), and a pixel check found no colored pixels.
- Source: `handoff_assets/build.js` (docx-js) and `erd.py` (Graphviz ERDs, grayscale). Run `python3 erd.py && node build.js`.
- Page map: 1 cover, 2 TOC, 3 Sec 1, 4 Sec 2, 5 Sec 3, 6 Sec 4.1-4.3, 7-9 landscape ERDs + table catalog, 10 data dictionary, 11 Sec 5.1-5.2, 12 Sec 5.3, 13 Sec 6, 14 Sec 7, 15 Sec 8 + APA 7 references. The TOC and footer ("Page X of 15") use fixed numbers, so keep this map.

## Still to do
1. Fill the placeholders on the cover: `[School / University Name]`, `[College Name]`, `[Professor's Full Name]` (search build.js).
2. Open in Word, check that it is still 15 pages, and check ERD readability (Figure 1 text is small, about 5-6 pt).
3. Optional: add app screenshots to Sec 5.3 only if the page count stays 15.

## Rules and facts
- Real data from Supabase project `zfxldzcoirqolflkkhrh` (Nov 13, 2026): 47 tables, RLS on all, 61 policies, 27 triggers, 12 users, 14 listings, 63 free beds (matches bh_activity). Show totals only; no personal data.
- Needs assessment is based on app requirements and testing, not a survey. Do not invent survey numbers.
- The old pharmacy idea ("Botica Santa Lucia") is replaced by BOFO.
