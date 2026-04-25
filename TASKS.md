# Paint-by-number — taski ulepszeń

## Stan

Zrobione: **T1–T7**. Aktualny `out/` powstał przed T5–T7, więc nie pokazuje ich
efektu — pierwszy krok każdej dalszej oceny: regenerować `out/` z domyślnymi
parametrami i porównać.

Pomiary obecnego `out/` (1600×900 Cybertruck, `k=12`, bez `--max-regions`):

- 869 spójnych regionów; 434 < 100 px, 671 < 300 px.
- Paleta zdominowana przez tło: 11/12 odcieni ciemny granat/szarość, kolor #5
  (RGB 20,15,40) pokrywa 53.6 % obrazu; jedyna ciepła barwa #12 — 0.2 %.
- Numery nieczytelne w obszarach teksturowanych (tłum, refleksy karoserii):
  wiele małych same-color regionów obok siebie, każdy z osobną cyfrą.

Wniosek: trzy największe problemy to **monopol tła w palecie**, **fragmentacja
regionów w teksturach** i **klastry same-color z odrębnymi cyframi**.

---

## P0 — największy zwrot, najpierw

### N1. Saliency-aware quantisation ✅ ZROBIONE
- Pliki: [src/pbn/saliency.py](src/pbn/saliency.py) (nowy),
  [src/pbn/quantize.py](src/pbn/quantize.py),
  [src/pbn/pipeline.py](src/pbn/pipeline.py),
  [src/pbn/cli.py](src/pbn/cli.py).
- `--saliency {none,center,auto}`. `auto` używa
  `cv2.saliency.StaticSaliencyFineGrained` (z `opencv-contrib-python`);
  gdy biblioteki brak — fallback na Sobel-magnitude na kanale L (Lab).
  Wagi clip'owane na progu 0.2 i normalizowane do średniej 1.
- Pomiary na syntetycznym fixture'ze
  [tests/fixtures/dark_scene.png](tests/fixtures/dark_scene.png):
  - `none`: effective_k=3, dominant_color 80.6 %, skin tone obecny (0.3 %).
  - `center`: effective_k=2, skin tone wycięty — twarz w fixture'rze jest
    przy krawędzi kadru, więc gaussian centralny ją penalizuje.
  - `auto`: effective_k=3, skin tone zachowany (Sobel/cv2 daje wagę
    krawędziom).
  Wniosek: dla zdjęć z subjectem ścisłe-centralnie zalecany `center`,
  dla nietypowej kompozycji `auto`.
- Pokryte testami: `test_quantize_accepts_sample_weight_and_shifts_centroids`,
  `test_quantize_rejects_wrongly_shaped_sample_weight`,
  `tests/test_saliency.py` (5 testów),
  `test_saliency_center_shifts_palette_towards_subject`,
  `test_saliency_none_matches_baseline`,
  `test_generate_rejects_unknown_saliency`,
  `test_saliency_flag_records_mode_in_palette_json`,
  `test_saliency_flag_rejects_unknown`. 85 passed (z 73 baseline).

### N2. Kontrola fragmentacji per-kolor ✅ ZROBIONE
- Pliki: [src/pbn/regions.py](src/pbn/regions.py),
  [src/pbn/pipeline.py](src/pbn/pipeline.py),
  [src/pbn/cli.py](src/pbn/cli.py).
- `cap_fragments_per_color` — dla każdego koloru z > N fragmentami bierze
  najmniejszy fragment i przemalowuje go na sąsiada o najdłuższej granicy
  i **innym** kolorze. CLI: `--max-per-color N`.
- Pomiar na noisy fixture'ze (400×600, 8 kolorów, max-regions=300):
  - bez cap: 113 komponentów total, najgorszy kolor 45 fragmentów.
  - cap=25: 81 komponentów total (-28 %), żaden kolor > 25.
  - cap=15: 60 komponentów total (-47 %).
- Pokryte testami: `test_cap_fragments_per_color_caps_count`,
  `test_cap_fragments_per_color_preserves_large_regions`,
  `test_cap_fragments_per_color_noop_when_under_cap`,
  `test_cap_fragments_per_color_rejects_bad_arg`,
  `test_cap_fragments_per_color_deterministic`,
  `test_max_per_color_caps_fragments_in_pipeline`,
  `test_max_per_color_none_is_noop`,
  `test_max_per_color_flag`,
  `test_max_per_color_flag_rejects_zero`. 94 passed (z 85 po N1).

### N3. Przedkwantyzacyjna segmentacja superpikselami (SLIC)
- Pliki: nowy `src/pbn/segment.py`, [src/pbn/pipeline.py](src/pbn/pipeline.py), [src/pbn/cli.py](src/pbn/cli.py).
- Zmiana: opcjonalny krok przed K-means: `skimage.segmentation.slic` (Lab,
  `n_segments` proporcjonalne do rozmiaru obrazu, `compactness ≈ 10`). K-means
  działa potem na **średnich kolorach superpikseli** (ważonych liczbą
  pikseli), a etykieta superpiksela przypisuje cały blok do jednego centroidu.
  CLI: `--presegment {none,slic}`, `--slic-segments N`.
- Akceptacja:
  - Dla obecnego Cybertrucka liczba spójnych regionów spada ≥ 70 % przy
    porównywalnej IoU obwiedni głównego obiektu (truck) względem oryginału.
  - Test `tests/test_pipeline.py::test_slic_presegment_reduces_regions`.

---

## P1 — duża poprawa jakości

### N4. Print-size aware (dawny T12)
- Pliki: [src/pbn/cli.py](src/pbn/cli.py), [src/pbn/pipeline.py](src/pbn/pipeline.py).
- Zmiana: `--print-size {A4,A3,Letter}` + `--dpi`. Wyliczają `scale`,
  `min_region_size` (≥ 4 mm²), grubość linii i minimalny rozmiar cyfry.
  Zostawić surowe `--scale`/`--min-region` jako advanced override.
- Akceptacja:
  - 1600×900 + `--print-size A4 --dpi 300` → template 3508×2480 px,
    `min_region_size` ≥ piksele odpowiadające 4 mm² na druku.
  - Test `tests/test_cli.py::test_print_size_resolves_scale`.

### N5. Tone-mapping pre-quantize dla ciemnych zdjęć
- Pliki: [src/pbn/pipeline.py](src/pbn/pipeline.py).
- Problem: na Cybertrucku 53 % pikseli to "prawie czarne" — K-means dzieli ten
  region na kilka prawie-identycznych centroidów zamiast oddać miejsce
  oświetlonemu obiektowi.
- Zmiana: opcjonalna normalizacja luminancji przed quantize: CLAHE na kanale
  L (Lab) lub gamma-stretch oparty o percentyle (1, 99). CLI: `--tone-map
  {none,clahe,stretch}`.
- Akceptacja:
  - Histogram L po normalizacji ma odchylenie ≥ 1.5× większe niż przed.
  - Na Cybertrucku effective_k bez kolapsu ΔE rośnie (mniej redundantnych
    ciemnych centroidów).
  - Test `tests/test_pipeline.py::test_tone_map_widens_l_histogram`.

### N6. Smoothing konturów regionów
- Pliki: nowy `src/pbn/postprocess.py`, [src/pbn/render.py](src/pbn/render.py).
- Problem: krawędzie po `nearest`-upscale są mocno schodkowe — wizualnie
  brzydkie, gorzej się też maluje.
- Zmiana: morfologiczne `opening` małym kołem (radius 1) na masce *każdego
  regionu osobno* (lub `cv2.findContours` + Douglas-Peucker, rasteryzacja).
  Oddzielnie od dilatacji krawędzi (T7), bo działa na masce labelmap, nie na
  finalnej linii.
- Akceptacja:
  - Suma dłuższych odcinków linii prostej (run-length ≥ 4 px) rośnie ≥ 30 %.
  - IoU regionów względem stanu sprzed N6 ≥ 0.95 (nie zmieniamy kształtów,
    tylko je gładzimy).
  - Test `tests/test_render.py::test_smoothed_boundaries_reduce_jaggedness`.

### N7. Czytelność cyfr na druku
- Pliki: [src/pbn/labels.py](src/pbn/labels.py), [src/pbn/render.py](src/pbn/render.py).
- Problemy widoczne w obecnym `out/template.png`:
  - W gęstych klastrach mikroregionów cyfry nakładają się na siebie i na
    krawędzie sąsiadów.
  - Cyfra czasem stoi na samej krawędzi (przy małym inscribed circle).
- Zmiana:
  - Po umieszczeniu wszystkich placement'ów uruchom kolizyjny check (bbox
    cyfry vs. sąsiednie cyfry/krawędzie) i — w razie kolizji — zmniejsz
    cyfrę lub przełącz region na lead-line.
  - Wprowadź minimalny rozmiar cyfry w **mm na druku** (powiązane z N4),
    nie w pikselach.
- Akceptacja:
  - Test `tests/test_labels.py::test_no_digit_overlap`.
  - Test `tests/test_labels.py::test_digit_min_print_size_mm`.

### N8. Snap granic do krawędzi Canny (dawny T9)
- Pliki: nowy `src/pbn/snap.py`, [src/pbn/pipeline.py](src/pbn/pipeline.py).
- Zmiana: po quantize i merge — dla każdego piksela boundary szukaj najbliższej
  krawędzi Canny w promieniu r (np. 3 px). Jeżeli istnieje, przyciągnij
  etykietę po stronie zgodnej z gradientem.
- Akceptacja:
  - IoU obwiedni regionów vs. Canny rośnie o ≥ 15 p.p. na zestawie testowym.
  - Test `tests/test_snap.py::test_snap_increases_canny_iou`.

---

## P2 — dalsze szlify

### N9. Eksport PDF + arkusz kontaktowy (dawny T10)
- Pliki: nowy `src/pbn/export.py`, [src/pbn/cli.py](src/pbn/cli.py).
- `--pdf out.pdf` (template/preview/legend, A4) + `--contact-sheet`.
- Akceptacja: `tests/test_cli.py` sprawdza, że PDF ma ≥ 3 strony i otwiera się
  przez `pypdf`.

### N10. Ostrzeżenia o mylących kolorach (dawny T11)
- Pliki: [src/pbn/quantize.py](src/pbn/quantize.py) lub `src/pbn/palette.py`.
- W `palette.json`: `"confusable_with": [idx,...]` dla par ΔE < 10.
- Akceptacja: rozszerzony test `tests/test_io.py`.

### N11. Foreground/background split-quantize
- Pliki: [src/pbn/quantize.py](src/pbn/quantize.py), [src/pbn/pipeline.py](src/pbn/pipeline.py).
- Zmiana: jeśli mamy mapę saliency (z N1) lub osobny `--mask`, kwantyzuj
  foreground i background osobno (np. `k_fg = 0.7·k`, `k_bg = 0.3·k`),
  potem złóż obie palety (z dedupem ΔE). Daje gwarancję, że bohater obrazu
  dostanie własne kolory.
- Akceptacja:
  - Na Cybertrucku ≥ 60 % palety to kolory dominujące w foreground.
  - Test `tests/test_quantize.py::test_split_quantize_respects_mask`.

### N12. Zapamiętanie/odtworzenie konfiguracji (`config.json`)
- Pliki: [src/pbn/cli.py](src/pbn/cli.py).
- Zapisuj wszystkie efektywne parametry obok `palette.json` jako
  `config.json` (łącznie z hashem wejścia). Pozwala reprodukować, porównywać
  generacje i pisać A/B.
- Akceptacja: test sprawdza idempotencję `pbn run` z `--config config.json`.

---

## Kolejność wykonania

Sugerowana ścieżka: **N1 → N2 → N3** (rozwiązują trzy największe widoczne
problemy w `out/`), potem N4 + N7 (UX druku), dalej N5/N6/N8 wg potrzeby na
realnym zestawie zdjęć, na końcu N9–N12.

Po N1+N2+N3 koniecznie zregeneruj `out/` z domyślnymi parametrami i przeprowadź
ponownie pomiary z sekcji "Stan" — może się okazać, że N5 lub N8 są już
zbędne.
