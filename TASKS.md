# Paint-by-number — taski ulepszeń

Uporządkowane wg oczekiwanego wpływu na jakość wyniku. Każdy task ma
kryterium akceptacji (testowalne) i wskazanie miejsca w kodzie.

## P0 — największy zwrot, najpierw

### T1. Kwantyzacja w CIELab zamiast RGB
- Plik: [src/pbn/quantize.py](src/pbn/quantize.py)
- Zmiana: konwersja RGB → Lab (`skimage.color.rgb2lab`) przed K-means,
  klastrowanie w Lab, konwersja centroidów z powrotem do RGB.
- Akceptacja:
  - Dla obrazu testowego z dominującym odcieniem (np. obecny Cybertruck)
    paleta ma ≥ 6 z 12 kolorów o ΔE\*ab ≥ 15 od każdego innego.
  - Nowy test `tests/test_quantize.py::test_lab_palette_is_perceptually_diverse`.
- Zależność: `scikit-image` (dodać do `pyproject.toml`).

### T2. Wygładzanie z zachowaniem krawędzi
- Plik: [src/pbn/pipeline.py:50](src/pbn/pipeline.py#L50)
- Zmiana: zastąpić `gaussian_filter` bilateralnym (`skimage.restoration.denoise_bilateral`)
  lub mean-shift (`cv2.pyrMeanShiftFiltering`). Zachować `--blur` jako
  fallback; dodać `--smooth {none,gaussian,bilateral,meanshift}`.
- Akceptacja:
  - Na zdjęciu z teksturą (liście/tłum) liczba regionów spada ≥ 40 %
    przy tym samym `k` i `--min-region`.
  - Krawędzie głównego obiektu (mierzone IoU względem Canny na oryginale)
    zachowane ≥ 90 %.

### T3. Docelowa liczba regionów (`--max-regions`)
- Pliki: [src/pbn/regions.py](src/pbn/regions.py), [src/pbn/cli.py](src/pbn/cli.py), [src/pbn/pipeline.py](src/pbn/pipeline.py)
- Zmiana: iteracyjne scalanie najmniejszych regionów z sąsiadem o
  najdłuższej wspólnej granicy, dopóki `num_regions > max_regions`.
  Priority queue po rozmiarze regionu.
- Akceptacja:
  - `generate(..., max_regions=400)` zwraca `indices` z dokładnie
    ≤ 400 spójnymi komponentami.
  - CLI: `--max-regions N` (domyślnie wyłączone).
  - Test `tests/test_regions.py::test_merge_to_target_count`.

## P1 — duża poprawa jakości

### T4. Morfologiczne czyszczenie mapy etykiet
- Plik: [src/pbn/pipeline.py](src/pbn/pipeline.py) (po kwantyzacji, przed merge)
- Zmiana: majority filter 3×3 na mapie indeksów (per-pixel głosowanie
  najczęstszej etykiety w sąsiedztwie), albo morfologiczne
  opening+closing per klasa.
- Akceptacja: liczba spójnych komponentów spada ≥ 30 % bez widocznej
  zmiany konturów głównego obiektu.

### T5. Wymuszenie minimalnej ΔE w palecie
- Plik: [src/pbn/quantize.py](src/pbn/quantize.py)
- Zmiana: po K-means, jeżeli jakaś para centroidów ma ΔE\*ab < próg
  (domyślnie 7), scal je (waga = liczba pikseli) i uruchom ponownie
  z `k-1`. Powtarzaj aż wszystkie pary ≥ próg lub k == 2.
- Akceptacja:
  - `palette.json` zawiera pole `"effective_k"` i `"min_delta_e"`.
  - Test `tests/test_quantize.py::test_palette_respects_min_delta_e`.

### T6. Adaptacyjny rozmiar cyfry + lead-line dla małych regionów
- Pliki: [src/pbn/labels.py](src/pbn/labels.py), [src/pbn/render.py](src/pbn/render.py)
- Zmiana:
  - Rozmiar cyfry = funkcja `max(distance_transform)` danego regionu.
  - Jeśli wpisane koło < próg (np. 8 px po `scale`), narysuj cyfrę
    poza regionem z cienką linią-wskaźnikiem (lead-line) do centroidu.
- Akceptacja:
  - Test `tests/test_labels.py::test_small_region_uses_leadline`.
  - Wizualnie: żadna cyfra nie nachodzi na linię konturu w
    `template.png`.

### T7. Dilatacja linii szablonu po upskalingu
- Plik: [src/pbn/render.py](src/pbn/render.py) (render_template)
- Zmiana: po nearest-neighbour upscale dylatuj krawędzie 1 px
  (dla `--scale ≥ 4`) lub 2 px (dla `--scale ≥ 6`).
- Akceptacja:
  - Krawędzie w `template.png` mają minimalną szerokość ≥ 2 px
    w dowolnym miejscu.
  - Test `tests/test_render.py::test_template_line_weight`.

## P2 — dalsze szlify

### T8. Saliency-aware quantization
- Plik: [src/pbn/quantize.py](src/pbn/quantize.py)
- Zmiana: opcjonalna mapa wag (domyślnie gaussian ważenie do centrum
  albo `cv2.saliency.StaticSaliencyFineGrained`) przekazywana jako
  `sample_weight` do K-means. CLI: `--saliency {none,center,fine-grained}`.
- Akceptacja: na obrazie z centralnym obiektem ≥ 60 % centroidów
  pochodzi z obszaru saliency ≥ mediana.

### T9. Snap granic regionów do krawędzi Canny
- Plik: [src/pbn/pipeline.py](src/pbn/pipeline.py) lub nowy `src/pbn/snap.py`
- Zmiana: po kwantyzacji, dla każdej granicy regionu szukaj najbliższej
  krawędzi Canny w promieniu r (np. 3 px) i przyciągnij.
- Akceptacja: IoU konturów regionów vs. Canny rośnie o ≥ 15 p.p.
  na zestawie testowym.

### T10. Export PDF + arkusz kontaktowy
- Plik: nowy `src/pbn/export.py`, rozszerzenie [src/pbn/cli.py](src/pbn/cli.py)
- Zmiana: `--pdf out.pdf` generuje A4 PDF z szablonem (strona 1),
  podglądem (strona 2) i legendą (strona 3). Plus `--contact-sheet`
  generujący pojedyncze PNG z wszystkimi trzema obok siebie.
- Akceptacja: CLI smoke test w `tests/test_cli.py` sprawdza że plik
  PDF ma ≥ 3 strony i daje się otworzyć przez `pypdf`.

### T11. Ostrzeżenia o mylących kolorach
- Plik: [src/pbn/quantize.py](src/pbn/quantize.py) lub nowy `src/pbn/palette.py`
- Zmiana: w `palette.json` dodaj dla każdego indeksu pole
  `"confusable_with": [idx,...]` zawierające kolory w ΔE < 10.
- Akceptacja:
  - Struktura palette.json rozszerzona i zweryfikowana w
    `tests/test_io.py`.

### T12. Parametry świadome druku (`--print-size`, `--dpi`)
- Plik: [src/pbn/cli.py](src/pbn/cli.py), [src/pbn/pipeline.py](src/pbn/pipeline.py)
- Zmiana: zamiast ręcznego `--scale` i `--min-region` akceptuj
  `--print-size {A4,A3,Letter}` + `--dpi`. Z nich policz:
  - `scale` = do osiągnięcia docelowego rozmiaru wydruku,
  - `min_region_size` = piksele odpowiadające ≥ 4 mm² na wydruku,
  - grubość linii.
- Akceptacja:
  - Na obrazie 1600×900 + `--print-size A4 --dpi 300` template ma
    dokładnie 3508×2480 px i min. region ≥ obliczony próg.
  - Test `tests/test_cli.py::test_print_size_resolves_scale`.

## Kolejność wykonania

Sugerowana ścieżka: T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10 → T11 → T12.

T1–T3 rozwiązują trzy największe widoczne problemy w obecnym `out/`
(monochromatyczna paleta, za dużo mikroregionów, zanikające cyfry/krawędzie).
Po nich warto ponownie przegenerować `out/` i ocenić, czy T8/T9 są nadal
potrzebne dla danego zestawu zdjęć.
