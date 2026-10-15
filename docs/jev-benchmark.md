# Jev vs Haiku — event categorisation benchmark (2026-10-12)

Same 378 events (360 stratified over 12 sources + 18 taxonomy-audit hard cases) through `anthropic/claude-haiku-4.5` (production prompt, batches of 10) and `typesafe/jev-1.13` (one decisions call per event). Disagreements + 17 control agreements adjudicated by `anthropic/claude-fable-5.1`. Spend: Haiku $0.309, Jev $0.0582, judge $6.27.

| arm | cat agree w/ db | sub agree w/ db | judge: correct (disagree) | judge: correct (control) | est. overall | $/event | p50 ms | p95 ms | errors |
|---|---|---|---|---|---|---|---|---|---|
| haiku | 67% | 34% | 52% | 94% | 76% | 0.000817 | 610 | 963 | 0 |
| jev | 70% | 34% | 63% | 94% | 80% | 0.000154 | 353 | 524 | 0 |

## Haiku ↔ Jev agreement

Category 82%, subcategory 57%, full pair 57% (n=378).

Confusion (rows Haiku, cols Jev):

| | music | nightlife | culture | food | markets | workshops | meetups | outdoors | family | None |
|---|---|---|---|---|---|---|---|---|---|---|
| **music** | 33 | 4 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| **nightlife** | 6 | 40 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| **culture** | 3 | 0 | 53 | 0 | 0 | 0 | 2 | 1 | 0 | 0 |
| **food** | 0 | 0 | 0 | 35 | 1 | 7 | 0 | 0 | 0 | 0 |
| **markets** | 0 | 0 | 2 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| **workshops** | 1 | 0 | 0 | 0 | 1 | 43 | 0 | 1 | 0 | 0 |
| **meetups** | 0 | 1 | 6 | 0 | 0 | 3 | 43 | 0 | 0 | 0 |
| **outdoors** | 0 | 1 | 4 | 0 | 0 | 0 | 2 | 8 | 1 | 0 |
| **family** | 0 | 0 | 8 | 0 | 2 | 2 | 2 | 0 | 52 | 0 |
| **None** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Judge

164 disagreements, 17 control agreements. Win rate on disagreements: haiku 26%, jev 37%, both 26%, neither 12%.

Judge's correct_category matches the arm's label: haiku 69%, jev 78% (over all judged cases). Control-sample pair accuracy (judge sanity): haiku 94%, jev 94%.

`est. overall = P(agree)·acc_control + P(disagree)·acc_disagree`. The judged set is hard by construction and the judge is an LLM, not ground truth — read these as relative, not absolute.

## Per source (stratified sample)

| source | n | cat agree h↔j | h vs db | j vs db | disagreements | judge: haiku | judge: jev |
|---|---|---|---|---|---|---|---|
| kulturdaten | 30 | 80% | 73% | 67% | 9 | 56% | 44% |
| rausgegangen | 30 | 77% | 63% | 73% | 12 | 42% | 100% |
| themakery | 30 | 77% | 80% | 97% | 16 | 62% | 75% |
| berlinmitkind | 30 | 93% | 93% | 93% | 10 | 90% | 70% |
| resident_advisor | 30 | 73% | 83% | 73% | 16 | 44% | 88% |
| planetarium | 30 | 63% | 53% | 63% | 20 | 5% | 65% |
| luma | 30 | 97% | 77% | 80% | 11 | 55% | 82% |
| eventbrite | 30 | 97% | 10% | 10% | 9 | 44% | 89% |
| wochenmaerkte | 30 | 100% | 100% | 100% | 0 | — | — |
| tip_berlin | 30 | 73% | 77% | 70% | 12 | 58% | 42% |
| discovery_agent | 30 | 90% | 87% | 77% | 21 | 81% | 19% |
| berlin_de | 30 | 63% | 27% | 50% | 22 | 46% | 50% |

## Hard cases (taxonomy-audit flags)

n=18. Category agreement h↔j 89%; with the audit's suggested label: haiku 50%, jev 39%. Judge on disagreements (6): haiku 67%, jev 67%.

## Jev calibration (category confidence)

| confidence | n | judged | correct vs judge | agrees w/ Haiku |
|---|---|---|---|---|
| [0.0, 0.5) | 41 | 30 | 43% | 51% |
| [0.5, 0.7) | 38 | 29 | 55% | 58% |
| [0.7, 0.9) | 50 | 35 | 66% | 72% |
| [0.9, 1.0) | 249 | 87 | 77% | 93% |

## Not-a-real-event

Haiku quality_score=0 vs Jev is_real_event=false: both 3, haiku only 0, jev only 81, neither 294.

## Option-order sensitivity (Jev)

Category options reversed on 100 events: 5 flips (5%), mean |Δconfidence| 0.0629, identical probability vectors 31/100.
- meetups (0.45) → culture (0.66)  `12464c66`
- family (0.51) → culture (0.55)  `2ce2596a`
- meetups (0.68) → culture (0.44)  `ca59b51e`
- music (0.58) → culture (0.54)  `d4ef312a`
- family (0.44) → culture (0.67)  `daa1c312`

## Known gaps

- The Jev arm produces no tags; production needs a tag strategy before any swap.
- Both arms received source_tags; production Haiku also sees existing_category/tags (cleared here).
- The judge is an LLM, not human ground truth; the control sample measures its agreement with consensus labels, not absolute accuracy.
- Judged accuracy is over disagreements (hard by construction); estimated_overall_acc re-weights by the observed agreement rate.
- Haiku latency is batch latency / batch size; both arms measured sequentially.
- Single run at temperature 0 — a point estimate.

## Example disagreements

| title | source | haiku | jev | judge | reason |
|---|---|---|---|---|---|
| Open Mike | berlin_de | category='nightlife', subcategory='bar-event' | category='nightlife', subcategory='comedy' | neither | The two-day 'Open Mike' in November listed on berlin.de is the annual Haus für Poesie literature competition for young German-language authors (readings of prose and poetry), not a comedy or bar open-mic night, so it belongs under culture/reading. |
| Historischer Weihnachtsmarkt auf dem RAW-Gelände | berlin_de | category='food', subcategory='food-market' | category='markets', subcategory='craft-market' | jev | A Weihnachtsmarkt is a shopping-focused market of stalls (a historical one especially features artisan/handicraft vendors), so it falls under markets/craft-market rather than food, since food and Glühwein are secondary rather than the event itself. |
| Provinziale: Internationales Filmfest Eberswalde | berlin_de | category='culture', subcategory='cinema' | category='culture', subcategory='festival' | both | An international film festival is a culture event that fits both the 'festival' (week-long Filmfest) and 'cinema' (film screenings) subcategories, with 'festival' being the slightly better fit for a multi-day Filmfest. |
| Christmas Garden Berlin | berlin_de | category='family', subcategory='family-event' | category='markets', subcategory=None | neither | Christmas Garden Berlin is an illuminated light-art trail (at the Botanischer Garten), not a shopping market and not explicitly a kids/family event per Rule 12, so it best fits culture/exhibition as an outdoor light installation show. |
| Weihnachts-Rummel an der Landsberger Allee | berlin_de | category='family', subcategory='family-event' | category='markets', subcategory=None | neither | A 'Weihnachts-Rummel' is a Christmas funfair/Volksfest with rides and attractions, not a shopping market and not explicitly a kids/family event per rule 12, so it fits best as culture → festival. |
| Film Restored - Filmerbe-Festival | berlin_de | category='culture', subcategory='cinema' | category='culture', subcategory='festival' | both | Film Restored is a multi-day film heritage festival of restored film screenings, so 'culture' is clearly right and both 'cinema' (the primary activity is watching films) and 'festival' (the event is explicitly a multi-day Filmerbe-Festival) are defensible subcategories, with 'festival' matching the title most directly. |
| IFA 2027 | berlin_de | category='meetups', subcategory='tech-startup' | category='culture', subcategory='festival' | haiku | IFA is Berlin's international consumer-electronics/technology trade fair, which maps best to the tech/startup industry-event bucket under meetups, not to an arts/culture festival. |
| Berliner Märchentage | berlin_de | category='family', subcategory='kids-program' | category='culture', subcategory='festival' | jev | Berliner Märchentage is a multi-week, city-wide fairy-tale festival of readings and performances for all ages; the listing contains none of the explicit children's keywords required by rule 12, so it is culture/festival rather than family. |
| Internationales Kurzfilmfestival Berlin | berlin_de | category='culture', subcategory='cinema' | category='culture', subcategory='festival' | both | A multi-day international short film festival is clearly culture; 'festival' best captures the event format while 'cinema' validly describes the content, so both label pairs are acceptable. |
| Zero Waste Future Festival | berlin_de | category='meetups', subcategory='talk-panel' | category='culture', subcategory='festival' | jev | The only usable signal is the word 'Festival' in the title, which the taxonomy maps to culture/festival, while B's 'talk-panel' assumes a format (panel discussion) that nothing in the sparse data supports. |
| Mary Jane Berlin | berlin_de | category='markets', subcategory='design-market' | category='culture', subcategory=None | jev | Mary Jane Berlin is a multi-day cannabis expo/festival with exhibitors, talks and stages, so 'culture' (ideally subcategory 'festival', null acceptable given the sparse data) fits, whereas 'markets/design-market' mislabels a trade-fair festival as a shopping design market. |
| Re:publica | berlin_de | category='meetups', subcategory='tech-startup' | category='meetups', subcategory=None | haiku | Re:publica is Berlin's well-known conference on digital society, internet and tech, so 'meetups' with the 'tech-startup' subcategory is the most specific correct label (A's null subcategory is not wrong but needlessly omits an obvious fit). |
