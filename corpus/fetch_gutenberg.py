import re
import pathlib
import time
import requests

RAW = pathlib.Path(__file__).resolve().parent / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BOOKS = {
    # ---- existing classics ----
    11: "alice_in_wonderland", 16: "peter_pan", 55: "wizard_of_oz",
    120: "treasure_island", 236: "jungle_book", 2591: "grimms_fairy_tales",
    1597: "andersens_fairy_tales", 2781: "just_so_stories",
    17396: "secret_garden", 289: "wind_in_the_willows", 45: "anne_of_green_gables",
    1998: "twenty_thousand_leagues", 76: "huckleberry_finn", 9040: "blue_fairy_book",
    84: "frankenstein", 1342: "pride_and_prejudice", 98: "tale_of_two_cities",
    1661: "sherlock_holmes_adventures", 74: "tom_sawyer", 43: "dracula",
    768: "wuthering_heights", 158: "emma", 514: "little_women",
    5200: "the_metamorphosis", 2542: "the_odyssey", 103: "around_the_world_in_eighty_days",
    1400: "great_expectations", 730: "oliver_twist", 1260: "jane_eyre",
    17989: "count_of_monte_cristo", 996: "don_quixote", 20781: "heidi",
    844: "importance_of_being_earnest", 3825: "pygmalion",
    1513: "romeo_and_juliet", 1519: "much_ado_about_nothing",
    # ---- essays & letters ----
    2760: "emerson_essays", 2944: "emerson_self_reliance",
    1184: "lamb_essays_of_elia", 134: "pope_essay_on_man",
    37729: "hazlitt_table_talk", 1170: "bacon_essays",
    # ---- history / biography ----
    582: "federalist_papers", 20203: "paine_common_sense",
    375: "franklin_autobiography", 653: "up_from_slavery_washington",
    4320: "douglass_narrative", 1232: "machiavelli_prince",
    1902: "smiles_self_help", 8448: "jordan_stories_of_great_men",
    29765: "james_the_varieties_of_religious_experience",
    1727: "hume_an_enquiry_concerning_human_understanding",
    3764: "locke_essay_concerning_human_understanding",
    # ---- science / natural history ----
    2009: "darwin_beagle_voyage", 726: "darwin_origin_of_species",
    38617: "faraday_chemical_history_of_a_candle",
    30360: "faraday_various_forces_of_nature",
    1080: "huxley_man_place_in_nature",
    # ---- travel ----
    2910: "twain_roughing_it", 3176: "twain_innocents_abroad",
    4000: "twain_a_tramp_abroad", 2852: "twain_following_the_equator",
    432: "irving_columbus", 9591: "twain_innocents_abroad_2",
    # ---- classic fiction expansion ----
    60: "wells_time_machine", 159: "wells_isle_of_dr_moreau",
    36: "wells_war_of_the_worlds", 5230: "wells_first_men_in_the_moon",
    5920: "wells_invisible_man", 143: "butler_way_of_all_flesh",
    205: "thoreau_walden", 174: "thoreau_civil_disobedience",
    # ---- short stories / anthologies ----
    2846: "twain_jumping_frog", 8438: "henry_gift_of_the_magi",
    19955: "crane_open_boat", 4217: "twain_hadleyburg",
    3177: "harte_luck_of_roaring_camp", 35820: "henry_cop_and_anthem",
    17250: "henry_anthology", 1086: "aesop_fables",
}
# drop any placeholder (empty-name) entries above
BOOKS = {k: v for k, v in BOOKS.items() if isinstance(v, str) and v}

START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", re.I)


def clean(text: str) -> str:
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[:m.start()]
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    total_words = 0
    for gid, name in BOOKS.items():
        out = RAW / f"gutenberg_{name}.txt"
        if out.exists():
            print(f"skip existing {name}")
            continue
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "zeus-corpus/0.1"})
            r.raise_for_status()
            text = clean(r.text)
            out.write_text(text, encoding="utf-8")
            wc = len(text.split())
            total_words += wc
            print(f"{name}: {wc:,} words")
        except Exception as e:
            print(f"FAIL {name} ({url}): {e}")
        time.sleep(1.0)
    print(f"done, total {total_words:,} words")


if __name__ == "__main__":
    main()
