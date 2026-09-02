import re
import pathlib
import time
import requests

RAW = pathlib.Path(__file__).resolve().parent / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BOOKS = {
    11: "alice_in_wonderland",
    16: "peter_pan",
    55: "wizard_of_oz",
    120: "treasure_island",
    236: "jungle_book",
    2591: "grimms_fairy_tales",
    1597: "andersens_fairy_tales",
    2781: "just_so_stories",
    17396: "secret_garden",
    289: "wind_in_the_willows",
    45: "anne_of_green_gables",
    1998: "twenty_thousand_leagues",
    76: "huckleberry_finn",
    9040: "blue_fairy_book",
    # --- expansion: classic prose ---
    84: "frankenstein",
    1342: "pride_and_prejudice",
    98: "tale_of_two_cities",
    1661: "sherlock_holmes_adventures",
    74: "tom_sawyer",
    43: "dracula",
    768: "wuthering_heights",
    158: "emma",
    514: "little_women",
    5200: "the_metamorphosis",
    2542: "the_odyssey",
    103: "around_the_world_in_eighty_days",
    1400: "great_expectations",
    730: "oliver_twist",
    1260: "jane_eyre",
    17989: "count_of_monte_cristo",
    996: "don_quixote",
    20781: "heidi",
    # --- expansion: dialogue-heavy plays ---
    844: "importance_of_being_earnest",
    3825: "pygmalion",
    1513: "romeo_and_juliet",
    1519: "much_ado_about_nothing",
}

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
