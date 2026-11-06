"""Make Kokoro's narration sound like a game show: an announcer, and a paper boy.

Kokoro reads text evenly, like an audiobook. A game-show announcer is the
same voice delivered differently: quicker, compressed so every word lands at
the same loudness, a lift in the presence band, and a little room around it.
That part is signal processing, done here on the clip the audio server
returns.

"Extra! Extra!" is a different character altogether: a newsboy shouting on a
street corner. The line's opening call is split off and given to him (a
younger Kokoro voice, sped and pitched up, narrowed and driven like a shout,
with a slap echo off the buildings); the announcer reads the rest.

Stdlib only (``array``, ``wave``, ``math``): the arcade image has no numpy,
and a few seconds of 24 kHz mono is a fraction of a second of pure Python.
"""
from __future__ import annotations

import array
import collections
import io
import math
import os
import re
import wave

#: The newsboy. No Kokoro voice is a boy (every male voice measured 88-166 Hz),
#: and pitching a man up by a third sounded like a machine. A bright young
#: voice (~185 Hz) lifted only a little (~2.4 semitones) stays human.
PAPERBOY_VOICE = os.getenv("PARTY_PAPERBOY_VOICE", "am_eric,af_kore")
PAPERBOY_SPEED = 1.08
#: No pitch shifting: the blend already sits at a teenage ~172 Hz, and every
#: resample above 1.0 was heard as "too high" or synthetic.
PAPERBOY_PITCH = 1.0
PAPERBOY_CALL = "Extra! Extra! Read all about it!"
#: Kokoro's own pace is the most natural; cadence varies it per sentence.
ANNOUNCER_SPEED = 1.08  # by ear: 1.15 (audition round three) was too fast in play
#: Everything but an exclamation sits this far under it, so the shouts land (gain only: nothing added).
SHOUT_LIFT_DB = 2.5

# --------------------------------------------------------------------------- Bible names
#
# Kokoro's phonemizer (misaki, American) guesses a name it has never seen from
# its spelling, and for Bible names it is wrong often enough to be noticed:
# "Barnabas" came out bar-NAH-bus, "Job" as the word "job", "Gethsemane" as
# GETH-sim-ain. Each entry below was measured: run through the audio server's
# own ``KPipeline(lang_code="a").g2p`` without an override, compared with the
# standard American church pronunciation, and kept only when misaki got it
# wrong. Names misaki already says right are listed in ``SAID_RIGHT`` instead.
#
# The IPA is misaki's own inventory, because the override is passed to Kokoro
# verbatim: ``A`` = eɪ, ``I`` = aɪ, ``O`` = oʊ, ``W`` = aʊ, ``Y`` = ɔɪ, ``ʤ`` /
# ``ʧ`` (one symbol each), ``ᵻ`` a reduced ɪ, ``ᵊ`` a syllabic schwa, and the
# stress mark sits right before the stressed vowel. Writing "oʊ" or "dʒ"
# instead still parses, but reaches the model as two symbols it was not
# trained on together.

#: Bible names and terms misaki mispronounces, as misaki IPA. ``pronounce()``
#: emits them as misaki inline markup, ``[word](/ipa/)``, which the audio
#: server's text normaliser leaves alone and the phonemizer honours exactly.
PRONUNCIATIONS: dict[str, str] = {
    # ---- books (the rest of the 66 are said right; see SAID_RIGHT)
    "Job": "ʤˈOb",  # never the word "job": see _not_the_man_job
    "Haggai": "hˈæɡI",
    "Habakkuk": "həbˈækək",
    "Colossians": "kəlˈɑʃənz",
    "Philemon": "fIlˈimən",
    # ---- Genesis to Ruth
    "Lamech": "lˈAmək",
    "Terah": "tˈɪɹə",
    "Keturah": "kətˈʊɹə",
    "Leah": "lˈiə",
    "Laban": "lˈAbən",
    "Tamar": "tˈAmɑɹ",
    "Ephraim": "ˈifɹiəm",
    "Moses": "mˈOzᵻz",
    "Zipporah": "zɪpˈɔɹə",
    "Jochebed": "ʤˈɑkəbˌɛd",
    "Gershom": "ɡˈɜɹʃəm",
    "Eleazar": "ˌɛliˈAzəɹ",
    "Phinehas": "fˈɪniəs",
    "Abihu": "əbˈIhu",
    "Balaam": "bˈAləm",
    "Balak": "bˈAlæk",
    "Caleb": "kˈAləb",
    "Rahab": "ɹˈAhæb",
    "Achan": "ˈAkæn",
    "Othniel": "ˈɑθniəl",
    "Ehud": "ˈihəd",
    "Barak": "bˈɛɹæk",
    "Sisera": "sˈɪsəɹə",
    "Jael": "ʤˈAəl",
    "Abimelech": "əbˈɪməlˌɛk",
    "Jephthah": "ʤˈɛfθə",
    "Manoah": "mənˈOə",
    "Boaz": "bˈOæz",
    "Obed": "ˈObɛd",
    "Gera": "ɡˈɪɹə",
    "Bezalel": "bˈɛzəlˌɛl",
    # ---- Samuel to Esther
    "Elkanah": "ɛlkˈAnə",
    "Peninnah": "pᵻnˈɪnə",
    "Michal": "mˈIkᵊl",
    "Abner": "ˈæbnəɹ",
    "Absalom": "ˈæbsələm",
    "Ahithophel": "əhˈɪθəfˌɛl",
    "Hushai": "hˈuʃI",
    "Shimei": "ʃˈɪmiˌI",
    "Mephibosheth": "məfˈɪbəʃˌɛθ",
    "Adonijah": "ˌædənˈIʤə",
    "Asa": "ˈAsə",
    "Jehoshaphat": "ʤəhˈɑʃəfˌæt",
    "Jehu": "ʤˈihju",
    "Naboth": "nˈAbɑθ",
    "Naaman": "nˈAəmən",
    "Gehazi": "ɡəhˈAzI",
    "Elisha": "ɪlˈIʃə",
    "Ahaz": "ˈAhæz",
    "Jehoiakim": "ʤəhˈYəkɪm",
    "Jehoiachin": "ʤəhˈYəkɪn",
    "Zerubbabel": "zəɹˈʌbəbᵊl",
    "Sanballat": "sænbˈælət",
    "Artaxerxes": "ˌɑɹtəzˈɜɹksiz",
    "Darius": "dəɹˈIəs",
    "Xerxes": "zˈɜɹksiz",
    "Ahasuerus": "əhˌæzjuˈɛɹəs",
    "Haman": "hˈAmən",
    "Belshazzar": "bɛlʃˈæzəɹ",
    "Shadrach": "ʃˈædɹæk",
    "Meshach": "mˈiʃæk",
    "Abednego": "əbˈɛdnᵻɡˌO",
    "Hananiah": "hˌænənˈIə",
    "Mishael": "mˈɪʃAəl",
    "Azariah": "ˌæzəɹˈIə",
    "Ishbosheth": "ɪʃbˈOʃɛθ",
    "Asahel": "ˈæsəhˌɛl",
    "Benaiah": "bᵻnˈAə",
    "Shunammite": "ʃˈunəmˌIt",
    "Shulammite": "ʃˈuləmˌIt",
    "Athaliah": "ˌæθəlˈIə",
    "Joash": "ʤˈOæʃ",
    "Amaziah": "ˌæməzˈIə",
    "Pekah": "pˈikə",
    "Hoshea": "hOʃˈiə",
    "Rabshakeh": "ɹˈæbʃəkə",
    # ---- the gospels and Acts
    "Zacharias": "zˌækəɹˈIəs",
    "Herodias": "həɹˈOdiəs",
    "Archelaus": "ˌɑɹkəlˈAəs",
    "Antipas": "ˈæntɪpəs",
    "Salome": "səlˈOmi",
    "Pilate": "pˈIlət",
    "Pontius": "pˈɑnʃəs",
    "Caiaphas": "kˈAəfəs",
    "Annas": "ˈænəs",
    "Barabbas": "bəɹˈæbəs",
    "Arimathea": "ˌɛɹəməθˈiə",
    "Nathanael": "nəθˈænjəl",
    "Alphaeus": "ælfˈiəs",
    "Cleopas": "klˈiəpəs",
    "Bartimaeus": "bˌɑɹtɪmˈiəs",
    "Jairus": "ʤˈIɹəs",
    "Magdalene": "mˈæɡdəlˌin",
    "Lazarus": "lˈæzəɹəs",
    "Zacchaeus": "zækˈiəs",
    "Sapphira": "səfˈIɹə",
    "Dorcas": "dˈɔɹkəs",
    "Tabitha": "tˈæbɪθə",
    "Barnabas": "bˈɑɹnəbəs",
    "Silas": "sˈIləs",
    "Apollos": "əpˈɑləs",
    "Eutychus": "jˈutɪkəs",
    "Agrippa": "əɡɹˈɪpə",
    "Bernice": "bəɹnˈis",
    "Gamaliel": "ɡəmˈAliəl",
    "Agabus": "ˈæɡəbəs",
    "Damaris": "dˈæməɹɪs",
    "Theophilus": "θiˈɑfələs",
    # ---- the letters
    "Onesimus": "Onˈɛsɪməs",
    "Epaphras": "ˈɛpəfɹəs",
    "Epaphroditus": "ɪpˌæfɹədˈItəs",
    "Tychicus": "tˈɪkɪkəs",
    "Demas": "dˈiməs",
    "Sosthenes": "sˈɑsθəniz",
    "Andronicus": "ˌændɹənˈIkəs",
    "Tertius": "tˈɜɹʃəs",
    "Syntyche": "sˈɪntɪki",
    "Archippus": "ɑɹkˈɪpəs",
    "Diotrephes": "dIˈɑtɹəfˌiz",
    "Immanuel": "ɪmˈænjuəl",
    # ---- places
    "Goshen": "ɡˈOʃən",
    "Carmel": "kˈɑɹməl",
    "Bashan": "bˈAʃən",
    "Jezreel": "ʤˈɛzɹiəl",
    "Gibeon": "ɡˈɪbiən",
    "Elah": "ˈilə",
    "Cherith": "kˈɪɹɪθ",
    "Zarephath": "zˈɛɹəfˌæθ",
    "Shinar": "ʃˈInɑɹ",
    "Endor": "ˈɛndɔɹ",
    "Cush": "kˈʌʃ",
    "Ninevite": "nˈɪnəvˌIt",
    "Ninevites": "nˈɪnəvˌIts",
    "Midianite": "mˈɪdiənˌIt",
    "Midianites": "mˈɪdiənˌIts",
    "Amalekite": "ˈæməlɛkˌIt",
    "Amalekites": "ˈæməlɛkˌIts",
    "Judea": "ʤudˈiə",
    "Sychar": "sˈIkɑɹ",
    "Siloam": "sIlˈOəm",
    "Capernaum": "kəpˈɜɹniəm",
    "Gethsemane": "ɡɛθsˈɛməni",
    "Gennesaret": "ɡənˈɛsəɹˌɛt",
    "Tiberias": "tIbˈɪɹiəs",
    "Gerasenes": "ɡˈɛɹəsˌinz",
    "Bethphage": "bˈɛθfəʤi",
    "Chorazin": "kɔɹˈAzɪn",
    "Emmaus": "ɛmˈAəs",
    "Philippi": "fˈɪləpˌI",
    "Derbe": "dˈɜɹbi",
    "Smyrna": "smˈɜɹnə",
    "Laodicea": "lˌAɑdəsˈiə",
    "Cenchreae": "sˈɛŋkɹiˌi",
    "Achaia": "əkˈAə",
    "Troas": "tɹˈOæs",
    # ---- terms
    "Nazarite": "nˈæzəɹˌIt",
    "Nazarites": "nˈæzəɹˌIts",
    "ephod": "ˈifɑd",
    "Rabboni": "ɹæbˈOnI",
    "Sheol": "ʃˈiOl",
    "Urim": "jˈʊɹɪm",
    "denarii": "dənˈɛɹiˌI",
    "Elul": "ɛlˈul",
    "Nisan": "nisˈɑn",
    "Ephphatha": "ˈɛfəθə",
    "Talitha": "tˈælɪθə",
    "Eloi": "ilˈOI",
    "Sabachthani": "sˌɑbɑkθˈɑni",
    "Mene": "mˈini",
    "Upharsin": "jufˈɑɹsɪn",
}

#: Bible words misaki already says right in mixed case, measured the same way.
#: They need no IPA, but a shouted one does: misaki spells an all-caps word
#: out letter by letter ("NOAH" is N-O-A-H), so ``pronounce()`` gives these
#: their ordinary capitalisation when they arrive in capitals.
_SAID_RIGHT_WORDS = """
    Genesis Exodus Leviticus Numbers Deuteronomy Joshua Judges Ruth Samuel Kings Chronicles Ezra
    Nehemiah Esther Psalms Psalm Proverbs Ecclesiastes Solomon Isaiah Jeremiah Lamentations Ezekiel
    Daniel Hosea Joel Amos Obadiah Jonah Micah Nahum Zephaniah Zechariah Malachi Matthew Mark Luke
    John Acts Romans Corinthians Galatians Ephesians Philippians Thessalonians Timothy Titus Hebrews
    James Peter Jude Revelation
    Adam Eve Cain Abel Seth Enoch Methuselah Noah Shem Japheth Nimrod Babel Abram Abraham Sarai
    Sarah Hagar Ishmael Ishmaelites Lot Melchizedek Isaac Rebekah Esau Jacob Rachel Bilhah Zilpah
    Reuben Simeon Levi Judah Naphtali Asher Issachar Zebulun Joseph Benjamin Benjamite Dinah
    Potiphar Manasseh Aaron Miriam Jethro Nadab Korah Eglon Shamgar Deborah Gideon Samson Delilah
    Naomi Orpah Jesse David Goliath Eli Hannah Saul Kish Jonathan Joab Nathan Abigail Nabal
    Bathsheba Uriah Amnon Ziba Rehoboam Jeroboam Ahab Jezebel Elijah Josiah Hezekiah Uzziah
    Zedekiah Tobiah Cyrus Vashti Mordecai Nebuchadnezzar Gomer Dagon Baal Beelzebub Lucifer Gabriel
    Raphael Mammon Belial
    Mary Elizabeth Herod Simon Andrew Philip Bartholomew Thomas Didymus Thaddaeus Thaddeus Judas
    Iscariot Matthias Zebedee Martha Nicodemus Stephen Ananias Cornelius Barnaby Lydia Priscilla
    Aquila Felix Festus Drusilla Publius Trophimus Erastus Gaius Crispus Phoebe Junia Rhoda Aeneas
    Lois Eunice Euodia Chloe Jesus Christ Emmanuel Yahweh Jehovah Elohim Caesar Augustus Tiberius
    Eden Canaan Horeb Sinai Kadesh Moriah Zion Tabor Hermon Gilead Megiddo Armageddon Samaria Tyre
    Sidon Decapolis Gadarenes Bethany Bethsaida Cana Nain Jericho Jordan Caesarea Antioch Iconium
    Lystra Tarsus Ephesus Pergamum Thyatira Sardis Patmos Areopagus Berea Thessalonica Corinth Crete
    Cyprus Malta Macedonia Galatia Cappadocia Bithynia Cilicia Phoenicia Ethiopia Nineveh Tarshish
    Joppa Babylon Babylonian Assyria Chaldeans Persia Moab Moabites Midian Edom Edomites Ammonites
    Hittites Jebusites Canaanites Philistine Philistines Israel Israelites Jerusalem Bethlehem
    Nazareth Galilee Golgotha Calvary Shiloh Hebron Bethel Beersheba Sodom Gomorrah Gilgal Shechem
    Ararat Damascus Gaza Sheba Euphrates Kidron Jabbok Mizpah Ziklag
    Pharisee Pharisees Sadducee Sadducees Levite Levites Samaritan Samaritans Gentile Gentiles
    Nazarene Nazirite Hebrew Israelite Sanhedrin Magi Pentecost Passover Messiah Hosanna Hallelujah
    Alleluia Amen Rabbi Abba Selah Shalom Hades Leviathan Behemoth Nephilim Thummim Torah Sabbath
    Herodians Maranatha Ebenezer Ichabod Corban Tekel Jubilee
    Zealots Hosannas Corinthian Philippian Ephesian Galilean Nazarenes Levitical Pharaohs Psalmist
    Pharaoh Rizpah Hophni Tirzah Huldah Sennacherib Ittai Zadok Abiathar Abishai Bezaleel Lamb
"""
SAID_RIGHT: frozenset[str] = frozenset(_SAID_RIGHT_WORDS.split()) - frozenset(PRONUNCIATIONS)

# Words the override list must take only exactly as written. "Job" in capitals
# could be an acronym, and the plain word "job" is the common one.
_EXACT_ONLY = frozenset({"Job"})

_POSSESSIVE = r"(?P<poss>['’][sS](?!\w)|(?<=[sSzZ])['’](?=\s+[a-z]))?"


def _forms(word: str) -> set[str]:
    if word in _EXACT_ONLY:
        return {word}
    return {word, word[:1].upper() + word[1:], word.upper()}


_IPA_FOR = {form: PRONUNCIATIONS[w] for w in PRONUNCIATIONS for form in _forms(w)}
_CAPS = {w.upper(): w for w in SAID_RIGHT if len(w) > 1}
_ALL = sorted(set(_IPA_FOR) | set(_CAPS), key=len, reverse=True)
_NAMES = re.compile(
    r"(?P<link>\[[^\]\n]*\]\(/[^)\n]*/\))"  # an existing misaki mark: leave it be
    r"|(?<![\w\[])(?P<word>" + "|".join(map(re.escape, _ALL)) + r")" + _POSSESSIVE + r"(?![\w\]])"
)

# "Job" is the man in a Bible game, except in the phrases where it is the word.
_JOB_IS_A_TASK_AFTER = re.compile(
    r"\b(?:a|an|good|great|nice|fine|odd|inside|side|dream|new|top|big|whole|blow|hatchet|desk|day|bad|hard|"
    r"tough|real|first|full-time|part-time|easy|dirty|paint)\W*$",
    re.IGNORECASE,
)
_JOB_IS_A_TASK_BEFORE = re.compile(
    r"^(?:-|\s+(?:well|done|one|offer|interview|title|hunt\w*|market|search|application|description|"
    r"site|security|fair|board|seeker\w*|posting|listing|opening|opportunit\w*|done)\b)",
    re.IGNORECASE,
)


def _not_the_man_job(text: str, start: int, end: int, poss: str) -> bool:
    """True where "Job" is the word, not the man: "Job well done!", "Good Job!", "found a Job", "Job's done".

    Capitalised and in a Bible game, "Job" is the man; these are the phrases
    where it is still the ordinary word (including the dad joke that rides
    on both). Lower-case "job" and all-caps "JOB" never match at all.
    """
    if _JOB_IS_A_TASK_AFTER.search(text[:start]):
        return True
    after = text[end:]
    if poss and poss[1:].lower() == "s" and re.match(r"\s+done\b", after, re.IGNORECASE):
        return True  # "Job's done" is "the job is done"
    return bool(_JOB_IS_A_TASK_BEFORE.match(after))


def _with_possessive(ipa: str, poss: str) -> str:
    """The possessive ending as English says it after that sound: Moses's -ᵻz, Ahab's -z, Job's -z."""
    if not poss or len(poss) == 1:  # "Moses' staff": a bare apostrophe adds no sound
        return ipa
    last = ipa.rstrip("ˈˌː")[-1:]
    if last in ("s", "z", "ʃ", "ʒ", "ʧ", "ʤ"):
        return ipa + "ᵻz"
    if last in ("p", "t", "T", "k", "f", "θ"):
        return ipa + "s"
    return ipa + "z"


def pronounce(text: str) -> str:
    """Mark the Bible names Kokoro would say wrong with how they are actually said.

    A listed name becomes misaki inline IPA, ``[Habakkuk](/həbˈækək/)``, and so
    does its possessive (``[Moses's](/mˈOzᵻzᵻz/)``) and its shouted form
    (``HABAKKUK``). A name misaki already says right is only rescued from
    capitals ("NOAH" -> "Noah"), which misaki would otherwise spell out.
    Existing marks are left alone, and only whole words match.
    """
    text = text or ""

    def mark(m: re.Match[str]) -> str:
        if m.group("link"):
            return m.group("link")
        word, poss = m.group("word"), m.group("poss") or ""
        if word in _IPA_FOR:
            if word == "Job" and _not_the_man_job(text, m.start(), m.end(), poss):
                return m.group(0)
            return f"[{word}{poss}](/{_with_possessive(_IPA_FOR[word], poss)}/)"
        return _CAPS[word] + (poss.lower() if poss else "")

    return unshout(_NAMES.sub(mark, text))


#: Capitals that really are letters said one by one.
ACRONYMS = frozenset({"TV", "AI", "OK", "USA", "HOA", "GPS", "ID", "VIP", "NIV", "KJV", "ESV", "NKJV", "DVD", "ATM",
                      "AM", "PM", "BC", "AD", "II", "III", "IV", "VI", "VII", "VIII", "IX", "XL", "UK", "MVP", "PB",
                      "CEO", "DIY", "FAQ", "RSVP", "VBS", "PTA"})
_SHOUT = re.compile(r"(?P<link>\[[^\]]*\]\(/[^)]*/\))|\b(?P<word>[A-Z][A-Z]+(?:['’][A-Z]+)?)\b")


def unshout(text: str) -> str:
    """Headlines and "WE ASKED 100 PARENTS" in lower case: misaki spells a word in capitals letter by letter."""
    def fix(m: re.Match[str]) -> str:
        if m.group("link"):
            return m.group("link")
        word = m.group("word")
        return word if word in ACRONYMS else word[0] + word[1:].lower()

    return _SHOUT.sub(fix, text or "")


_SENTENCE = re.compile(r"(?<=[.!?…])\s+(?=\S)")


_ABBREV = re.compile(r"(?:^|\s)(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|Mt)\.$", re.IGNORECASE)
_PUNCT_ONLY = re.compile(r"^[.…!?\s]+$")


def sentences(text: str) -> list[str]:
    """A line as a host breathes it: one sentence at a time.

    "Mrs." does not end a sentence, and a lone "..." (the beat before a
    punchline) stays with the sentence before it: sent on its own it was a
    clip of nothing, or no clip at all, which lost the whole line.
    """
    out: list[str] = []
    for piece in (s.strip() for s in _SENTENCE.split((text or "").strip())):
        if not piece:
            continue
        if out and (_PUNCT_ONLY.match(piece) or _ABBREV.search(out[-1])):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def cadence(sentence: str, base: float = ANNOUNCER_SPEED) -> tuple[float, float]:
    """``(speed, pause after in ms)`` for one sentence, the way a person reads it.

    Questions slow down and leave room for the answer, exclamations push
    forward, a trailing "..." hangs before the punchline. A little variation
    (fixed per sentence) keeps a run of sentences from sounding metronomic.
    """
    s = sentence.rstrip()
    jitter = ((sum(map(ord, s)) % 9) - 4) * 0.008  # ±3.2 %, same every time for the same words
    if s.endswith(("...", "…")):
        return base * 0.96 + jitter, 480.0
    if s.endswith("?"):
        return base * 0.98 + jitter, 300.0
    if s.endswith("!"):
        return base * 1.06 + jitter, 210.0
    return base + jitter, 260.0

_CALL = re.compile(r"^\s*((?:extra\b[!.,]?\s*){1,3})(read all about it[!.]?\s*)?", re.IGNORECASE)


def split_show_line(text: str) -> list[tuple[str, str]]:
    """``[(role, text), ...]``: an opening "Extra! Extra!" is the paper boy's, the rest the announcer's."""
    clean = (text or "").strip()
    m = _CALL.match(clean)
    if not m or not m.group(1).strip():
        return [("announcer", clean)] if clean else []
    rest = clean[m.end():].strip()
    parts = [("paperboy", PAPERBOY_CALL)]
    if rest:
        parts.append(("announcer", rest))
    return parts


# --------------------------------------------------------------------------- wav io

def read_wav(data: bytes) -> tuple[list[float], int]:
    with wave.open(io.BytesIO(data)) as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("expected 16-bit PCM")
    s = array.array("h")
    s.frombytes(raw)
    if ch > 1:
        return [sum(s[i:i + ch]) / (ch * 32768.0) for i in range(0, len(s), ch)], rate
    return [x / 32768.0 for x in s], rate


def write_wav(samples: list[float], rate: int) -> bytes:
    out = array.array("h", (max(-32767, min(32767, int(x * 32767.0))) for x in samples))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(out.tobytes())
    return buf.getvalue()


# --------------------------------------------------------------------------- building blocks

def _alpha(cutoff_hz: float, rate: int) -> float:
    return 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / rate)


def highpass(x: list[float], rate: int, cutoff_hz: float) -> list[float]:
    a, lp, out = _alpha(cutoff_hz, rate), 0.0, []
    for v in x:
        lp += a * (v - lp)
        out.append(v - lp)
    return out


def lowpass(x: list[float], rate: int, cutoff_hz: float) -> list[float]:
    a, lp, out = _alpha(cutoff_hz, rate), 0.0, []
    for v in x:
        lp += a * (v - lp)
        out.append(lp)
    return out


def presence(x: list[float], rate: int, cutoff_hz: float = 2500.0, gain: float = 0.6) -> list[float]:
    """High-shelf lift: add back the part above ``cutoff_hz`` (gain 0.6 ~ +4 dB)."""
    a, lp, out = _alpha(cutoff_hz, rate), 0.0, []
    for v in x:
        lp += a * (v - lp)
        out.append(v + gain * (v - lp))
    return out


def compress(x: list[float], rate: int, threshold_db: float = -20.0, ratio: float = 3.5,
             attack_ms: float = 4.0, release_ms: float = 140.0, lookahead_ms: float = 5.0) -> list[float]:
    """Feed-forward compressor: even loudness, the way a broadcast voice sounds.

    It never clicks. The detector follows the level, but the gain is smoothed on
    its own (so it cannot wobble within a waveform, which is distortion), and it
    looks ahead, so the gain is already down when a hard onset like "Ooh" arrives
    after a pause. The old one clamped down in the first milliseconds of the
    vowel, and that sudden change was heard as a click.
    """
    n = len(x)
    if not n:
        return []
    det_a = math.exp(-1.0 / (rate * 0.0005))
    det_r = math.exp(-1.0 / (rate * release_ms / 1000.0))
    g_att = math.exp(-1.0 / (rate * attack_ms / 1000.0))
    g_rel = math.exp(-1.0 / (rate * release_ms / 1000.0))
    thr = 10 ** (threshold_db / 20.0)
    env = 0.0
    target = [1.0] * n
    for i, v in enumerate(x):
        level = abs(v)
        env = det_a * env + (1 - det_a) * level if level > env else det_r * env + (1 - det_r) * level
        if env > thr:
            target[i] = (thr * (env / thr) ** (1.0 / ratio)) / env
    look = max(0, int(rate * lookahead_ms / 1000.0))
    # The quietest gain in the next `look` samples, as a sliding-window minimum (one pass).
    ahead = [1.0] * n
    window: collections.deque[int] = collections.deque()
    for j in range(n + look):
        if j < n:
            while window and target[window[-1]] >= target[j]:
                window.pop()
            window.append(j)
        i = j - look
        if i >= 0:
            while window[0] < i:
                window.popleft()
            ahead[i] = target[window[0]]
    out = [0.0] * n
    g = 1.0
    for i in range(n):
        t = ahead[i]
        g = g_att * g + (1 - g_att) * t if t < g else g_rel * g + (1 - g_rel) * t
        out[i] = x[i] * g
    return out


def echoes(x: list[float], rate: int, taps: list[tuple[float, float]], tail_lowpass_hz: float = 3000.0) -> list[float]:
    """Add delayed, darkened copies (ms, gain): a room for the announcer, a street for the newsboy."""
    longest = max((int(rate * ms / 1000.0) for ms, _ in taps), default=0)
    dark = lowpass(x, rate, tail_lowpass_hz)
    out = x + [0.0] * longest
    for ms, g in taps:
        d = int(rate * ms / 1000.0)
        for i, v in enumerate(dark):
            out[i + d] += g * v
    return out


def drive(x: list[float], amount: float = 1.8) -> list[float]:
    """Soft clipping: the edge of a voice pushed to a shout."""
    norm = math.tanh(amount)
    return [math.tanh(amount * v) / norm for v in x]


def resample(x: list[float], factor: float) -> list[float]:
    """Play ``factor`` times faster (and higher). Linear interpolation is plenty for a shout."""
    if factor == 1.0 or not x:
        return list(x)
    n = int(len(x) / factor)
    last = len(x) - 1
    out = []
    for i in range(n):
        p = i * factor
        j = int(p)
        f = p - j
        a = x[j]
        b = x[j + 1] if j < last else a
        out.append(a + (b - a) * f)
    return out


def normalize(x: list[float], peak: float = 0.89) -> list[float]:
    top = max((abs(v) for v in x), default=0.0)
    return [v * peak / top for v in x] if top > 1e-6 else list(x)


def silence(rate: int, ms: float) -> list[float]:
    return [0.0] * int(rate * ms / 1000.0)


# --------------------------------------------------------------------------- the two voices

def announcer_fx(x: list[float], rate: int) -> list[float]:
    # Clean on purpose (chosen by ear): no room echo, no drive, nothing that adds
    # noise. A leveller and a little presence.
    y = highpass(normalize(x, 0.9), rate, 85.0)
    y = presence(y, rate, 2600.0, 0.6)
    y = compress(y, rate, -26.0, 3.0, attack_ms=5.0, release_ms=150.0)
    return normalize(y, 0.9)


def quieter(wav: bytes, db: float) -> bytes:
    """The clip ``db`` quieter: a plain gain, so nothing is added to the sound."""
    if not db:
        return wav
    x, rate = read_wav(wav)
    g = 10 ** (-db / 20.0)
    return write_wav([v * g for v in x], rate)


def paperboy_fx(x: list[float], rate: int) -> list[float]:
    y = resample(x, PAPERBOY_PITCH) if PAPERBOY_PITCH != 1.0 else list(x)
    y = highpass(y, rate, 200.0)
    y = lowpass(y, rate, 5200.0)
    y = compress(normalize(y, 0.9), rate, -22.0, 3.0, attack_ms=3.0, release_ms=110.0)
    y = drive(normalize(y, 0.95), 1.25)
    y = echoes(y, rate, [(135.0, 0.24), (270.0, 0.11), (420.0, 0.05)], 2400.0)
    return normalize(y, 0.93)


def _trim(x: list[float], rate: int, keep_ms: float = 70.0) -> list[float]:
    """Drop the dead air Kokoro leaves around a clip, so the pauses are ours.

    The cut is relative to the clip's own loudness and is faded (a hard cut
    was heard as a click in every pause).
    """
    peak = max((abs(v) for v in x), default=0.0)
    floor = max(0.006, 0.04 * peak)
    keep = int(rate * keep_ms / 1000.0)
    idx = [i for i, v in enumerate(x) if abs(v) > floor]
    if not idx:
        return x
    y = x[max(0, idx[0] - keep): min(len(x), idx[-1] + keep)]
    fin, fout = min(len(y), int(rate * 0.012)), min(len(y), int(rate * 0.06))
    # S-curve fades: a straight ramp leaves a kink where it ends.
    for i in range(fin):
        y[i] *= 0.5 - 0.5 * math.cos(math.pi * i / fin)
    for i in range(fout):
        y[len(y) - 1 - i] *= 0.5 - 0.5 * math.cos(math.pi * i / fout)
    return y


def gate(x: list[float], rate: int, threshold_db: float = -40.0) -> list[float]:
    """A soft noise gate: what is left between words (breath, hiss) goes quiet.

    Gain follows a smoothed envelope (quick open, slow close) so it never
    chatters, and below the threshold it falls off steeply rather than cutting.
    """
    peak = max((abs(v) for v in x), default=0.0) or 1.0
    thr = peak * 10 ** (threshold_db / 20.0)
    env_a, env_r = math.exp(-1 / (rate * 0.002)), math.exp(-1 / (rate * 0.08))
    g_a, g_r = math.exp(-1 / (rate * 0.003)), math.exp(-1 / (rate * 0.06))
    env, gain, out = 0.0, 1.0, []
    for v in x:
        lv = abs(v)
        env = env_a * env + (1 - env_a) * lv if lv > env else env_r * env + (1 - env_r) * lv
        target = 1.0 if env >= thr else (env / thr) ** 3
        gain = g_a * gain + (1 - g_a) * target if target > gain else g_r * gain + (1 - g_r) * target
        out.append(v * gain)
    return out


def assemble(parts: list[tuple[str, bytes]] | list[tuple[str, bytes, float]]) -> bytes:
    """``[(role, kokoro wav[, pause after ms]), ...]`` -> one processed show line.

    Each role's sentences are joined with their own pauses first and then
    processed together, so the voice's character stays even across them.
    """
    rate = 0
    groups: list[tuple[str, list[float], float]] = []
    for part in parts:
        role, wav = part[0], part[1]
        gap = float(part[2]) if len(part) > 2 else 260.0  # type: ignore[misc]
        samples, r = read_wav(wav)
        if rate and r != rate:
            samples = resample(samples, r / rate)
        rate = rate or r
        samples = _trim(samples, rate)
        if groups and groups[-1][0] == role:
            _, prev, prev_gap = groups[-1]
            groups[-1] = (role, prev + silence(rate, prev_gap) + samples, gap)
        else:
            groups.append((role, samples, gap))
    out: list[float] = []
    for i, (role, samples, gap) in enumerate(groups):
        shaped = paperboy_fx(samples, rate) if role == "paperboy" else announcer_fx(samples, rate)
        out += gate(shaped, rate)
        if i < len(groups) - 1:
            out += silence(rate, max(gap, 380.0))
    return write_wav(out, rate or 24000)
