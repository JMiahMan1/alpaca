"""Multiplayer Biblical Worldview party game engine ("Parable Party").

Implements a Jackbox / *Use Your Words* style room state machine for the
Jarvis Family Hub and Alpaca Arcade:

- **5 Round Types across a 7-Round Match**:
  1. ``blank_o_matic`` - Proverb & Parable fill-in-the-blank scenarios
  2. ``extra_extra`` - *The Jerusalem Gazette* headline/caption writing for illustrated scenes
  3. ``truth_or_bluff`` - Biblical Worldview Fibbage: write a believable bluff to fool
     others while spotting the real Scriptural/historical truth
  4. ``sub_the_scroll`` - Write the spoken dialogue/subtitle for an animated Biblical scene
  5. ``survey_says`` - Round 7 Double-Point Lightning Finale (2x stakes!)

- **Signature "Scribe's Decoy / House Trap" Mechanic**:
  - Every comedy round mixes a pre-written **Scribe's Trap** answer anonymously
    into the voting pool (or lets a stuck player submit it as a **Scribe's Decoy**).
  - Original answer: **+1,000 pts** per vote (2,000 in Lightning Round).
  - Scribe's Decoy (submitted by a player): **+500 pts** per vote (1,000 in Lightning Round).
  - Falling for the Scribe's Trap: **-250 pts** to the voter (-500 in Lightning Round).
  - In ``truth_or_bluff``: **+1,000 pts** for spotting the True Biblical Answer,
    **+750 pts** per player fooled by your bluff, and **-250 pts** for falling
    for the Scribe's Decoy.

- **AI Companions & Spectators**:
  - Works with 1 to 8 players (friendly Biblical scribes *Barnabas*, *Priscilla*,
    and *Apollos* automatically fill seats when fewer than 3 humans join so a
    solo player or pair can play a full party match anytime).
  - Additional family members beyond 8 (or late joiners) can join as voting
    Audience Spectators.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    from arcade import announcer
except ImportError:  # the arcade image runs with arcade/ on sys.path
    import announcer  # type: ignore[no-redef]

AUDIO_SERVER_URL = os.getenv("AUDIO_SERVER_URL", "http://audio-server:8082").rstrip("/")
PROXY_URL = os.getenv("PROXY_URL", "http://host.docker.internal:11434").rstrip("/")
EXECUTION_SVC_URL = os.getenv("EXECUTION_SVC_URL", "").rstrip("/")

DEFAULT_ANNOUNCER_VOICE = "am_onyx,am_fenrir"

CURATED_ANNOUNCER_VOICES: tuple[dict[str, str], ...] = (
    {
        "id": "am_onyx,am_fenrir",
        "label": "📣 The Big Voice (Classic Game-Show Announcer)",
        "kind": "kokoro",
    },
    {
        "id": "am_fenrir,am_michael",
        "label": "🎙️ Brother Barnaby (Warm Game-Show Host)",
        "kind": "kokoro",
    },
    {
        "id": "bm_george,bm_fable",
        "label": "🎩 Oxford Scribe (British Narrator)",
        "kind": "kokoro",
    },
    {
        "id": "af_heart,af_bella",
        "label": "🌟 Sister Lydia (Bright Co-Host)",
        "kind": "kokoro",
    },
    {
        "id": "am_michael",
        "label": "📜 Michael (Classic Studio)",
        "kind": "kokoro",
    },
    {
        "id": "am_fenrir",
        "label": "🦁 Fenrir (Deep Announcer)",
        "kind": "kokoro",
    },
    {
        "id": "af_heart",
        "label": "💛 Heart (Warm Storyteller)",
        "kind": "kokoro",
    },
    {
        "id": "bm_george",
        "label": "🇬🇧 George (Royal Herald)",
        "kind": "kokoro",
    },
    {
        "id": "am_puck",
        "label": "⚡ Puck (Upbeat Showman)",
        "kind": "kokoro",
    },
)

ROOM_TTL_SECONDS = 7200
#: A room nobody's phone has checked in on for this long is closed.
ROOM_IDLE_SECONDS = 600
ROOM_EMPTY_GRACE_SECONDS = 120
STAGE_PRESENCE_SECONDS = 8.0
#: How far behind real time a streamed stage (Roku, plain video TVs) is seen:
#: 2 s HLS segments, the player's buffer and the render. The stage page reports
#: when it is the one being streamed (it runs in headless Chromium), and while
#: it is, phones hold each phase until the picture catches up and every phase
#: gets the same extra time.
STREAM_STAGE_DELAY_S = float(os.getenv("PARTY_STREAM_DELAY_S", "8"))
WRITE_SECONDS = 55
#: The announcer reads the prompt (and a quip) as writing starts; that time is not the players'.
PROMPT_READ_SECONDS = 8
#: After the last answer is revealed, how long the scores stay up before the next round starts on its own.
REVEAL_LINGER_SECONDS = 12
VOTE_SECONDS = 35
REVEAL_SECONDS = 8
#: The TV reveals answers one at a time (fewest votes first, the winner last),
#: so the reveal lasts longer when there are more answers to unveil.
REVEAL_PER_ANSWER_SECONDS = 3.5
TOTAL_ROUNDS = 7

# Memorable 4-letter Biblical room codes for easy TV-to-phone entry.
ROOM_WORDS = (
    "NOAH",
    "ZION",
    "DOVE",
    "LION",
    "HARP",
    "LAMP",
    "SALT",
    "STAR",
    "VINE",
    "ROCK",
    "GOLD",
    "EDEN",
    "ARK7",
    "CANA",
    "RUTH",
    "AMEN",
    "HOPE",
    "KING",
    "LAMB",
    "WAVE",
)

AI_COMPANIONS = (
    {"id": "bot-barnabas", "name": "Barnabas", "initials": "BAR", "avatar": "🧔🏽"},
    {"id": "bot-priscilla", "name": "Priscilla", "initials": "PRI", "avatar": "🧕🏽"},
    {"id": "bot-apollos", "name": "Apollos", "initials": "APO", "avatar": "📜"},
)

PLAYER_AVATARS = ("👑", "🦁", "🕊️", "🐑", "🐪", "🐟", "🎺", "🏺", "⭐", "🛡️", "🌿", "🍇", "⛵", "🔥", "🌈", "🦉")
#: A phone that has not polled for this long is shown as away, and a host who
#: has been away this long hands the host controls to the next player.
PLAYER_AWAY_SECONDS = 15.0
HOST_HANDOFF_SECONDS = 45.0

# 7-round match schedule covering all 5 requested modes.
MATCH_ROUND_SCHEDULE = (
    "blank_o_matic",
    "extra_extra",
    "truth_or_bluff",
    "sub_the_scroll",
    "blank_o_matic",
    "truth_or_bluff",
    "survey_says",
)

ROUND_META: dict[str, dict[str, Any]] = {
    "blank_o_matic": {
        "title": "Proverb & Parable Blank-O-Matic",
        "badge": "FILL IN THE BLANK",
        "icon": "📜",
        "multiplier": 1,
        "instructions": "Fill in the blank with your funniest or wisest punchline - or play the Scribe's Decoy to set a trap!",
    },
    "extra_extra": {
        "title": "Extra! Extra! The Jerusalem Gazette",
        "badge": "BREAKING HEADLINE",
        "icon": "📰",
        "multiplier": 1,
        "instructions": "Write a breaking front-page newspaper headline or caption for this illustrated scene!",
    },
    "truth_or_bluff": {
        "title": "Truth or Bluff: Biblical Worldview",
        "badge": "FIBBAGE & TRUTH",
        "icon": "🏛️",
        "multiplier": 1,
        "instructions": "Write a believable FAKE answer to fool your family - then spot the real Scriptural truth in voting!",
    },
    "sub_the_scroll": {
        "title": "Sub the Scroll!",
        "badge": "SCENE DIALOGUE",
        "icon": "🎭",
        "multiplier": 1,
        "instructions": "Write what the character in this scene is ACTUALLY saying or thinking right now!",
    },
    "survey_says": {
        "title": "Survey Says! Double-Point Finale",
        "badge": "⚡ 2X LIGHTNING ROUND",
        "icon": "⚡",
        "multiplier": 2,
        "instructions": "DOUBLE POINTS! Every vote is worth 2,000 points - give your best crowd-pleasing family answer!",
    },
}


PROMPT_BANK: dict[str, list[dict[str, Any]]] = {
    "blank_o_matic": [
        {
            "id": "bom_noah_off_ark",
            "prompt": "When Noah's family finally stepped off the Ark after 370 days with two of every animal, the very first thing Mrs. Noah said was: \"______.\"",
            "decoy": "Nobody say the word 'moist' or 'petting zoo' for at least six centuries",
            "bot_answers": [
                "If anyone tracks mud into this tent, you're swimming back to Ararat",
                "Who forgot to let the skunks off last?",
                "I am never hosting a family staycation again",
            ],
            "scripture_note": "Genesis 8:16-18 - After more than a year aboard the Ark, God brought Noah's whole household safely onto dry ground.",
            "announcer": "Round time! When Noah's family finally stepped off the Ark, what was the very first thing Mrs. Noah said?",
        },
        {
            "id": "bom_solomon_prayer",
            "prompt": "King Solomon was the wisest man alive, which means his most frequent dinner-table prayer was: \"Lord, please help me ______.\"",
            "decoy": "remember which anniversary is this Tuesday",
            "bot_answers": [
                "find the end of this 40-foot palace grocery scroll",
                "look wise while pretending I know where I left my keys",
                "settle who gets the last fig without needing a sword",
            ],
            "scripture_note": "1 Kings 3:9 - Solomon asked God for an understanding heart to discern between good and evil.",
            "announcer": "Fill in the blank! Even wise King Solomon needed extra prayer at the dinner table!",
        },
        {
            "id": "bom_paul_autoreply",
            "prompt": "If the Apostle Paul had a church group chat with Corinth, his pinned auto-reply message would say: \"Grace and peace! Before you text me, please ______.\"",
            "decoy": "ask yourself if this needs a whole epistle or just an apology",
            "bot_answers": [
                "stop dividing into 'Team Paul' and 'Team Apollos' over the potluck menu",
                "check First Corinthians before making me write a Fourth",
                "remember love is patient, even when the Wi-Fi in Ephesus is not",
            ],
            "scripture_note": "1 Corinthians 1:10 - Paul urged the church to agree with one another and be united in mind and purpose.",
            "announcer": "Imagine the Apostle Paul managing a modern church group chat!",
        },
        {
            "id": "bom_proverbs_roof",
            "prompt": "Proverbs 21:9 says it is better to live on a corner of the roof than in a house shared with ______.",
            "decoy": "someone who eats the last cinnamon roll and puts the empty box back",
            "bot_answers": [
                "three hungry kids arguing over who called the front seat first",
                "a rooster practicing for Peter's denial at 3:00 AM",
                "an unplugged router during family movie night",
            ],
            "scripture_note": "Proverbs 21:9 & 15:17 - Scripture prizes peace and kindness at home over luxury accompanied by strife.",
            "announcer": "A classic Proverb gets a modern family update! Better to live on a corner of the roof than with what?",
        },
        {
            "id": "bom_five_thousand_boy",
            "prompt": "After the Feeding of the 5,000, the little boy who packed five barley loaves and two fish ran home and told his mom: \"Mom, you won't believe ______!\"",
            "decoy": "how many leftovers I brought back in these twelve giant baskets",
            "bot_answers": [
                "everyone finally stopped complaining about your sardine sandwiches",
                "Jesus blessed my lunchbox and catered the whole hillside",
                "I traded my crust for enough bread to open a bakery",
            ],
            "scripture_note": "John 6:9-13 - What seems small in our hands becomes abundant when offered to Jesus.",
            "announcer": "Imagine the little boy explaining the Feeding of the Five Thousand to his mom that evening!",
        },
        {
            "id": "bom_jonah_brochure",
            "prompt": "The real reason Jonah bought a one-way ship ticket to Tarshish instead of going to Nineveh was because the Tarshish travel brochure promised ______.",
            "decoy": "100% fewer awkward repentance sermons and zero giant fish",
            "bot_answers": [
                "all-you-can-eat Mediterranean olives and no prophetic overtime",
                "a quiet lower-deck hammock where nobody asks where you're from",
                "complimentary sea-breeze naps with zero gourd-eating worms",
            ],
            "scripture_note": "Jonah 1:3 - You can buy a ticket to Tarshish, but you can never outrun the Lord's mercy and call!",
            "announcer": "Why did Tarshish sound so appealing to Jonah? Fill in the blank!",
        },
        {
            "id": "bom_david_potluck",
            "prompt": "If King David wrote a Psalm right before a Sunday church potluck, verse 1 would read: \"O Lord, shield my plate from ______.\"",
            "decoy": "the mystery green Jell-O salad that contains shredded carrots",
            "bot_answers": [
                "Lukewarm casserole number seven and tiny paper forks",
                "being last in line behind the entire youth group",
                "burnt coffee that tests the perseverance of the saints",
            ],
            "scripture_note": "Acts 2:46 - Breaking bread together with glad and generous hearts has been a church hallmark from day one!",
            "announcer": "A Psalm for Sunday Fellowship Lunch! Shield my plate from what?",
        },
        {
            "id": "bom_daniel_lions",
            "prompt": "When Daniel woke up completely unharmed after a full night's sleep in the lions' den, the only awkward part was ______.",
            "decoy": "telling the biggest lion to stop hogging the blanket",
            "bot_answers": [
                "waking up with terrible mane-hair and cat fur on his robe",
                "King Darius yelling down the pit before Daniel had his morning coffee",
                "three lions asking if he brought any snacks",
            ],
            "scripture_note": "Daniel 6:22 - 'My God sent his angel and shut the lions' mouths, and they have not harmed me.'",
            "announcer": "Daniel had the best night's sleep in Babylon while King Darius paced the floor!",
        },
        {
            "id": "bom_prodigal_yelp",
            "prompt": "If the Prodigal Son's older brother had Yelp, his 1-star review of the welcome-home banquet would complain: \"______.\"",
            "decoy": "Zero goat for the guy who worked the field, fatted calf for the guy who smelled like pigs",
            "bot_answers": [
                "Music too loud, ring too shiny, nobody asked my opinion",
                "I've been here the whole time and didn't even get a commemorative sandal",
                "Five stars for the beef, one star for my seating assignment",
            ],
            "scripture_note": "Luke 15:31-32 - The Father reminds us to rejoice when the lost are found rather than keeping a ledger of resentment.",
            "announcer": "Uh oh, the older brother pulled out his phone outside the banquet hall!",
        },
        {
            "id": "bom_sunday_morning",
            "prompt": "The biggest miracle that happens in a Christian household before 8:45 AM on Sunday morning is ______.",
            "decoy": "finding two matching dress socks that belong to the same decade",
            "bot_answers": [
                "everyone getting into the car with both shoes AND a cheerful attitude",
                "switching from 'where are my keys?!' to 'Good morning, brother!' in the church parking lot",
                "a toddler keeping their outfit clean for twelve whole minutes",
            ],
            "scripture_note": "Psalm 122:1 - 'I was glad when they said to me, Let us go to the house of the Lord!' (Eventually!)",
            "announcer": "Every family knows the Sunday morning scramble! What's the biggest miracle before 8:45 AM?",
        },
    ],
    "extra_extra": [
        {
            "id": "ee_jonah_beach",
            "scene": "jonah_beach",
            "subprompt": "SCENE: A giant whale spouts offshore while a kelp-covered prophet washes up on Nineveh Beach holding a soggy scroll as fishermen drop their nets.",
            "prompt": "Write the front-page headline for THE NINEVEH DAILY HERALD!",
            "decoy": "LOCAL SUBMARINE ROUTE CANCELLED AFTER PASSENGER EJECTED WITHOUT PARACHUTE",
            "bot_answers": [
                "SMELLS LIKE REPENTANCE: Seaweed-Clad Preacher Arrives Via Express Whale!",
                "NINEVEH FISHERMEN STUNNED: 'That One Definitely Wasn't On Our Bait List!'",
                "MAN SKIPS BOAT FARE, PAYS HEAVY CLEANING DEPOSIT IN BELLY OF GREAT FISH",
            ],
            "scripture_note": "Jonah 2:10 - The Lord spoke to the fish, and it vomited Jonah out onto dry land - right back on mission.",
            "announcer": "Extra! Extra! Look at the scene on screen and write the front-page headline for The Nineveh Daily Herald!",
        },
        {
            "id": "ee_ark_hoa",
            "scene": "ark_hoa",
            "subprompt": "SCENE: Noah's 450-foot wooden Ark towers over a dry neighborhood while a confused Homeowners Association inspector measures it with a tiny ruler.",
            "prompt": "Write the breaking headline for THE MESOPOTAMIA GAZETTE!",
            "decoy": "LOCAL MAN VIOLATES DRIVEWAY BOAT ORDINANCE BY 300 CUBITS, CITES 'UPCOMING RAIN'",
            "bot_answers": [
                "HOA FURIOUS AS NEIGHBOR PARKS THREE-STORY ZOO CRUISE IN FRONT YARD",
                "WEATHER FORECASTERS LAUGH AT GOPHER-WOOD YACHT; GIRAFFES ALREADY BOARDING",
                "PERMIT DENIED? NOAH SAYS 'WAIT UNTIL YOU SEE THE 40-DAY DRAINAGE REPORT'",
            ],
            "scripture_note": "Hebrews 11:7 - By faith Noah, being warned by God concerning events as yet unseen, in reverent fear constructed an ark.",
            "announcer": "Extra! Extra! Noah's neighbors have called the HOA inspector! Write the front-page headline!",
        },
        {
            "id": "ee_goliath_presser",
            "scene": "goliath_presser",
            "subprompt": "SCENE: Nine-foot bronze-clad Goliath flexes at a pre-fight podium while shepherd boy David casually twirls a sling and checks five smooth stones.",
            "prompt": "Write the sports headline for THE VALLEY OF ELAH SPORTS DESK!",
            "decoy": "UNDisputed Heavyweight Champion Defeated By Kid Who Isn't Old Enough To Shave",
            "bot_answers": [
                "UPSET OF THE MILLENNIUM: Shepherd Boy Brings Pebble To Sword Fight - And Wins!",
                "PHILISTINE ODDSMAKERS STUNNED AS 500-POUND ARMOR LOSES TO FAITH AND SLING",
                "DELIVERY BOY DROPS OFF CHEESE FOR BROTHERS, PICKS UP NATIONAL CHAMPIONSHIP",
            ],
            "scripture_note": "1 Samuel 17:47 - 'The battle is the Lord's, and He will give you into our hand!'",
            "announcer": "Live from the Valley of Elah! Write the sports headline for David versus Goliath!",
        },
        {
            "id": "ee_jericho_walls",
            "scene": "jericho_walls",
            "subprompt": "SCENE: Seven priests blast ram's-horn trumpets on Lap #7 around Jericho while two city wall engineers stare in horror at a massive crack.",
            "prompt": "Write the breaking headline for THE JERICHO ARCHITECTURAL DIGEST!",
            "decoy": "CITY WALL WARRANTY VOIDED BY UNEXPECTED BRASS BAND PARADE",
            "bot_answers": [
                "MARCHING BAND HITS HIGH NOTE; ENTIRE CITY SKYLINE NOW CONVERTIBLE",
                "SEVEN LAPS LATER: Jericho Contractors Admit 'We Did Not Spec For Trumpet Blast'",
                "QUIET WALKERS TURN LOUD ON DAY SEVEN; WALLS IMMEDIATELY RETIRE",
            ],
            "scripture_note": "Joshua 6:20 - God's ways often look unconventional to the world, yet obedience brings down strongholds.",
            "announcer": "Lap number seven around Jericho! Write the breaking headline as the trumpets sound!",
        },
        {
            "id": "ee_zacchaeus_tree",
            "scene": "zacchaeus_tree",
            "subprompt": "SCENE: Chief tax collector Zacchaeus in silk robes perches high on a sycamore branch while Jesus looks up and invites Himself over for lunch.",
            "prompt": "Write the headline for THE JERICHO FINANCIAL TIMES!",
            "decoy": "LOCAL TAX AUDITOR OPENS BRANCH OFFICE IN SYCAMORE TREE, REFUNDS EVERYONE 400%",
            "bot_answers": [
                "SHORT TAX BOSS CLIMBS HIGH, COMES DOWN WITH NEW HEART AND EMPTY WALLET",
                "SURPRISE AUDIT OF GRACE: Jesus Picks Tree-Percher For Lunch Date!",
                "JERICHO CITIZENS REJOICE AS CHIEF COLLECTOR DISCOVERS GENEROSITY",
            ],
            "scripture_note": "Luke 19:9-10 - 'Today salvation has come to this house... For the Son of Man came to seek and to save the lost.'",
            "announcer": "Look up in that sycamore tree! Write the headline for Zacchaeus's big day!",
        },
        {
            "id": "ee_water_wine",
            "scene": "water_wine",
            "subprompt": "SCENE: The Cana wedding master-of-ceremonies takes one sip from a stone water-jar ladle with wide-eyed astonishment while the servants exchange high-fives.",
            "prompt": "Write the society headline for THE CANA WEDDING WEEKLY!",
            "decoy": "BRIDEGROOM SAVES CATERING BUDGET WITH SIX STONE WATER JARS AND ONE VIP GUEST",
            "bot_answers": [
                "CATERING CRISIS AVERTED: Tap Water Upgraded To Vintage Reserve In Seconds!",
                "WEDDING STEWARD BAFFLED: 'Who Saves The Best Beverage For Last?!'",
                "LISTEN TO HIM: Mary's Advice To Kitchen Staff Produces Miracle Vintage!",
            ],
            "scripture_note": "John 2:11 - This, the first of His signs, Jesus did at Cana in Galilee, and manifested His glory.",
            "announcer": "Breaking news from the wedding feast at Cana! Write the front-page headline!",
        },
    ],
    "sub_the_scroll": [
        {
            "id": "sts_balaam_donkey",
            "scene": "balaam_donkey",
            "speaker": "Balaam's Donkey",
            "subprompt": "SCENE: Balaam raises his staff in frustration while his donkey turns around, points a hoof toward an unseen angel with a fiery sword, and opens its mouth.",
            "prompt": "What is Balaam's Donkey ACTUALLY saying in that speech bubble?",
            "decoy": "Look, one of us is a prophet and the other just spotted a ten-foot angel - guess which is which!",
            "bot_answers": [
                "I just saved you from a flaming sword and you're mad about your foot against the wall?!",
                "First time talking, long-time carrier: you need your spiritual glasses checked, boss.",
                "Even I can see the angel in the road, and my diet is literally thistles!",
            ],
            "scripture_note": "Numbers 22:28-31 - Sometimes the humble donkey sees the hand of God clearer than the proud rider!",
            "announcer": "Sub the Scroll! Balaam's donkey just turned around and started talking - write its dialogue!",
        },
        {
            "id": "sts_aaron_calf",
            "scene": "aaron_calf",
            "speaker": "Aaron",
            "subprompt": "SCENE: Moses descends Mount Sinai with glowing stone tablets while Aaron stands awkwardly in front of a shiny golden calf trying to explain.",
            "prompt": "Write Aaron's least-convincing excuse in the speech bubble!",
            "decoy": "Funny story, Moses - we tossed the earrings in the campfire and it just 3D-printed itself!",
            "bot_answers": [
                "Before you get mad, the committee voted that a calf was more approachable than a cloud of fire.",
                "I was trying to melt it into a casserole dish and my chisel slipped!",
                "Technically, I didn't build an idol - the furnace had an unauthorized moo!",
            ],
            "scripture_note": "Exodus 32:24 - Aaron literally claimed 'I threw it into the fire, and out came this calf!'-a timeless picture of shifting blame.",
            "announcer": "Moses just walked down Mount Sinai! What excuse is Aaron putting in that speech bubble?",
        },
        {
            "id": "sts_elijah_ravens",
            "scene": "elijah_ravens",
            "speaker": "The Delivery Raven",
            "subprompt": "SCENE: Elijah sits by the Brook Cherith as two carrier-ravens swoop in carrying fresh bread and meat in a little woven napkin.",
            "prompt": "What is the Lead Raven saying as it drops off Elijah's breakfast?",
            "decoy": "Five-star DoorDash from Heaven - please don't ask where we sourced the sourdough.",
            "bot_answers": [
                "Special delivery from the Almighty! Sign here, prophet, and watch out for crumbs.",
                "Do you know how much willpower it takes a raven not to eat a warm bagel mid-flight?",
                "Morning shift is bread and meat; evening shift is meat and bread. Tip your flyers!",
            ],
            "scripture_note": "1 Kings 17:6 - The ravens brought Elijah bread and meat in the morning and evening, and he drank from the brook.",
            "announcer": "Heaven's original delivery service has landed at Brook Cherith! What is the raven saying?",
        },
        {
            "id": "sts_eutychus_window",
            "scene": "eutychus_window",
            "speaker": "Eutychus (at 11:58 PM)",
            "subprompt": "SCENE: Midnight in an upstairs room full of smoky oil lamps in Troas. Paul is on point #14 of his sermon while Eutychus sits on a third-story windowsill.",
            "prompt": "What is going through Eutychus's head right before he nods off?",
            "decoy": "I'm not sleeping, I'm just examining the inside of my eyelids in the Greek original!",
            "bot_answers": [
                "Third-story open window + twelve oil lamps + midnight theology = totally safe seat choice.",
                "If I blink for just three seconds, nobody will notice...",
                "Note to self: next Sunday, pick a seat on the first floor with a backrest.",
            ],
            "scripture_note": "Acts 20:9 - A young man named Eutychus, sitting at the window, sank into a deep sleep as Paul talked still longer.",
            "announcer": "It's midnight in Troas and Eutychus is on the windowsill! Write his caption!",
        },
        {
            "id": "sts_peter_waves",
            "scene": "peter_waves",
            "speaker": "Simon Peter",
            "subprompt": "SCENE: Peter takes three bold steps onto the Sea of Galilee during a windstorm while the other eleven disciples peer over the boat rail.",
            "prompt": "What is Peter saying right as a giant wave splashes his knees?",
            "decoy": "Hey guys, look - no boat! Wait... why is the water suddenly looking very downhill?!",
            "bot_answers": [
                "Step one: eyes on Jesus! Step two: DO NOT LOOK AT THAT FOUR-FOOT SWELL!",
                "John, don't you dare write the sinking part in your Gospel!",
                "Lord, I have great faith, and also very wet sandals!",
            ],
            "scripture_note": "Matthew 14:29-31 - As long as Peter kept his eyes on Jesus he walked on the water; when Jesus caught him, both returned safely to the boat.",
            "announcer": "Peter is out of the boat and on the waves! What is he saying right now?",
        },
        {
            "id": "sts_wise_men",
            "scene": "wise_men",
            "speaker": "The Third Magi",
            "subprompt": "SCENE: Three Magi on camels pause under a blazing star outside Bethlehem checking their gift boxes of gold, frankincense, and myrrh.",
            "prompt": "What is the Third Wise Man saying to the other two right before knocking on the door?",
            "decoy": "Wait - you brought solid gold and you let me show up with embalming spice?!",
            "bot_answers": [
                "See? I told you following the giant glowing star was faster than stopping at Herod's palace!",
                "Did anyone remember to include a gift receipt from Persia?",
                "Next time we coordinate the baby registry BEFORE riding 800 miles on camels!",
            ],
            "scripture_note": "Matthew 2:10-11 - When they saw the star, they rejoiced exceedingly with great joy and offered Him their treasures.",
            "announcer": "The Magi have arrived in Bethlehem! What is the third Wise Man saying?",
        },
    ],
    "truth_or_bluff": [
        {
            "id": "tob_eutychus_after",
            "prompt": "In Acts 20, after young Eutychus fell from a third-story window at midnight and Paul revived him, what did Paul immediately do until daybreak? ______",
            "truth": "Went back upstairs, broke bread, and kept preaching until dawn",
            "truth_keywords": ["kept preaching", "preached until dawn", "talked until daybreak", "went back upstairs", "continued preaching"],
            "decoy": "Dismissed the service early and installed wooden shutters on all windows",
            "bot_answers": [
                "Moved the congregation to the ground-floor courtyard for a hymn sing",
                "Declared a mandatory 10:00 PM curfew for all future church meetings",
                "Appointed two deacons to hold the back row awake",
            ],
            "scripture_note": "Acts 20:10-11 - After embracing Eutychus and assuring everyone his life was in him, Paul went right back upstairs, broke bread, and conversed with them until daybreak!",
            "announcer": "Truth or Bluff! Write a believable FAKE answer to fool your family, then try to spot what Paul REALLY did in Acts chapter 20!",
        },
        {
            "id": "tob_elisha_axe",
            "prompt": "In 2 Kings 6, when a young prophet accidentally dropped a borrowed iron axe head into the Jordan River, how did Elisha get it back? ______",
            "truth": "He threw a cut stick into the water and the iron axe head floated to the surface",
            "truth_keywords": ["floated", "iron float", "threw a stick", "stick in", "axe head floated"],
            "decoy": "He struck the Jordan with his cloak so the water parted around the axe",
            "bot_answers": [
                "He prayed and a giant river fish swam up holding the axe handle",
                "He had the prophets link arms and dredge the riverbed with a linen net",
                "He magnetised his bronze staff to lift it from the mud",
            ],
            "scripture_note": "2 Kings 6:5-7 - Elisha cut off a stick, threw it in where the borrowed axe sank, and made the heavy iron float so the student could reach out and take it!",
            "announcer": "Truth or Bluff! How did Elisha recover a sunken iron axe head in Second Kings chapter 6? Write a bluff to trick the room!",
        },
        {
            "id": "tob_ehud_detail",
            "prompt": "In Judges 3, what specific physical detail does Scripture highlight about Ehud, the judge who delivered Israel from King Eglon of Moab? ______",
            "truth": "He was left-handed (allowing him to strap his sword to his right thigh undetected)",
            "truth_keywords": ["left-handed", "left handed", "lefthanded"],
            "decoy": "He had seven locks of red hair and could outrun a chariot",
            "bot_answers": [
                "He was the shortest warrior in the tribe of Benjamin",
                "He spoke with a whisper so quiet guards leaned in to hear him",
                "He was born with six fingers on each hand",
            ],
            "scripture_note": "Judges 3:15-16 - Ehud son of Gera, a Benjamite, was a left-handed man, so palace guards checking his left side missed the blade on his right thigh.",
            "announcer": "Truth or Bluff! What unusual detail does the Book of Judges record about Ehud? Write a convincing fake!",
        },
        {
            "id": "tob_dagon_statue",
            "prompt": "When the Philistines placed the captured Ark of the Covenant beside their idol Dagon in 1 Samuel 5, what happened to the statue of Dagon on the second morning? ______",
            "truth": "It fell face down before the Ark with its head and both hands broken off on the threshold",
            "truth_keywords": ["head and both hands", "head and hands", "hands broken", "fell face down"],
            "decoy": "It melted into a puddle of bronze facing toward Shiloh",
            "bot_answers": [
                "It was found turned backwards facing the temple wall in shame",
                "It crumbled into fine white dust from top to bottom",
                "Its stone crown cracked in half and rolled under the Ark",
            ],
            "scripture_note": "1 Samuel 5:3-4 - Dagon fell face down before the Ark of the Lord, and its head and both hands were lying cut off on the threshold - showing the Lord alone is God.",
            "announcer": "Truth or Bluff! What happened to the Philistine idol Dagon when left overnight with the Ark of the Covenant?",
        },
        {
            "id": "tob_peter_fish_count",
            "prompt": "In John 21, when the risen Jesus cooked breakfast on the beach for His disciples, exactly how many large fish did Simon Peter haul ashore without the net tearing? ______",
            "truth": "153 large fish",
            "truth_keywords": ["153", "one hundred fifty-three", "one hundred and fifty-three"],
            "decoy": "144 large fish (twelve for each tribe of Israel)",
            "bot_answers": [
                "70 large fish",
                "120 large fish",
                "40 large fish",
            ],
            "scripture_note": "John 21:11 - John records the exact eyewitness count: 'full of large fish, 153 of them; and although there were so many, the net was not torn.'",
            "announcer": "Truth or Bluff! John chapter 21 gives an exact eyewitness number of fish in the net - can you bluff your family?",
        },
        {
            "id": "tob_absalom_hair",
            "prompt": "According to 2 Samuel 14:26, why did King David's son Absalom have to cut his hair once at the end of every year? ______",
            "truth": "Because it grew so heavy on him that the cut hair weighed 200 shekels (about 5 pounds)",
            "truth_keywords": ["heavy", "200 shekels", "weighed", "five pounds", "5 pounds"],
            "decoy": "Because royal princes were required to renew a yearly Nazarite vow at Passover",
            "bot_answers": [
                "Because his chariot helmet no longer fit over his curls",
                "Because he donated a lock to every city gate in Hebron",
                "Because palace etiquette forbade hair touching the royal belt",
            ],
            "scripture_note": "2 Samuel 14:26 & 18:9 - Absalom's prized five-pound mane became his downfall when his head caught fast in the branches of a great oak!",
            "announcer": "Truth or Bluff! Why did Absalom only cut his hair once a year, and why did he have to?",
        },
        {
            "id": "tob_paul_trade",
            "prompt": "According to Acts 18:3, what hands-on trade did the Apostle Paul practice alongside Aquila and Priscilla in Corinth to pay his own way? ______",
            "truth": "Tentmaking (working with leather and goat-hair cloth)",
            "truth_keywords": ["tentmaking", "tentmaker", "tent makers", "tents"],
            "decoy": "Scroll-binding and parchment ink mixing",
            "bot_answers": [
                "Sandal-cobbling for Roman travelers",
                "Olive-press carpentry and wheel repair",
                "Weaving fishing nets for the Corinthian harbor",
            ],
            "scripture_note": "Acts 18:2-3 - Paul stayed and worked with Aquila and Priscilla because they were tentmakers by trade, modeling diligent bi-vocational ministry.",
            "announcer": "Truth or Bluff! What day job did Paul share with Aquila and Priscilla in Corinth?",
        },
    ],
    "survey_says": [
        {
            "id": "ss_pastor_conclusion",
            "prompt": "WE ASKED 100 CHURCHGOERS: What is the #1 unspoken thought when the pastor says \"And in conclusion...\" for the third time?",
            "decoy": "The Baptists are already at the restaurant taking our booth!",
            "bot_answers": [
                "My crockpot roast just went from 'medium-rare' to 'burnt offering'!",
                "That means we only have twenty-five minutes left!",
                "Even Paul let Eutychus go home eventually!",
            ],
            "scripture_note": "Ecclesiastes 3:1 - For everything there is a season, and a time for every matter under heaven - including lunch!",
            "announcer": "Double-Point Lightning Finale! Survey Says: What is the number one thought when the pastor says 'And in conclusion' for the third time?",
        },
        {
            "id": "ss_noah_question",
            "prompt": "WE ASKED 100 BIBLE READERS: If you could ask Noah ONE practical question about life inside the Ark, what would it be?",
            "decoy": "Where did you put the woodpeckers and termites so the boat didn't sink?!",
            "bot_answers": [
                "Whose turn was it to scoop the elephant deck on Day 39?",
                "How did you keep the lions from looking at the antelopes like a buffet?",
                "Did the roosters all crow at different time zones?",
            ],
            "scripture_note": "Genesis 6:14-22 - Noah did all that the Lord commanded him, down to every cubit and provision.",
            "announcer": "Double Points on the line! Survey Says: What is the number one practical question you would ask Noah about life on the Ark?",
        },
        {
            "id": "ss_sunday_school",
            "prompt": "WE ASKED 100 SUNDAY SCHOOL TEACHERS: What is the funniest answer a kid gives when they don't know the answer to a Bible question?",
            "decoy": "Say 'JESUS!' really loud with confidence and hope it wasn't 'Who betrayed the Lord?'",
            "bot_answers": [
                "Raise your hand to answer and then tell a ten-minute story about your dog!",
                "Guess 'Moses' for every Old Testament question and 'Paul' for the rest!",
                "Ask if we can have the goldfish crackers early!",
            ],
            "scripture_note": "Proverbs 22:6 - Train up a child in the way he should go - and enjoy the hilarious detours along the way!",
            "announcer": "Lightning Finale for Double Points! What is the classic Sunday School answer strategy?",
        },
        {
            "id": "ss_without_grumbling",
            "prompt": "WE ASKED 100 FAMILIES: Philippians 2:14 says \"Do all things without grumbling.\" What everyday chore puts that verse to the ultimate test?",
            "decoy": "Stepping on a stray LEGO barefoot at 10:00 PM while carrying laundry",
            "bot_answers": [
                "Unloading the dishwasher right after someone says 'Is that clean or dirty?'",
                "Untangling Christmas lights that were 'put away neatly' last January",
                "Cleaning out the back seat of the family van after a road trip",
            ],
            "scripture_note": "Philippians 2:14-15 - Doing even ordinary tasks without grumbling makes us shine as lights in the world!",
            "announcer": "Double-Point Finale! What everyday family chore tests 'Do all things without grumbling' the hardest?",
        },
    ],
}


_lock = threading.Lock()
_rooms: dict[str, dict[str, Any]] = {}
_tts_clips: dict[str, bytes] = {}


def _clean_text(raw: Any, max_len: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    return text[:max_len]


def _clean_code(raw: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(raw or "")).upper()[:8]


def _clean_initials(raw: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(raw or ""))[:3].upper() or "YOU"


def _normalize_compare(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def _matches_truth(text: str, prompt_item: dict[str, Any]) -> bool:
    """True when a player's submitted bluff in truth_or_bluff gives away the real answer."""
    if prompt_item.get("kind") != "truth_or_bluff":
        return False
    norm = _normalize_compare(text)
    if not norm:
        return False
    truth_norm = _normalize_compare(str(prompt_item.get("truth") or ""))
    if truth_norm and (norm == truth_norm or (len(norm) >= 6 and norm in truth_norm)):
        return True
    for kw in prompt_item.get("truth_keywords") or []:
        kw_norm = _normalize_compare(str(kw))
        if kw_norm and kw_norm in norm:
            return True
    return False


def _humans_unlocked(room: dict[str, Any]) -> list[dict[str, Any]]:
    """Human contestants (not AI scribes, not spectators)."""
    return [p for p in room["players"].values() if not p.get("is_bot") and not p.get("spectator")]


def _abandoned(room: dict[str, Any], now: float) -> bool:
    """No human contestant left, or none has checked in for ROOM_IDLE_SECONDS."""
    humans = _humans_unlocked(room)
    if not humans:
        # A room someone just opened (say, from a TV or as a spectator) gets a
        # moment for its first contestant to arrive.
        return now - float(room.get("created_at") or now) > ROOM_EMPTY_GRACE_SECONDS
    last = max(float(p.get("last_seen") or p.get("joined_at") or 0) for p in humans)
    return now - last > ROOM_IDLE_SECONDS


def _prune_expired_unlocked(now: float) -> None:
    stale = [
        code for code, r in _rooms.items()
        if now - float(r.get("updated_at") or 0) > ROOM_TTL_SECONDS or _abandoned(r, now)
    ]
    for code in stale:
        _rooms.pop(code, None)


def _pick_room_code_unlocked() -> str:
    available = [w for w in ROOM_WORDS if w not in _rooms]
    if available:
        return random.choice(available)
    for _ in range(50):
        candidate = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=4))
        if candidate not in _rooms:
            return candidate
    return f"GW{random.randint(10, 99)}"


def _select_prompt_unlocked(room: dict[str, Any], kind: str) -> dict[str, Any]:
    used: set[str] = set(room.get("used_prompt_ids") or [])
    pool = PROMPT_BANK.get(kind) or PROMPT_BANK["blank_o_matic"]
    fresh = [p for p in pool if p["id"] not in used]
    chosen = dict(random.choice(fresh if fresh else pool))
    chosen["kind"] = kind
    room.setdefault("used_prompt_ids", []).append(chosen["id"])
    return chosen


def _ensure_bots_unlocked(room: dict[str, Any]) -> None:
    """Ensure at least 3 active contestants by adding friendly Biblical AI scribes if needed."""
    players = room["players"]
    humans = [p for p in players.values() if not p.get("is_bot") and not p.get("spectator")]
    bots = [p for p in players.values() if p.get("is_bot")]
    target_total = 3 if len(humans) <= 2 else len(humans)
    needed_bots = max(0, target_total - len(humans))
    # Remove excess bots if more humans joined
    if len(bots) > needed_bots:
        for b in bots[needed_bots:]:
            players.pop(b["id"], None)
    elif len(bots) < needed_bots:
        for comp in AI_COMPANIONS:
            if len([p for p in players.values() if p.get("is_bot")]) >= needed_bots:
                break
            if comp["id"] not in players:
                players[comp["id"]] = {
                    "id": comp["id"],
                    "name": comp["name"],
                    "initials": comp["initials"],
                    "avatar": comp["avatar"],
                    "score": 0,
                    "round_delta": 0,
                    "is_bot": True,
                    "spectator": False,
                    "joined_at": time.time(),
                }


def _active_contestants_unlocked(room: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in room["players"].values() if not p.get("spectator")]


def _start_round_unlocked(room: dict[str, Any], round_num: int, custom_prompt: dict[str, Any] | None = None) -> None:
    now = time.time()
    schedule_idx = max(0, min(len(MATCH_ROUND_SCHEDULE) - 1, round_num - 1))
    kind = MATCH_ROUND_SCHEDULE[schedule_idx]
    meta = ROUND_META[kind]
    prompt_item = custom_prompt if custom_prompt else _select_prompt_unlocked(room, kind)
    prompt_item["kind"] = kind

    for p in room["players"].values():
        p["round_delta"] = 0

    submissions: dict[str, dict[str, Any]] = {}
    # Pre-populate AI companion submissions so they are ready when humans finish.
    bot_pool = list(prompt_item.get("bot_answers") or [])
    random.shuffle(bot_pool)
    for idx, bot in enumerate(p for p in room["players"].values() if p.get("is_bot")):
        text = bot_pool[idx % len(bot_pool)] if bot_pool else f"{bot['name']}'s wise quip #{round_num}"
        submissions[bot["id"]] = {
            "player_id": bot["id"],
            "text": text,
            "is_decoy": False,
            "submitted_at": now + 0.5,
        }

    room["round"] = round_num
    room["total_rounds"] = TOTAL_ROUNDS
    room["phase"] = "writing"
    room["phase_started_at"] = now
    room["phase_deadline"] = now + WRITE_SECONDS + PROMPT_READ_SECONDS + _stage_delay(room)
    room["current_prompt"] = {
        "id": prompt_item["id"],
        "kind": kind,
        "title": meta["title"],
        "badge": meta["badge"],
        "icon": meta["icon"],
        "multiplier": meta["multiplier"],
        "instructions": meta["instructions"],
        "prompt": prompt_item["prompt"],
        "subprompt": prompt_item.get("subprompt", ""),
        "scene": prompt_item.get("scene", ""),
        "speaker": prompt_item.get("speaker", ""),
        "decoy": prompt_item["decoy"],
        "truth": prompt_item.get("truth", ""),
        "truth_keywords": prompt_item.get("truth_keywords", []),
        "scripture_note": prompt_item.get("scripture_note", ""),
        "announcer": prompt_item.get("announcer") or f"Round {round_num}: {meta['title']}! {prompt_item['prompt']}",
    }
    room["submissions"] = submissions
    room["options"] = []
    room["votes"] = {}
    room["reveal"] = None
    room["updated_at"] = now


def _build_voting_options_unlocked(room: dict[str, Any]) -> None:
    now = time.time()
    prompt = room["current_prompt"]
    kind = prompt["kind"]
    submissions: dict[str, dict[str, Any]] = room["submissions"]

    # If any human contestant didn't submit before timeout, give them the Scribe's Decoy
    # so they still participate in the round.
    for p in _active_contestants_unlocked(room):
        if p["id"] not in submissions:
            submissions[p["id"]] = {
                "player_id": p["id"],
                "text": prompt["decoy"],
                "is_decoy": True,
                "auto_filled": True,
                "submitted_at": now,
            }

    options: list[dict[str, Any]] = []
    decoy_authors: list[str] = []

    for pid, sub in submissions.items():
        if sub.get("is_decoy"):
            decoy_authors.append(pid)
            continue
        sub_text = sub["text"]
        opt_hash = hashlib.sha1(f"{pid}:{sub_text}".encode()).hexdigest()[:8]
        options.append(
            {
                "option_id": f"opt_{opt_hash}",
                "text": sub_text,
                "kind": "player",
                "authors": [pid],
                "voters": [],
            }
        )

    # Always include the Scribe's Trap / Decoy option (credits any players who chose Decoy,
    # or acts as the pure House Trap if no one chose it).
    options.append(
        {
            "option_id": "opt_house_decoy",
            "text": prompt["decoy"],
            "kind": "decoy",
            "authors": decoy_authors,
            "voters": [],
        }
    )

    # In Truth or Bluff, also include the True Biblical Answer!
    if kind == "truth_or_bluff" and prompt.get("truth"):
        options.append(
            {
                "option_id": "opt_biblical_truth",
                "text": prompt["truth"],
                "kind": "truth",
                "authors": [],
                "voters": [],
            }
        )

    # Deterministic shuffle seeded by room code + round so order stays stable.
    rng = random.Random(f"{room['code']}:{room['round']}:{len(options)}")
    rng.shuffle(options)

    # Bots cast their votes automatically (never voting for their own answer).
    votes: dict[str, str] = {}
    for bot in (p for p in room["players"].values() if p.get("is_bot")):
        valid = [o for o in options if bot["id"] not in (o.get("authors") or [])]
        if valid:
            # Bots prefer player answers or truth 80% of the time so humans get votes!
            non_decoy = [o for o in valid if o["kind"] != "decoy"]
            pick_pool = non_decoy if (non_decoy and rng.random() < 0.82) else valid
            chosen = rng.choice(pick_pool)
            votes[bot["id"]] = chosen["option_id"]

    room["options"] = options
    room["votes"] = votes
    room["phase"] = "voting"
    room["phase_started_at"] = now
    room["phase_deadline"] = now + VOTE_SECONDS + _stage_delay(room)
    room["updated_at"] = now


def _score_round_unlocked(room: dict[str, Any]) -> None:
    now = time.time()
    prompt = room["current_prompt"]
    kind = prompt["kind"]
    mult = int(prompt.get("multiplier") or 1)
    players = room["players"]
    options = room["options"]
    votes: dict[str, str] = room["votes"]

    opt_by_id = {o["option_id"]: o for o in options}
    for o in options:
        o["voters"] = []
        o["points_awarded"] = 0

    for pid in players:
        players[pid]["round_delta"] = 0

    events: list[dict[str, Any]] = []

    for voter_id, opt_id in votes.items():
        opt = opt_by_id.get(opt_id)
        voter = players.get(voter_id)
        if not opt or not voter:
            continue
        opt["voters"].append(
            {
                "id": voter_id,
                "name": voter["name"],
                "initials": voter["initials"],
                "avatar": voter.get("avatar", "⭐"),
            }
        )

        if opt["kind"] == "truth":
            # Truth or Bluff: Voter spotted the real Scriptural truth!
            gain = 1000 * mult
            voter["score"] += gain
            voter["round_delta"] += gain
            opt["points_awarded"] += gain
            events.append(
                {
                    "type": "truth_found",
                    "player_id": voter_id,
                    "player_name": voter["name"],
                    "delta": gain,
                    "text": f"{voter['name']} spotted the Biblical Truth! (+{gain:,} pts)",
                }
            )
        elif opt["kind"] == "decoy":
            # Voter fell for the Scribe's Trap!
            penalty = -250 * mult
            voter["score"] = max(0, voter["score"] + penalty)
            voter["round_delta"] += penalty
            events.append(
                {
                    "type": "trap_fell",
                    "player_id": voter_id,
                    "player_name": voter["name"],
                    "delta": penalty,
                    "text": f"📜 {voter['name']} fell for the Scribe's Trap! ({penalty:,} pts)",
                }
            )
            # Any player who deliberately played the Scribe's Decoy earns +500 * mult per vote!
            for author_id in opt.get("authors") or []:
                if author_id == voter_id or author_id not in players:
                    continue
                author = players[author_id]
                decoy_gain = 500 * mult
                author["score"] += decoy_gain
                author["round_delta"] += decoy_gain
                opt["points_awarded"] += decoy_gain
                events.append(
                    {
                        "type": "decoy_bait",
                        "player_id": author_id,
                        "player_name": author["name"],
                        "delta": decoy_gain,
                        "text": f"{author['name']}'s Scribe Decoy tricked {voter['name']}! (+{decoy_gain:,} pts)",
                    }
                )
        elif opt["kind"] == "player":
            per_vote = (750 if kind == "truth_or_bluff" else 1000) * mult
            for author_id in opt.get("authors") or []:
                if author_id == voter_id or author_id not in players:
                    continue
                author = players[author_id]
                author["score"] += per_vote
                author["round_delta"] += per_vote
                opt["points_awarded"] += per_vote
                label = "bluff fooled" if kind == "truth_or_bluff" else "got a vote from"
                events.append(
                    {
                        "type": "vote_earned",
                        "player_id": author_id,
                        "player_name": author["name"],
                        "delta": per_vote,
                        "text": f"{author['name']} {label} {voter['name']}! (+{per_vote:,} pts)",
                    }
                )

    # Award a +200 * mult "Best Word" bonus to the player option with the most votes (if > 0)
    player_opts = [o for o in options if o["kind"] == "player" and len(o.get("voters") or []) > 0]
    round_winner_names: list[str] = []
    if player_opts:
        top_votes = max(len(o["voters"]) for o in player_opts)
        bonus = 200 * mult
        for o in player_opts:
            if len(o["voters"]) == top_votes:
                o["is_round_winner"] = True
                for author_id in o.get("authors") or []:
                    if author_id in players:
                        players[author_id]["score"] += bonus
                        players[author_id]["round_delta"] += bonus
                        round_winner_names.append(players[author_id]["name"])

    # Build announcer summary for the reveal screen
    if round_winner_names:
        winner_str = " and ".join(round_winner_names)
        announcer_reveal = f"The votes are in! {winner_str} takes the round bonus!"
    elif any(e["type"] == "truth_found" for e in events):
        announcer_reveal = "The truth is revealed! Let's see who spotted the real Scripture!"
    else:
        announcer_reveal = "The Scribe's Trap caught the room! Look at those scores!"

    room["reveal"] = {
        "events": events,
        "round_winners": round_winner_names,
        "scripture_note": prompt.get("scripture_note", ""),
        "announcer": announcer_reveal,
    }
    room["phase"] = "reveal"
    room["phase_started_at"] = now
    room["phase_deadline"] = (now + REVEAL_SECONDS + REVEAL_PER_ANSWER_SECONDS * len(room.get("options") or [])
                              + REVEAL_LINGER_SECONDS + _stage_delay(room))
    room["updated_at"] = now
    _prefetch_next_prompt_unlocked(room)


# --- The host's patter: what makes it sound like a game show, not a narrator ---
#
# Lines are picked per match and round (seeded), so the TV and every phone say
# the same thing, and a replayed round does not repeat the last match's jokes.

HOST_OPENERS = (
    "Welcome back, word-slingers!",
    "Phones up, scribes! Here we go!",
    "Oh, I like this one!",
    "Ready? Of course you are!",
    "Settle in, saints. This one's a doozy!",
    "Keep it clean, keep it clever!",
)
HOST_VOTING = (
    "All the scrolls are in! Pick your favorite, and watch out for the Scribe's Trap!",
    "Pencils down! Well, phones down. Vote for the best one, but not your own!",
    "The answers are in! One of these is a trap. Choose wisely!",
    "Time to vote! Which one made you laugh the loudest?",
)
HOST_DRUMROLL = (
    "Drumroll, please...",
    "And now... the moment of truth!",
    "Let's see who fooled whom...",
    "The votes are counted. Brace yourselves...",
)
HOST_TRAPPED = (
    "Oh no! The Scribe's Trap strikes again! Somebody wasn't reading carefully!",
    "Gotcha! The Scribe's Trap caught a few of you! Look at those scores!",
    "Ha! That one was the trap all along!",
)
HOST_FINALE = (
    "{name} wins the Golden Scroll with {score} points! Well played, everyone!",
    "Ladies and gentlemen, your champion: {name}, with {score} points! Give it up!",
    "And the Golden Scroll goes to... {name}! {score} points! What a match!",
)


#: Clean dad jokes for all ages, Bible-flavoured: (setup, punchline).
DAD_JOKES: tuple[tuple[str, str], ...] = (
    ("Who was the greatest comedian in the Bible?", "Samson. He brought the house down!"),
    ("What kind of car did the disciples drive?", "A Honda. They were all in one Accord!"),
    ("Why didn't anyone play cards on the Ark?", "Because Noah was standing on the deck!"),
    ("What did Adam say the day before Christmas?", "It's Christmas, Eve!"),
    ("Where is the first math problem in the Bible?", "When God told Adam and Eve to go forth and multiply!"),
    ("How does Moses make his coffee?", "Hebrews it!"),
    ("Who was the fastest runner in history?", "Adam. He was first in the human race!"),
    ("What kind of lights did Noah put on the Ark?", "Floodlights!"),
    ("Why did the man get so excited reading his Bible?", "He thought he'd found a Job!"),
    ("Who was the shortest man in the Bible?", "Nehemiah. Knee-high-miah!"),
    ("How do we know Peter was a rich fisherman?", "By his net income!"),
    ("Which animal could Noah never trust?", "The cheetah!"),
    ("Why didn't Noah go fishing?", "He only had two worms!"),
    ("Which Bible character had no parents?", "Joshua, son of Nun!"),
    ("Why was Goliath so surprised when David's stone hit him?", "Because the thought had never entered his head before!"),
    ("Where did Noah keep his bees?", "In the Ark hives!"),
    ("How long did Cain hold a grudge against his brother?", "As long as he was Abel!"),
    ("Who was the first tennis player in the Bible?", "Joseph. He served in Pharaoh's court!"),
    ("What do you call a sheep with no legs?", "A cloud!"),
    ("Why did Noah scold the chickens on the Ark?", "They were using fowl language!"),
    ("What's the best way to study the Bible?", "Just Luke into it!"),
    ("Why did the donkey get the best seat on Palm Sunday?", "Because he was the one carrying the King!"),
    ("What did the whale say after swallowing Jonah?", "I can't stomach this guy!"),
    ("Why did Noah make the woodpeckers sit still?", "So they wouldn't peck a hole in the Ark!"),
    ("What do you call a lamb covered in chocolate?", "A candy baa!"),
    ("What's a sheep's favourite part of the Sermon on the Mount?", "The Baa-atitudes!"),
)


def joke_for(room: dict[str, Any], salt: str) -> tuple[str, str]:
    seed = f"{room.get('match_id', '') or room.get('code', '')}:{room.get('round', 0)}:{salt}"
    return random.Random(seed).choice(DAD_JOKES)


#: While players write and vote, the host keeps talking: a remark about the
#: round, a joke or a nudge, the way a game-show host fills dead air.
HOST_CHATTER: dict[str, tuple[str, ...]] = {
    "blank_o_matic": (
        "Fill in that blank! Funny beats right, but right and funny wins the day!",
        "I've seen some answers in my time, folks. Surprise me!",
        "Remember, somebody has to vote for it. Make it good!",
    ),
    "extra_extra": (
        "Every great headline needs a pun. I'm just saying!",
        "Short, punchy, front-page worthy. Think like a newsboy!",
        "Read all about it... as soon as you write it!",
    ),
    "truth_or_bluff": (
        "Here's the trick: make your fake sound more true than the truth!",
        "One of these answers will be the real one from Scripture. Can you out-bluff it?",
        "Bluffers, put on your best poker face!",
    ),
    "sub_the_scroll": (
        "What are they saying? Only you know!",
        "Put words in their mouths. Kind words, please!",
        "I'd love to hear this conversation. Go on, write it!",
    ),
    "survey_says": (
        "Double points, people! This is where champions are made!",
        "Think like the crowd. What would most folks say?",
        "Last round! Everything's on the line!",
    ),
    "voting": (
        "Read them carefully. One of those is the Scribe's Trap!",
        "No voting for yourself. I'm watching!",
        "Go with your gut. Or your funny bone!",
    ),
}
HOST_HURRY = ("Ten seconds left!", "Ten seconds! Finish that thought!", "Hurry, hurry! Ten seconds!")


def chatter_for(room: dict[str, Any]) -> list[dict[str, Any]]:
    """Lines to say during the current phase: ``[{"at": seconds into it, "text"}]``.

    Spaced through the phase and kept clear of its first lines (the prompt is
    still being read) and its last ten seconds (the hurry call).
    """
    phase = room.get("phase")
    if phase not in ("writing", "voting"):
        return []
    kind = "voting" if phase == "voting" else str((room.get("current_prompt") or {}).get("kind") or "blank_o_matic")
    pool = HOST_CHATTER.get(kind) or HOST_CHATTER["blank_o_matic"]
    rng = random.Random(f"{room.get('match_id', '')}:{room.get('round', 0)}:{phase}:chatter")
    lines = rng.sample(pool, k=min(2, len(pool)))
    if phase == "writing":
        setup, punch = joke_for(room, "writing")
        lines.append(f"While you think... {setup} ... {punch}")
    length = float(room.get("phase_deadline") or 0) - float(room.get("phase_started_at") or 0)
    delay = _stage_delay(room)
    start, end = 14.0 + delay, max(14.0 + delay, length - 14.0)
    step = (end - start) / max(1, len(lines))
    out = [{"at": round(start + i * step, 1), "text": t} for i, t in enumerate(lines) if start + i * step < end]
    if length - delay > 20:
        out.append({"at": round(length - 10.0, 1), "text": rng.choice(HOST_HURRY)})
    return out


#: In the lobby the TV tells a new joke this often while people join.
LOBBY_JOKE_SECONDS = 45.0


def _pick(room: dict[str, Any], pool: tuple[str, ...], salt: str) -> str:
    seed = f"{room.get('match_id', '')}:{room.get('round', 0)}:{salt}"
    return random.Random(seed).choice(pool)


def announce_line(room: dict[str, Any]) -> str:
    """What the announcer says as the room enters its current phase."""
    phase = room.get("phase")
    prompt = room.get("current_prompt") or {}
    if phase == "lobby":
        code = " ".join(str(room.get("code") or ""))
        return f"Welcome to Parable Party! Scan the code on the screen, or join room {code} in Jarvis."
    if phase == "writing":
        line = str(prompt.get("announcer") or "")
        if line.lower().startswith("extra"):
            return line  # the paper boy opens these
        return f"{_pick(room, HOST_OPENERS, 'open')} {line}".strip()
    if phase == "voting":
        return _pick(room, HOST_VOTING, "vote")
    if phase == "reveal":
        reveal = room.get("reveal") or {}
        if reveal.get("round_winners") or any(e.get("type") == "truth_found" for e in reveal.get("events") or []):
            result = str(reveal.get("announcer") or "")
        else:
            result = _pick(room, HOST_TRAPPED, "trap")
        line = f"{_pick(room, HOST_DRUMROLL, 'drum')} {result}".strip()
        if int(room.get("round") or 0) < int(room.get("total_rounds") or TOTAL_ROUNDS):
            setup, punch = joke_for(room, "between")
            line += f" And while we catch our breath... {setup} ... {punch}"
        return line
    if phase == "finished":
        players = [p for p in (room.get("players") or {}).values() if not p.get("spectator")]
        if not players:
            return "What a match!"
        best = max(players, key=lambda p: int(p.get("score") or 0))
        return _pick(room, HOST_FINALE, "finale").format(name=best.get("name", "Our winner"), score=best.get("score", 0))
    return ""


def _stage_delay(room: dict[str, Any], now: float | None = None) -> float:
    """Seconds the room's TV picture lags: the stream delay while a streamed stage is live."""
    now = time.time() if now is None else now
    seen = float(room.get("video_stage_seen_at") or 0.0)
    if not (seen and now - seen < STAGE_PRESENCE_SECONDS):
        return 0.0
    return float(room.get("stream_delay_s") or STREAM_STAGE_DELAY_S)


def set_stream_delay(code: str, seconds: float, *, player_id: str | None = None) -> float:
    """The host tunes how far behind their streamed TV is (it varies by TV and Wi-Fi).

    Applies from the next phase; only matters while the room's TV is a stream.
    """
    with _lock:
        room = _rooms.get(_clean_code(code))
        if room is None:
            raise LookupError(f"Room '{_clean_code(code)}' not found.")
        _require_host_unlocked(room, player_id)
        room["stream_delay_s"] = round(max(0.0, min(30.0, float(seconds))), 1)
        return room["stream_delay_s"]


def _check_auto_advance_unlocked(room: dict[str, Any]) -> None:
    now = time.time()
    phase = room.get("phase")
    deadline = float(room.get("phase_deadline") or 0)

    if phase == "writing":
        contestants = _active_contestants_unlocked(room)
        humans = [p for p in contestants if not p.get("is_bot")]
        all_submitted = bool(humans) and all(p["id"] in room["submissions"] for p in humans)
        if all_submitted or (deadline and now >= deadline):
            _build_voting_options_unlocked(room)
    elif phase == "voting":
        humans = [p for p in _active_contestants_unlocked(room) if not p.get("is_bot")]
        all_voted = bool(humans) and all(p["id"] in room["votes"] for p in humans)
        if all_voted or (deadline and now >= deadline):
            _score_round_unlocked(room)
    elif phase == "reveal" and deadline and now >= deadline:
        # The scores have been up long enough: the show goes on without waiting for the host.
        _advance_unlocked(room)


AI_PROMPT_KINDS = ("blank_o_matic", "survey_says")


def _next_kind(room: dict[str, Any]) -> tuple[int, str] | None:
    nxt = int(room.get("round") or 0) + 1
    if nxt > int(room.get("total_rounds") or TOTAL_ROUNDS):
        return None
    return nxt, MATCH_ROUND_SCHEDULE[max(0, min(len(MATCH_ROUND_SCHEDULE) - 1, nxt - 1))]


def _prefetch_next_prompt_unlocked(room: dict[str, Any]) -> None:
    """Write the next round's AI prompt while the scores are on screen.

    The language model is never called while the room lock is held: that lock
    serves every room, so a slow model would freeze every game's clock.
    """
    nxt = _next_kind(room)
    if not room.get("ai_prompts") or nxt is None or nxt[1] not in AI_PROMPT_KINDS:
        return
    code, round_now, (round_next, kind) = room["code"], room.get("round"), nxt

    def work() -> None:
        custom = _generate_ai_prompt(kind)
        if not custom:
            return
        with _lock:
            r = _rooms.get(code)
            if r is not None and r.get("round") == round_now and r.get("phase") == "reveal":
                r["next_prompt"] = {"round": round_next, "prompt": custom}

    threading.Thread(target=work, name=f"party-prompt-{code}", daemon=True).start()


def _advance_unlocked(room: dict[str, Any], custom: dict[str, Any] | None = None) -> None:
    """writing -> voting -> reveal -> next round (or finished) -> new match."""
    phase = room["phase"]
    if phase == "lobby":
        _ensure_bots_unlocked(room)
        _start_round_unlocked(room, 1, custom_prompt=custom)
    elif phase == "writing":
        _build_voting_options_unlocked(room)
    elif phase == "voting":
        _score_round_unlocked(room)
    elif phase == "reveal":
        nxt = _next_kind(room)
        if nxt is None:
            room["phase"] = "finished"
            room["updated_at"] = time.time()
        else:
            ready = room.pop("next_prompt", None) or {}
            if custom is None and ready.get("round") == nxt[0]:
                custom = ready.get("prompt")
            _start_round_unlocked(room, nxt[0], custom_prompt=custom)
    elif phase == "finished":
        for p in room["players"].values():
            p["score"] = 0
            p["round_delta"] = 0
        _ensure_bots_unlocked(room)
        room["match_id"] = f"{room['code']}-{int(time.time())}"
        _start_round_unlocked(room, 1, custom_prompt=custom)


def serialize_room(room: dict[str, Any], viewer_id: str | None = None) -> dict[str, Any]:
    """Build a JSON-safe snapshot of the room for a phone or TV screen."""
    now = time.time()
    phase = room["phase"]
    prompt = room.get("current_prompt") or {}
    submissions = room.get("submissions") or {}
    votes = room.get("votes") or {}

    players_list = sorted(
        room["players"].values(),
        key=lambda p: (-int(p.get("score") or 0), float(p.get("joined_at") or 0)),
    )
    public_players = [
        {
            "id": p["id"],
            "name": p["name"],
            "initials": p["initials"],
            "avatar": p.get("avatar", "⭐"),
            "score": int(p.get("score") or 0),
            "round_delta": int(p.get("round_delta") or 0),
            "is_bot": bool(p.get("is_bot")),
            "spectator": bool(p.get("spectator")),
            "submitted": p["id"] in submissions,
            "voted": p["id"] in votes,
            "is_host": p["id"] == room.get("host_id"),
            "connected": bool(p.get("is_bot")) or now - float(p.get("last_seen") or 0) <= PLAYER_AWAY_SECONDS,
        }
        for p in players_list
    ]

    # During voting, hide option authors and option kind (player vs decoy vs truth)
    # so nobody knows which answer is the Scribe's Trap or who wrote what!
    public_options = []
    for opt in room.get("options") or []:
        is_mine = bool(viewer_id and viewer_id in (opt.get("authors") or []))
        if phase == "voting":
            public_options.append(
                {
                    "option_id": opt["option_id"],
                    "text": opt["text"],
                    "is_mine": is_mine,
                }
            )
        else:
            author_names = [
                room["players"][aid]["name"]
                for aid in (opt.get("authors") or [])
                if aid in room["players"]
            ]
            public_options.append(
                {
                    "option_id": opt["option_id"],
                    "text": opt["text"],
                    "kind": opt["kind"],
                    "authors": opt.get("authors") or [],
                    "author_names": author_names,
                    "voters": opt.get("voters") or [],
                    "points_awarded": int(opt.get("points_awarded") or 0),
                    "is_round_winner": bool(opt.get("is_round_winner")),
                    "is_mine": is_mine,
                }
            )

    public_prompt = None
    if prompt:
        public_prompt = {
            "id": prompt.get("id"),
            "kind": prompt.get("kind"),
            "title": prompt.get("title"),
            "badge": prompt.get("badge"),
            "icon": prompt.get("icon"),
            "multiplier": prompt.get("multiplier", 1),
            "instructions": prompt.get("instructions", ""),
            "prompt": prompt.get("prompt", ""),
            "subprompt": prompt.get("subprompt", ""),
            "scene": prompt.get("scene", ""),
            "speaker": prompt.get("speaker", ""),
            "announcer": prompt.get("announcer", ""),
            "scripture_note": prompt.get("scripture_note", "") if phase in ("reveal", "finished") else "",
            "truth": prompt.get("truth", "") if phase in ("reveal", "finished") else "",
        }

    my_sub = submissions.get(viewer_id) if viewer_id else None
    my_vote = votes.get(viewer_id) if viewer_id else None
    remaining = max(0, round(float(room.get("phase_deadline") or now) - now, 1)) if phase in ("writing", "voting", "reveal") else 0.0

    return {
        "code": room["code"],
        "host_id": room.get("host_id"),
        "host_name": room.get("host_name", "Host"),
        "phase": phase,
        "round": int(room.get("round") or 0),
        "total_rounds": int(room.get("total_rounds") or TOTAL_ROUNDS),
        "seconds_remaining": remaining,
        "phase_seconds": round(float(room.get("phase_deadline") or now) - float(room.get("phase_started_at") or now), 1)
        if phase in ("writing", "voting", "reveal") else 0.0,
        "reveal_step_seconds": REVEAL_PER_ANSWER_SECONDS,
        "players": public_players,
        "prompt": public_prompt,
        "options": public_options,
        "reveal": room.get("reveal") if phase in ("reveal", "finished") else None,
        "my_submission": {"text": my_sub["text"], "is_decoy": bool(my_sub.get("is_decoy"))} if my_sub else None,
        "my_vote": my_vote,
        "you": viewer_id if viewer_id and viewer_id in room["players"] else None,
        "you_are_host": bool(viewer_id) and viewer_id == room.get("host_id"),
        "avatars": list(PLAYER_AVATARS),
        "match_id": room.get("match_id", ""),
        "voice": room.get("voice") or DEFAULT_ANNOUNCER_VOICE,
        "announcer_voice": room.get("voice") or DEFAULT_ANNOUNCER_VOICE,
        # A TV stage polled recently: it speaks the announcer, so phones stay quiet.
        "stage_connected": now - float(room.get("stage_seen_at") or 0) < STAGE_PRESENCE_SECONDS,
        # The TV is a video of the stage, this many seconds behind. Phones hold
        # each phase until it catches up (``phase_age_s`` avoids clock skew),
        # and the host phone speaks since a stream carries no announcer.
        "stage_delay_s": _stage_delay(room),
        "announce": announce_line(room),
        "chatter": chatter_for(room),
        # A dad joke for the lobby, a new one every LOBBY_JOKE_SECONDS while people join.
        "joke": dict(zip(("setup", "punch"), joke_for(room, f"lobby{int(now // LOBBY_JOKE_SECONDS)}"), strict=True))
        if phase == "lobby" else None,
        "music": bool(room.get("music", True)),
        "phase_age_s": round(max(0.0, now - float(room.get("phase_started_at") or now)), 2)
        if phase in ("writing", "voting", "reveal") else 0.0,
    }


def list_active_rooms() -> list[dict[str, Any]]:
    now = time.time()
    with _lock:
        _prune_expired_unlocked(now)
        out = []
        for r in sorted(_rooms.values(), key=lambda x: -float(x.get("updated_at") or 0)):
            humans = [p for p in r["players"].values() if not p.get("is_bot")]
            out.append(
                {
                    "code": r["code"],
                    "host_name": r.get("host_name", "Family"),
                    "phase": r["phase"],
                    "round": r.get("round", 0),
                    "total_rounds": r.get("total_rounds", TOTAL_ROUNDS),
                    "player_count": len(humans),
                    "players": [p["name"] for p in humans[:6]],
                    "stage_connected": now - float(r.get("stage_seen_at") or 0) < STAGE_PRESENCE_SECONDS,
                    "created_at": r.get("created_at"),
                }
            )
        return out


def create_or_join_room(
    *,
    code: str | None = None,
    player_id: str | None = None,
    name: str | None = None,
    initials: str | None = None,
    spectator: bool = False,
    create_if_missing: bool = True,
    voice: str | None = None,
    avatar: str | None = None,
) -> dict[str, Any]:
    now = time.time()
    clean_code = _clean_code(code)
    clean_name = _clean_text(name, 24) or _clean_initials(initials)
    clean_init = _clean_initials(initials or clean_name)
    pid = _clean_text(player_id, 64) or f"p_{clean_init}_{int(now * 1000) % 10000}"

    with _lock:
        _prune_expired_unlocked(now)
        room = _rooms.get(clean_code) if clean_code else None
        if room is None:
            if not create_if_missing and clean_code:
                raise LookupError(f"Room '{clean_code}' not found.")
            room_code = clean_code if (clean_code and len(clean_code) >= 3) else _pick_room_code_unlocked()
            room = {
                "code": room_code,
                "host_id": pid,
                "host_name": clean_name,
                "phase": "lobby",
                "round": 0,
                "total_rounds": TOTAL_ROUNDS,
                "players": {},
                "used_prompt_ids": [],
                "submissions": {},
                "options": [],
                "votes": {},
                "reveal": None,
                "match_id": f"{room_code}-{int(now)}",
                "voice": _clean_text(voice, 64) or DEFAULT_ANNOUNCER_VOICE,
                "created_at": now,
                "updated_at": now,
            }
            _rooms[room_code] = room
        elif voice and _clean_text(voice, 64):
            room["voice"] = _clean_text(voice, 64)

        players = room["players"]
        human_contestants = [p for p in players.values() if not p.get("is_bot") and not p.get("spectator")]
        is_spec = bool(spectator or (pid not in players and len(human_contestants) >= 8))

        taken = {p.get("avatar") for p in players.values() if p["id"] != pid}
        wanted = avatar if avatar in PLAYER_AVATARS and avatar not in taken else ""
        if pid not in players:
            avatar = wanted or next((a for a in PLAYER_AVATARS if a not in taken), PLAYER_AVATARS[0])
            players[pid] = {
                "id": pid,
                "name": clean_name,
                "initials": clean_init,
                "avatar": "👀" if is_spec else avatar,
                "score": 0,
                "round_delta": 0,
                "is_bot": False,
                "spectator": is_spec,
                "joined_at": now,
                "last_seen": now,
            }
        else:
            players[pid]["name"] = clean_name
            players[pid]["initials"] = clean_init
            players[pid]["last_seen"] = now
            if wanted and not players[pid].get("spectator"):
                players[pid]["avatar"] = wanted
            if not spectator and players[pid].get("spectator"):
                players[pid]["spectator"] = False
                if players[pid].get("avatar") == "👀":
                    players[pid]["avatar"] = PLAYER_AVATARS[len(human_contestants) % len(PLAYER_AVATARS)]

        room["updated_at"] = now
        _check_auto_advance_unlocked(room)
        return {"player_id": pid, "room": serialize_room(room, viewer_id=pid)}


def _touch_unlocked(room: dict[str, Any], viewer_id: str | None, now: float) -> None:
    """Note that a phone is still here, and hand the host role on if the host left."""
    players = room["players"]
    if viewer_id and viewer_id in players:
        players[viewer_id]["last_seen"] = now
    host = players.get(room.get("host_id") or "")
    host_gone = host is None or now - float(host.get("last_seen") or 0) > HOST_HANDOFF_SECONDS
    if not host_gone:
        return
    here = [
        p for p in players.values()
        if not p.get("is_bot") and now - float(p.get("last_seen") or 0) <= PLAYER_AWAY_SECONDS
    ]
    if here:
        heir = min(here, key=lambda p: float(p.get("joined_at") or 0))
        room["host_id"] = heir["id"]
        room["host_name"] = heir["name"]


def _hand_off_host_unlocked(room: dict[str, Any]) -> None:
    humans = sorted(_humans_unlocked(room), key=lambda p: float(p.get("joined_at") or 0))
    if humans:
        room["host_id"] = humans[0]["id"]
        room["host_name"] = humans[0]["name"]


def leave_room(code: str, *, player_id: str) -> dict[str, Any]:
    """A player leaves. The room closes when no human contestant is left.

    Their unscored answer and vote go with them, so a round never waits on, or
    credits, someone who is gone; the host role passes to the next player.
    """
    clean_code = _clean_code(code)
    pid = _clean_text(player_id, 64)
    with _lock:
        room = _rooms.get(clean_code)
        if room is None:
            raise LookupError(f"Room '{clean_code}' not found.")
        if pid not in room["players"]:
            return {"closed": False, "room": serialize_room(room)}
        room["players"].pop(pid, None)
        if room.get("phase") == "writing":
            (room.get("submissions") or {}).pop(pid, None)
        if room.get("phase") == "voting":
            (room.get("votes") or {}).pop(pid, None)
        if not _humans_unlocked(room):
            _rooms.pop(clean_code, None)
            return {"closed": True, "room": None}
        if room.get("host_id") == pid:
            _hand_off_host_unlocked(room)
        room["updated_at"] = time.time()
        _check_auto_advance_unlocked(room)
        return {"closed": False, "room": serialize_room(room)}


def end_room(code: str, *, player_id: str | None = None, admin: bool = False) -> None:
    """Close a room for everyone: its host, or an arcade admin."""
    clean_code = _clean_code(code)
    with _lock:
        room = _rooms.get(clean_code)
        if room is None:
            raise LookupError(f"Room '{clean_code}' not found.")
        if not admin:
            pid = _clean_text(player_id, 64)
            if not pid or pid != room.get("host_id"):
                raise PermissionError("Only the host or an admin can end the game.")
        _rooms.pop(clean_code, None)


def _require_host_unlocked(room: dict[str, Any], player_id: str | None) -> None:
    """Host-only controls. A call without a player id is the timer/automation."""
    pid = _clean_text(player_id, 64)
    if pid and pid != room.get("host_id"):
        raise PermissionError("Only the host can do that.")


def set_room_voice(code: str, voice: str) -> dict[str, Any]:
    clean_code = _clean_code(code)
    clean_voice = _clean_text(voice, 64) or DEFAULT_ANNOUNCER_VOICE
    with _lock:
        room = _rooms.get(clean_code)
        if room is None:
            raise LookupError(f"Room '{clean_code}' not found.")
        room["voice"] = clean_voice
        room["updated_at"] = time.time()
        return serialize_room(room)


def set_room_music(code: str, on: bool, *, player_id: str | None = None) -> dict[str, Any]:
    """Host only: game-show music on or off for the room."""
    with _lock:
        room = _rooms.get(_clean_code(code))
        if room is None:
            raise LookupError(f"Room '{_clean_code(code)}' not found.")
        _require_host_unlocked(room, player_id)
        room["music"] = bool(on)
        room["updated_at"] = time.time()
        return serialize_room(room, viewer_id=player_id)


def get_room_state(code: str, viewer_id: str | None = None, *, stage: bool = False,
                   video: bool = False) -> dict[str, Any]:
    """A room snapshot. ``stage=True`` is the TV stage polling, which marks it live;
    ``video`` says that stage is being rendered into a video stream."""
    now = time.time()
    clean_code = _clean_code(code)
    with _lock:
        _prune_expired_unlocked(now)
        room = _rooms.get(clean_code)
        if room is None:
            raise LookupError(f"Room '{clean_code}' not found.")
        if stage:
            room["stage_seen_at"] = now
            if video:
                room["video_stage_seen_at"] = now
        _touch_unlocked(room, viewer_id, now)
        _check_auto_advance_unlocked(room)
        return serialize_room(room, viewer_id=viewer_id)


def start_match(code: str, *, use_ai_prompt: bool = False, player_id: str | None = None) -> dict[str, Any]:
    clean_code = _clean_code(code)
    with _lock:
        room = _rooms.get(clean_code)
        if room is None:
            raise LookupError(f"Room '{clean_code}' not found.")
        _require_host_unlocked(room, player_id)
    custom = _generate_ai_prompt("blank_o_matic") if use_ai_prompt else None
    with _lock:
        room = _rooms.get(clean_code)
        if room is None:
            raise LookupError(f"Room '{clean_code}' not found.")
        # Reset scores for a fresh 7-round match
        for p in list(room["players"].values()):
            if p.get("is_bot"):
                room["players"].pop(p["id"], None)
            else:
                p["score"] = 0
                p["round_delta"] = 0
        _ensure_bots_unlocked(room)
        room["match_id"] = f"{clean_code}-{int(time.time())}"
        room["ai_prompts"] = bool(use_ai_prompt)
        room.pop("next_prompt", None)
        _start_round_unlocked(room, 1, custom_prompt=custom)
        return serialize_room(room, viewer_id=_clean_text(player_id, 64) or None)


def submit_answer(
    code: str,
    *,
    player_id: str,
    text: str = "",
    use_decoy: bool = False,
) -> dict[str, Any]:
    clean_code = _clean_code(code)
    pid = _clean_text(player_id, 64)
    clean_ans = _clean_text(text, 120)
    with _lock:
        room = _rooms.get(clean_code)
        if room is None:
            raise LookupError(f"Room '{clean_code}' not found.")
        _check_auto_advance_unlocked(room)
        if room["phase"] != "writing":
            raise ValueError("This round's writing phase has already closed.")
        if pid not in room["players"]:
            raise ValueError("Join the room before submitting an answer.")
        if room["players"][pid].get("spectator"):
            room["players"][pid]["spectator"] = False
            if room["players"][pid].get("avatar") == "👀":
                room["players"][pid]["avatar"] = "👑"
        prompt = room["current_prompt"]
        if use_decoy:
            room["submissions"][pid] = {
                "player_id": pid,
                "text": prompt["decoy"],
                "is_decoy": True,
                "submitted_at": time.time(),
            }
        else:
            if not clean_ans:
                raise ValueError("Type an answer first, or tap 'Play Scribe's Decoy'!")
            if _matches_truth(clean_ans, prompt):
                raise ValueError("🎉 You know the real Biblical truth! Now write a FAKE bluff to trick your family!")
            if _normalize_compare(clean_ans) == _normalize_compare(prompt.get("decoy", "")):
                use_decoy = True
            room["submissions"][pid] = {
                "player_id": pid,
                "text": clean_ans,
                "is_decoy": use_decoy,
                "submitted_at": time.time(),
            }
        room["updated_at"] = time.time()
        _check_auto_advance_unlocked(room)
        return serialize_room(room, viewer_id=pid)


def submit_vote(code: str, *, player_id: str, option_id: str) -> dict[str, Any]:
    clean_code = _clean_code(code)
    pid = _clean_text(player_id, 64)
    oid = _clean_text(option_id, 64)
    with _lock:
        room = _rooms.get(clean_code)
        if room is None:
            raise LookupError(f"Room '{clean_code}' not found.")
        _check_auto_advance_unlocked(room)
        if room["phase"] != "voting":
            raise ValueError("Voting is not open right now.")
        if pid not in room["players"]:
            raise ValueError("Join the room to vote.")
        target = next((o for o in room["options"] if o["option_id"] == oid), None)
        if target is None:
            raise ValueError("Unknown answer option.")
        if pid in (target.get("authors") or []):
            raise ValueError("You cannot vote for your own answer!")
        room["votes"][pid] = oid
        room["updated_at"] = time.time()
        _check_auto_advance_unlocked(room)
        return serialize_room(room, viewer_id=pid)


def advance_phase(code: str, *, use_ai_prompt: bool = False, player_id: str | None = None) -> dict[str, Any]:
    """The host moves the show on: writing -> voting -> reveal -> next round / finished."""
    clean_code = _clean_code(code)
    with _lock:
        room = _rooms.get(clean_code)
        if room is None:
            raise LookupError(f"Room '{clean_code}' not found.")
        _require_host_unlocked(room, player_id)
        if use_ai_prompt:
            room["ai_prompts"] = True
        phase, round_now = room["phase"], room.get("round")
        nxt = _next_kind(room) if phase == "reveal" else None
        prefetched = (room.get("next_prompt") or {}).get("round") == (nxt[0] if nxt else None)
        if not (use_ai_prompt and nxt and nxt[1] in AI_PROMPT_KINDS and not prefetched):
            _advance_unlocked(room)
            return serialize_room(room, viewer_id=_clean_text(player_id, 64) or None)
    # Write the prompt with the lock released, then advance if nothing moved on meanwhile.
    custom = _generate_ai_prompt(nxt[1])
    with _lock:
        room = _rooms.get(clean_code)
        if room is None:
            raise LookupError(f"Room '{clean_code}' not found.")
        if room["phase"] == phase and room.get("round") == round_now:
            _advance_unlocked(room, custom=custom)
        return serialize_room(room, viewer_id=_clean_text(player_id, 64) or None)


def _generate_ai_prompt(kind: str) -> dict[str, Any] | None:
    """Optional live prompt + Scribe's Decoy generation via Alpaca Proxy."""
    if kind not in ("blank_o_matic", "survey_says"):
        return None
    system = (
        "You write warm, hilarious, family-safe Christian / Biblical worldview party game prompts "
        "in the style of 'Use Your Words' and Jackbox Quiplash. Return ONLY valid JSON with keys: "
        '{"prompt": "...with ______...", "decoy": "clever pre-written house trap answer", '
        '"bot_answers": ["answer1", "answer2", "answer3"], "scripture_note": "Book Ch:V - brief note"}.'
    )
    user = (
        "Write one fresh fill-in-the-blank prompt about everyday Christian family life, church potlucks, "
        "or a famous Bible story (Noah, Moses, David, Jonah, Peter, Paul, Solomon, Proverbs)."
    )
    payload = json.dumps(
        {
            "model": os.getenv("PARTY_LLM_MODEL", "default"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "think": False,
            "temperature": 0.85,
            "max_tokens": 260,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{PROXY_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        content = (((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S)
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            return None
        parsed = json.loads(m.group(0))
        prompt_str = _clean_text(parsed.get("prompt"), 220)
        decoy_str = _clean_text(parsed.get("decoy"), 100)
        if not prompt_str or not decoy_str:
            return None
        return {
            "id": f"ai_{int(time.time() * 1000)}",
            "kind": kind,
            "prompt": prompt_str,
            "decoy": decoy_str,
            "bot_answers": [_clean_text(x, 100) for x in (parsed.get("bot_answers") or []) if _clean_text(x, 100)][:3],
            "scripture_note": _clean_text(parsed.get("scripture_note"), 160) or "Colossians 4:6 - Let your speech always be gracious, seasoned with salt!",
        }
    except Exception:
        return None


# --- Alpaca TTS (Kokoro-82M Neural Voices + OpenVoice V2 Voice Clones) ---


def _audio_server_urls() -> list[str]:
    """Alpaca audio-server endpoints, from configuration only."""
    out: list[str] = []
    for u in (AUDIO_SERVER_URL, os.getenv("ALPACA_AUDIO_URL", "").rstrip("/")):
        if u and u not in out:
            out.append(u)
    return out


def list_tts_voices() -> dict[str, Any]:
    """Return Alpaca's Kokoro neural announcer presets + enrolled OpenVoice V2 voice clones."""
    voices: list[dict[str, Any]] = [dict(v) for v in CURATED_ANNOUNCER_VOICES]
    clones: list[dict[str, Any]] = []
    kokoro_raw: list[str] = []

    for base_url in _audio_server_urls():
        try:
            with urllib.request.urlopen(f"{base_url}/api/voices", timeout=2.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_clones = data.get("clones") if isinstance(data.get("clones"), list) else (data.get("voices") or [])
                    for c in raw_clones:
                        cid = str(c.get("id") or "").strip()
                        cname = str(c.get("name") or cid).strip()
                        if cid:
                            entry = {
                                "id": f"clone:{cid}",
                                "clone_id": cid,
                                "label": f"🗣️ {cname} (Alpaca Cloned Voice)",
                                "kind": "clone",
                            }
                            clones.append(entry)
                            voices.insert(0, entry)
                    break
        except Exception:
            continue

    for base_url in _audio_server_urls():
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    kokoro_raw = list((data.get("tts") or {}).get("voices") or [])
                    break
        except Exception:
            continue

    seen_ids = {v["id"] for v in voices}
    for kv in kokoro_raw:
        if kv not in seen_ids:
            voices.append({"id": kv, "label": f"🎙️ Kokoro · {kv}", "kind": "kokoro"})
            seen_ids.add(kv)

    return {
        "default_voice": DEFAULT_ANNOUNCER_VOICE,
        "voices": voices,
        "clones": clones,
        "kokoro_voices": kokoro_raw,
    }


def _decode_tts_response(raw: bytes) -> bytes | None:
    """Decode audio-server's JSON ``{"audio_b64": "..."}`` (or direct WAV bytes) into raw WAV bytes."""
    if not raw or len(raw) < 44:
        return None
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return raw
    try:
        parsed = json.loads(raw.decode("utf-8"))
        b64 = parsed.get("audio_b64") if isinstance(parsed, dict) else None
        if isinstance(b64, str) and b64:
            wav = base64.b64decode(b64)
            if len(wav) >= 44:
                return wav
    except Exception:
        return None
    return None


def _tts_request(text: str, voice: str, clone: str | None, speed: float) -> tuple[str, dict[str, Any]] | None:
    """The clip id and audio-server payload for one announcer line.

    A ``clone:<id>`` voice is sent as ``clone`` with **no** ``voice``: the
    audio server then pairs the Kokoro base voice that needs no pitch
    correction for that speaker. Naming a fixed base voice instead forces a
    phase vocoder over the whole line, which is what makes a clone sound sung.
    """
    clean = _clean_text(text, 280)
    if not clean:
        return None
    clean_voice = _clean_text(voice, 64) or DEFAULT_ANNOUNCER_VOICE
    clone_id = _clean_text(clone, 64) if clone else ""
    if clean_voice.startswith("clone:"):
        clone_id = clean_voice.split(":", 1)[1].strip()
        clean_voice = ""
    rate = max(0.7, min(1.5, float(speed)))
    payload: dict[str, Any] = {"text": clean, "speed": rate}
    if clone_id:
        payload["clone"] = clone_id
    if clean_voice:
        payload["voice"] = clean_voice
    cache_key = f"{clean_voice}:{clone_id}:{round(rate, 2)}:{clean}"
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16], payload


def synthesize_announcer_clip(
    text: str,
    voice: str = DEFAULT_ANNOUNCER_VOICE,
    *,
    clone: str | None = None,
    speed: float = 1.04,
) -> tuple[str, bytes] | None:
    """Announcer audio from Alpaca's Kokoro / OpenVoice audio-server as ``(clip_id, wav)``.

    Clips are cached by id so a TV stage and every phone that asks for the same
    line share one synthesis, and can stream it from ``/api/party/tts/<id>.wav``.
    """
    req_info = _tts_request(text, voice, clone, speed)
    if req_info is None:
        return None
    clip_id, payload_dict = req_info
    with _lock:
        cached = _tts_clips.get(clip_id)
    if cached:
        return clip_id, cached
    payload = json.dumps(payload_dict).encode("utf-8")
    # A cloned voice's first line pairs a base voice, which converts every
    # candidate once; give it the time it needs.
    timeout = 90.0 if "clone" in payload_dict else 30.0
    for base_url in _audio_server_urls():
        req = urllib.request.Request(
            f"{base_url}/api/tts", data=payload, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    continue
                wav = _decode_tts_response(resp.read())
        except Exception:
            continue
        if wav and len(wav) >= 44:
            with _lock:
                while len(_tts_clips) >= 96:
                    _tts_clips.pop(next(iter(_tts_clips)))
                _tts_clips[clip_id] = wav
            return clip_id, wav
    return None


def synthesize_announcer_wav(
    text: str,
    voice: str = DEFAULT_ANNOUNCER_VOICE,
    *,
    clone: str | None = None,
    speed: float = 1.04,
) -> bytes | None:
    """Announcer WAV bytes (see ``synthesize_announcer_clip``), or None when the audio server is unavailable."""
    clip = synthesize_announcer_clip(text, voice, clone=clone, speed=speed)
    return clip[1] if clip else None


def synthesize_show_line(text: str, voice: str = DEFAULT_ANNOUNCER_VOICE, *,
                         clone: str | None = None) -> tuple[str, bytes] | None:
    """A line as the show delivers it: ``(clip_id, wav)``.

    The announcer's part gets the broadcast treatment (``announcer.announcer_fx``);
    an opening "Extra! Extra!" is shouted by the paper boy first. Each Kokoro
    piece is cached on its own, and so is the finished line.
    """
    parts = announcer.split_show_line(text)
    if not parts:
        return None
    pieces: list[tuple[str, str, bytes]] = []
    for role, line in parts:
        if role == "paperboy":
            clip = synthesize_announcer_clip(line, announcer.PAPERBOY_VOICE, speed=announcer.PAPERBOY_SPEED)
        else:
            clip = synthesize_announcer_clip(line, voice, clone=clone, speed=announcer.ANNOUNCER_SPEED)
        if clip is None:
            if role == "paperboy":
                continue  # no newsboy today: the announcer still reads the line
            return None
        pieces.append((role, clip[0], clip[1]))
    if not pieces:
        return None
    show_id = hashlib.sha256(("show:v1:" + ":".join(f"{r}={c}" for r, c, _ in pieces)).encode()).hexdigest()[:16]
    with _lock:
        cached = _tts_clips.get(show_id)
    if cached:
        return show_id, cached
    try:
        wav = announcer.assemble([(r, w) for r, _, w in pieces])
    except Exception:
        wav = pieces[-1][2]  # an unreadable clip is still better spoken plainly
    with _lock:
        while len(_tts_clips) >= 96:
            _tts_clips.pop(next(iter(_tts_clips)))
        _tts_clips[show_id] = wav
    return show_id, wav


def get_cached_tts_wav(clip_id: str) -> bytes | None:
    clean_id = re.sub(r"[^a-f0-9]", "", (clip_id or "").lower())[:32]
    with _lock:
        return _tts_clips.get(clean_id)


# --- TV stage: discovered household screens (via Jarvis / SharedLLM) ---
#
# The arcade runs on a Docker bridge network, where mDNS and SSDP discovery
# cannot work and no TV can be reached by name. SharedLLM's execution service
# runs on the host network with Home Assistant, pychromecast, the LG/Samsung
# clients and a headless renderer, so it owns discovery and display; the arcade
# only names the room. Nothing here knows about any particular device.
#
# Two ways to reach it, preferred first:
#   1. JARVIS_GATEWAY_URL: the Jarvis gateway's /api/arcade/{tvs,cast}, the
#      same routes the Jarvis app uses. The gateway is the public front door, so
#      it is reachable wherever Jarvis is, and it builds the stage URL itself.
#   2. EXECUTION_SVC_URL: the execution service directly (same host only; it
#      is normally not exposed to the network).
# Both accept internal callers with JARVIS_INTERNAL_SECRET, which acts as the
# household, never as an administrator.

JARVIS_GATEWAY_URL = os.getenv("JARVIS_GATEWAY_URL", "").rstrip("/")


def _internal_secret() -> str:
    return (os.getenv("JARVIS_INTERNAL_SECRET") or os.getenv("EXECUTION_INTERNAL_SECRET") or "").strip()


def stage_url(base_url: str, code: str = "") -> str:
    """The live TV stage page for a room, at ``base_url`` (the address phones use)."""
    clean = _clean_code(code)
    base = (base_url or "").rstrip("/")
    return f"{base}/tv/{clean}" if clean else f"{base}/tv"


def _relay_target() -> tuple[str, str]:
    """``(kind, base_url)`` of the service that discovers and drives TVs."""
    if JARVIS_GATEWAY_URL:
        return "gateway", JARVIS_GATEWAY_URL
    if EXECUTION_SVC_URL:
        return "execution", EXECUTION_SVC_URL
    raise ConnectionError("Neither JARVIS_GATEWAY_URL nor EXECUTION_SVC_URL is set, so TV discovery is unavailable.")


def _relay_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 20.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    secret = _internal_secret()
    if secret:
        headers["X-Internal-Secret"] = secret
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        parsed = json.loads(resp.read().decode("utf-8") or "{}")
    return parsed if isinstance(parsed, dict) else {}


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    detail = ""
    with contextlib.suppress(Exception):
        detail = str(json.loads(exc.read().decode("utf-8")).get("detail") or "")
    if exc.code in (401, 403):
        return f"the TV service refused the arcade (HTTP {exc.code}); set JARVIS_INTERNAL_SECRET, or play inside Jarvis"
    return detail or f"the TV service returned HTTP {exc.code}"


def discover_screens(*, refresh: bool = False) -> dict[str, Any]:
    """Household screens that can show the stage, as discovered by Jarvis."""
    flag = "true" if refresh else "false"
    try:
        kind, base = _relay_target()
        if kind == "gateway":
            data = _relay_json("GET", f"{base}/api/arcade/tvs?refresh={flag}", timeout=30.0)
            screens = data.get("tvs")
        else:
            data = _relay_json("GET", f"{base}/execute/stage/screens?refresh={flag}", timeout=30.0)
            screens = data.get("screens")
    except urllib.error.HTTPError as exc:
        return {"success": False, "tvs": [], "error": f"Screen discovery failed: {_http_error_detail(exc)}"}
    except Exception as exc:
        return {"success": False, "tvs": [], "error": f"Screen discovery is unavailable: {exc}"}
    if data.get("success") is False and not screens:
        return {"success": False, "tvs": [], "error": str(data.get("error") or "Screen discovery failed")}
    return {"success": True, "tvs": list(screens or []), "home_assistant": bool(data.get("home_assistant"))}


def show_stage_on_screen(screen_id: str, url: str, *, room_code: str = "", method: str = "", stop: bool = False) -> dict[str, Any]:
    """Put a room's ``/tv`` stage page on a discovered screen, or take it off."""
    sid = _clean_text(screen_id, 160)
    if not sid:
        return {"success": False, "error": "screen_id is required"}
    try:
        kind, base = _relay_target()
        if kind == "gateway":
            # The gateway builds the stage URL from the arcade's public address.
            return _relay_json(
                "POST", f"{base}/api/arcade/cast",
                {"screen_id": sid, "room_code": _clean_code(room_code), "method": _clean_text(method, 32), "stop": stop},
                timeout=100.0,
            )
        action = "stop" if stop else "show"
        return _relay_json(
            "POST", f"{base}/execute/stage/{action}",
            {"screen_id": sid, "url": url, "method": _clean_text(method, 32)},
            timeout=100.0,
        )
    except urllib.error.HTTPError as exc:
        return {"success": False, "error": _http_error_detail(exc)}
    except Exception as exc:
        return {"success": False, "error": f"TV service unreachable: {exc}"}
