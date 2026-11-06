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
import collections
import contextlib
import hashlib
import json
import os
import random
import re
import secrets
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

DEFAULT_ANNOUNCER_VOICE = "bm_fable,bm_lewis,bm_george"  # chosen by ear, audition round three

CURATED_ANNOUNCER_VOICES: tuple[dict[str, str], ...] = (
    {
        "id": "bm_fable,bm_lewis,bm_george",
        "label": "🎩 The Sassy Brit (Game-Show Host)",
        "kind": "kokoro",
    },
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
#: A phone's clock and the network lag the server's by a second or two, so an
#: answer or vote sent as a phone shows 0 still counts: the phase stays open
#: this much longer than the deadline everyone sees (unless everyone is in).
SUBMIT_GRACE_SECONDS = 4.0
#: The longest announcer line the TTS route will voice (the longest real line is ~260).
TTS_MAX_CHARS = 320
#: The announcer clip cache is bounded by size as well as count.
TTS_CACHE_MAX_BYTES = 48 * 1024 * 1024
#: An answer that still arrives after voting opened replaces the house answer
#: it was given, as long as nobody has voted yet and voting is this fresh.
LATE_ANSWER_SECONDS = 10.0
#: The announcer reads the prompt (and a quip) as writing starts; that time is not the players'.
PROMPT_READ_SECONDS = 8
#: After the last answer is revealed, how long the scores stay up before the next round starts on its own.
REVEAL_LINGER_SECONDS = 16
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
            "prompt": "Noah's family finally stepped off the Ark after a whole year with two of every animal. The very first thing Mrs. Noah said was: \"______.\"",
            "decoy": "My review of this cruise: one star. Two of everything.",
            "bot_answers": [
                "Shem, we are not keeping the skunks. I don't care that they're cute.",
                "Somebody count the rabbits. Two got on. I see forty.",
                "Next vacation, we stay home. In a house. That does not moo.",
            ],
            "scripture_note": "Genesis 8:16-18 - After more than a year aboard the Ark, God brought Noah's whole household safely onto dry ground.",
            "announcer": "Round time! After a year on the Ark with two of every animal, Mrs. Noah finally steps onto dry land. What does she say first?",
        },
        {
            "id": "bom_solomon_prayer",
            "prompt": "King Solomon was the wisest man alive, but his most frequent dinner-table prayer was still: \"Lord, please help me ______.\"",
            "decoy": "remember where I parked the chariot",
            "bot_answers": [
                "stop at one honey cake. I literally wrote a proverb about this.",
                "look wise while I quietly look up what 'discernment' means",
                "deal with kids who say 'I'm not hungry' and then eat everyone's fries",
            ],
            "scripture_note": "1 Kings 3:9 - Solomon asked God for an understanding heart to discern between good and evil.",
            "announcer": "Fill in the blank! The wisest man who ever lived still needed a little help at the dinner table.",
        },
        {
            "id": "bom_paul_autoreply",
            "prompt": "If the Apostle Paul had a church group chat with Corinth, his pinned auto-reply would say: \"Grace and peace! Before you text me, please ______.\"",
            "decoy": "decide if this needs a text, a call, or a sixteen-chapter letter",
            "bot_answers": [
                "stop splitting into Team Paul and Team Apollos over the potluck menu",
                "remember I'm shipwrecked again, so replies may be slow and slightly damp",
                "read my last letter. All of it. Yes, even the really long sentences.",
            ],
            "scripture_note": "1 Corinthians 1:10 - Paul urged the church to agree with one another and be united in mind and purpose.",
            "announcer": "Imagine the Apostle Paul running a modern church group chat. What does his auto-reply say?",
        },
        {
            "id": "bom_proverbs_roof",
            "prompt": "Proverbs 21:9 says it is better to live on a corner of the roof than in a house shared with ______.",
            "decoy": "a little brother who just learned to whistle and only knows one note",
            "bot_answers": [
                "someone who eats the last cinnamon roll and puts the empty box back",
                "a dad guarding the thermostat like it's the Ark of the Covenant",
                "a rooster rehearsing for Peter's big scene at three in the morning",
            ],
            "scripture_note": "Proverbs 21:9 & 15:17 - Scripture prizes peace and kindness at home over luxury accompanied by strife.",
            "announcer": "A classic proverb gets a family update! It's better to live on the corner of the roof than in a house with... what?",
        },
        {
            "id": "bom_five_thousand_boy",
            "prompt": "After the Feeding of the 5,000, the boy who packed five loaves and two fish ran home and told his mom: \"Mom, you won't believe ______!\"",
            "decoy": "how many leftovers there were. We need twelve baskets and a bigger fridge.",
            "bot_answers": [
                "it, but I shared my lunch. On purpose. Write it on the calendar.",
                "five thousand people ate your fish sandwiches and nobody asked for ketchup",
                "it. I'm officially the most famous lunchbox in history.",
            ],
            "scripture_note": "John 6:9-13 - What seems small in our hands becomes abundant when offered to Jesus.",
            "announcer": "The boy with five loaves and two fish just got home! What does he tell his mom?",
        },
        {
            "id": "bom_jonah_brochure",
            "prompt": "The real reason Jonah bought a one-way ship ticket to Tarshish instead of going to Nineveh was because the Tarshish travel brochure promised ______.",
            "decoy": "zero fish, zero Ninevites, and a pool that is definitely not the ocean",
            "bot_answers": [
                "a relaxing cruise where nobody asks you to preach anything, ever",
                "a nap below deck so peaceful you could sleep through a storm. He did.",
                "free whale watching. He really should have read the fine print.",
            ],
            "scripture_note": "Jonah 1:3 - You can buy a ticket to Tarshish, but you can never outrun the Lord's mercy and call!",
            "announcer": "Why did Tarshish look so good to Jonah? Fill in the blank from the travel brochure!",
        },
        {
            "id": "bom_david_potluck",
            "prompt": "If King David wrote a Psalm right before a Sunday church potluck, verse 1 would read: \"O Lord, shield my plate from ______.\"",
            "decoy": "the Jell-O salad with carrots in it, for reasons no one will explain",
            "bot_answers": [
                "casserole number seven, which is casserole number three in a disguise",
                "the youth group, who descend on the dessert table like locusts",
                "the deacon's five-alarm chili, which has humbled mightier men than Goliath",
            ],
            "scripture_note": "Acts 2:46 - Breaking bread together with glad and generous hearts has been a church hallmark from day one!",
            "announcer": "A psalm for the Sunday potluck! O Lord, shield my plate from... what?",
        },
        {
            "id": "bom_daniel_lions",
            "prompt": "When Daniel woke up completely unharmed after a full night's sleep in the lions' den, the only awkward part was ______.",
            "decoy": "the lions wanted a group photo and kept saying 'one more'",
            "bot_answers": [
                "the biggest lion snored like a chariot with a flat wheel",
                "waking up with cat hair all over his only good robe",
                "King Darius shouting 'Daniel, you up?' at six in the morning",
            ],
            "scripture_note": "Daniel 6:22 - 'My God sent his angel and shut the lions' mouths, and they have not harmed me.'",
            "announcer": "Daniel slept great in the lions' den. King Darius did not! What was the awkward part of Daniel's morning?",
        },
        {
            "id": "bom_prodigal_yelp",
            "prompt": "If the Prodigal Son's older brother had a review app, his 1-star review of the welcome-home banquet would say: \"______.\"",
            "decoy": "Fatted calf for the guy who smelled like pigs. Zero goats for the guy who did his chores.",
            "bot_answers": [
                "Music too loud. Ring too shiny. Nobody asked my opinion. One star.",
                "Came home from work to a surprise party for my brother. Nobody told me the dress code.",
                "The food was amazing, which honestly makes me even more upset.",
            ],
            "scripture_note": "Luke 15:31-32 - The Father reminds us to rejoice when the lost are found rather than keeping a ledger of resentment.",
            "announcer": "Uh oh. The older brother is standing outside the party, and he just opened his review app.",
        },
        {
            "id": "bom_sunday_morning",
            "prompt": "The biggest miracle that happens in a Christian household before 8:45 AM on Sunday morning is ______.",
            "decoy": "two matching socks, from the same decade, on the same kid",
            "bot_answers": [
                "everyone in the van, shoes on, and nobody touching anybody",
                "going from 'Where are my keys?' to 'Good morning, brother!' in one parking spot",
                "a toddler staying clean for twelve whole minutes, which scientists can't explain",
            ],
            "scripture_note": "Psalm 122:1 - 'I was glad when they said to me, Let us go to the house of the Lord!' (Eventually!)",
            "announcer": "Every family knows the Sunday morning scramble! What's the biggest miracle in your house before 8:45?",
        },
    ],
    "extra_extra": [
        {
            "id": "ee_jonah_beach",
            "scene": "jonah_beach",
            "subprompt": "SCENE: A giant whale spouts offshore while a kelp-covered prophet washes up on Nineveh Beach holding a soggy scroll as fishermen drop their nets.",
            "prompt": "Write the front-page headline for THE NINEVEH DAILY HERALD!",
            "decoy": "Whale Leaves One-Star Review: 'Passenger Prayed Loudly, Refused To Stay'",
            "bot_answers": [
                "Seaweed Preacher Lands On Beach; Fishermen Ask For His Travel Agent",
                "Prophet Delivered Express, Smells Like Day-Old Tuna",
                "Fishermen Admit: 'Biggest Catch Of Our Lives, And Then He Started Preaching'",
            ],
            "scripture_note": "Jonah 2:10 - The Lord spoke to the fish, and it vomited Jonah out onto dry land - right back on mission.",
            "announcer": "Extra! Extra! A whale just dropped off a prophet on Nineveh Beach! Write the front-page headline for The Nineveh Daily Herald!",
        },
        {
            "id": "ee_ark_hoa",
            "scene": "ark_hoa",
            "subprompt": "SCENE: Noah's 450-foot wooden Ark towers over a dry neighborhood while a confused Homeowners Association inspector measures it with a tiny ruler.",
            "prompt": "Write the breaking headline for THE MESOPOTAMIA GAZETTE!",
            "decoy": "HOA Fines Noah For Boat In Driveway; Noah Says 'Check The Forecast'",
            "bot_answers": [
                "Neighbors Furious As Local Man Parks Floating Zoo On Front Lawn",
                "Inspector Measures Ark With Six-Inch Ruler, Gives Up By Lunch",
                "Weatherman Says Zero Percent Chance Of Rain; Giraffes Board Anyway",
            ],
            "scripture_note": "Hebrews 11:7 - By faith Noah, being warned by God concerning events as yet unseen, in reverent fear constructed an ark.",
            "announcer": "Extra! Extra! Noah's neighbors called the HOA on his boat! Write the breaking headline!",
        },
        {
            "id": "ee_goliath_presser",
            "scene": "goliath_presser",
            "subprompt": "SCENE: Nine-foot bronze-clad Goliath flexes at a pre-fight podium while shepherd boy David casually twirls a sling and checks five smooth stones.",
            "prompt": "Write the sports headline for THE VALLEY OF ELAH SPORTS DESK!",
            "decoy": "Giant Says He Is 'Not Worried At All' About Kid With Rocks",
            "bot_answers": [
                "Shepherd Boy Brings Five Stones To The Big Match, Only Needs One",
                "Philistine Fans Demand Refunds After Championship Lasts Four Seconds",
                "Boy Delivers Cheese To His Brothers, Accidentally Wins National Title",
            ],
            "scripture_note": "1 Samuel 17:47 - 'The battle is the Lord's, and He will give you into our hand!'",
            "announcer": "Extra! Extra! Live from the Valley of Elah, it's the giant versus the shepherd boy! Write the sports headline!",
        },
        {
            "id": "ee_jericho_walls",
            "scene": "jericho_walls",
            "subprompt": "SCENE: Seven priests blast ram's-horn trumpets on Lap #7 around Jericho while two city wall engineers stare in horror at a massive crack.",
            "prompt": "Write the breaking headline for THE JERICHO ARCHITECTURAL DIGEST!",
            "decoy": "City Wall Warranty Did Not Cover 'Marching And Trumpets'",
            "bot_answers": [
                "Seven Laps, One Shout, Zero Walls: Jericho Contractor Not Returning Calls",
                "Brass Section Hits High Note; Entire City Now A Convertible",
                "Quiet Neighbors Finally Make Some Noise; Walls Take It Personally",
            ],
            "scripture_note": "Joshua 6:20 - God's ways often look unconventional to the world, yet obedience brings down strongholds.",
            "announcer": "Extra! Extra! It's lap seven around Jericho, and the trumpets are sounding! Write the breaking headline!",
        },
        {
            "id": "ee_zacchaeus_tree",
            "scene": "zacchaeus_tree",
            "subprompt": "SCENE: Chief tax collector Zacchaeus in silk robes perches high on a sycamore branch while Jesus looks up and invites Himself over for lunch.",
            "prompt": "Write the headline for THE JERICHO FINANCIAL TIMES!",
            "decoy": "Tax Collector Spotted In Tree; Neighbors Say He's Finally Up To Something Good",
            "bot_answers": [
                "Local Man Climbs Tree For Better View, Gets Lunch Guest Instead",
                "Tax Boss Pays Everyone Back Four Times Over; Jericho Accountants Faint",
                "Sycamore Tree Named Best Seat In Jericho",
            ],
            "scripture_note": "Luke 19:9-10 - 'Today salvation has come to this house... For the Son of Man came to seek and to save the lost.'",
            "announcer": "Extra! Extra! Look up in that sycamore tree! Write the headline for Zacchaeus's big day!",
        },
        {
            "id": "ee_water_wine",
            "scene": "water_wine",
            "subprompt": "SCENE: The Cana wedding master-of-ceremonies takes one sip from a stone water-jar ladle with wide-eyed astonishment while the servants exchange high-fives.",
            "prompt": "Write the society headline for THE CANA WEDDING WEEKLY!",
            "decoy": "Wedding Planner Baffled: 'Who Saves The Best For Last?'",
            "bot_answers": [
                "Caterers Panic, Then Celebrate: Plain Water Upgraded To Finest In Galilee",
                "Mary Tells Servants 'Do Whatever He Says'; Servants Do; Party Saved",
                "Groom Gets Credit For The Drinks, Has No Idea What Just Happened",
            ],
            "scripture_note": "John 2:11 - This, the first of His signs, Jesus did at Cana in Galilee, and manifested His glory.",
            "announcer": "Extra! Extra! Breaking news from the wedding at Cana! Write the society headline!",
        },
    ],
    "sub_the_scroll": [
        {
            "id": "sts_balaam_donkey",
            "scene": "balaam_donkey",
            "speaker": "Balaam's Donkey",
            "subprompt": "SCENE: Balaam raises his staff in frustration while his donkey turns around, points a hoof toward an unseen angel with a fiery sword, and opens its mouth.",
            "prompt": "What is Balaam's Donkey ACTUALLY saying in that speech bubble?",
            "decoy": "I'd like to file a complaint with management. And by management, I mean the angel.",
            "bot_answers": [
                "Boss, there's an angel with a flaming sword right there, and you're mad at me?",
                "First time talking, long-time carrier. You need your eyes checked.",
                "I eat thistles for a living, and even I can see the angel in the road.",
            ],
            "scripture_note": "Numbers 22:28-31 - Sometimes the humble donkey sees the hand of God clearer than the proud rider!",
            "announcer": "Sub the Scroll! Balaam's donkey just turned around and started talking. What is it saying?",
        },
        {
            "id": "sts_aaron_calf",
            "scene": "aaron_calf",
            "speaker": "Aaron",
            "subprompt": "SCENE: Moses descends Mount Sinai with glowing stone tablets while Aaron stands awkwardly in front of a shiny golden calf trying to explain.",
            "prompt": "Write Aaron's least-convincing excuse in the speech bubble!",
            "decoy": "Funny story. We threw the earrings in the fire, and out popped this calf. I was as shocked as you!",
            "bot_answers": [
                "Before you say anything, the committee voted while I was in the bathroom.",
                "I was trying to make a casserole dish. It got away from me.",
                "Moses! You're back! Great timing! Love the glowing face! Anyway, bye!",
            ],
            "scripture_note": "Exodus 32:24 - Aaron literally claimed 'I threw it into the fire, and out came this calf!'-a timeless picture of shifting blame.",
            "announcer": "Moses is back down the mountain, and there's a golden calf. What's Aaron's excuse?",
        },
        {
            "id": "sts_elijah_ravens",
            "scene": "elijah_ravens",
            "speaker": "The Delivery Raven",
            "subprompt": "SCENE: Elijah sits by the Brook Cherith as two carrier-ravens swoop in carrying fresh bread and meat in a little woven napkin.",
            "prompt": "What is the Lead Raven saying as it drops off Elijah's breakfast?",
            "decoy": "Special delivery! Please don't tell the other birds we're doing this.",
            "bot_answers": [
                "Bread and meat, as ordered. Five stars, please. We worked very hard not to eat it.",
                "Morning menu is bread and meat. Evening menu is meat and bread. No substitutions.",
                "Do you know how hard it is for a raven to carry a warm sandwich and not take one bite?",
            ],
            "scripture_note": "1 Kings 17:6 - The ravens brought Elijah bread and meat in the morning and evening, and he drank from the brook.",
            "announcer": "Heaven's delivery service has landed at the Brook Cherith! What is the lead raven saying?",
        },
        {
            "id": "sts_eutychus_window",
            "scene": "eutychus_window",
            "speaker": "Eutychus (at 11:58 PM)",
            "subprompt": "SCENE: Midnight in an upstairs room full of smoky oil lamps in Troas. Paul is on point #14 of his sermon while Eutychus sits on a third-story windowsill.",
            "prompt": "What is going through Eutychus's head right before he nods off?",
            "decoy": "I'm not asleep. I'm just resting my eyes in the original Greek.",
            "bot_answers": [
                "Third-floor window. Warm lamps. Point fourteen. What could possibly go wrong?",
                "If I close my eyes for just three seconds, nobody will notice.",
                "Note to self: next week, sit on the floor. The low floor. The lowest floor.",
            ],
            "scripture_note": "Acts 20:9 - A young man named Eutychus, sitting at the window, sank into a deep sleep as Paul talked still longer.",
            "announcer": "It's almost midnight in Troas, and Eutychus is on the windowsill. What's he thinking right before he nods off?",
        },
        {
            "id": "sts_peter_waves",
            "scene": "peter_waves",
            "speaker": "Simon Peter",
            "subprompt": "SCENE: Peter takes three bold steps onto the Sea of Galilee during a windstorm while the other eleven disciples peer over the boat rail.",
            "prompt": "What is Peter saying right as a giant wave splashes his knees?",
            "decoy": "Look, guys, no boat! Wait. Why does the water look so downhill?",
            "bot_answers": [
                "Step one, eyes on Jesus. Step two, do not look at the big wave. Step two! Step two!",
                "Andrew, stop drawing this. I can see you drawing this.",
                "Lord, my faith is great! My sandals, however, are soaked.",
            ],
            "scripture_note": "Matthew 14:29-31 - As long as Peter kept his eyes on Jesus he walked on the water; when Jesus caught him, both returned safely to the boat.",
            "announcer": "Peter just stepped out of the boat and onto the waves! What is he saying?",
        },
        {
            "id": "sts_wise_men",
            "scene": "wise_men",
            "speaker": "The Third Magi",
            "subprompt": "SCENE: Three Magi on camels pause under a blazing star outside Bethlehem checking their gift boxes of gold, frankincense, and myrrh.",
            "prompt": "What is the Third Wise Man saying to the other two right before knocking on the door?",
            "decoy": "Wait. You brought gold, he brought perfume, and I brought embalming spice? Who planned this?",
            "bot_answers": [
                "See? Following the giant star was way faster than asking Herod for directions.",
                "Did anybody keep the gift receipt from Persia?",
                "Next time, we coordinate the gift list before riding eight hundred miles on camels.",
            ],
            "scripture_note": "Matthew 2:10-11 - When they saw the star, they rejoiced exceedingly with great joy and offered Him their treasures.",
            "announcer": "The Wise Men have finally reached Bethlehem! What's the third one saying before they knock?",
        },
    ],
    "truth_or_bluff": [
        {
            "id": "tob_eutychus_after",
            "prompt": "In Acts 20, after young Eutychus fell from a third-story window at midnight and Paul revived him, what did Paul immediately do until daybreak? ______",
            "truth": "Went back upstairs, broke bread, and kept preaching until dawn",
            "truth_keywords": ["kept preaching", "preached until dawn", "talked until daybreak", "went back upstairs", "continued preaching"],
            "decoy": "Moved everyone downstairs and banned window seats forever",
            "bot_answers": [
                "Ended the sermon on the spot and sent everyone home to bed",
                "Appointed two deacons to keep the back row awake",
                "Led a very loud hymn so nobody else could doze off",
            ],
            "scripture_note": "Acts 20:10-11 - After embracing Eutychus and assuring everyone his life was in him, Paul went right back upstairs, broke bread, and conversed with them until daybreak!",
            "announcer": "Truth or Bluff! After Eutychus fell from the window and Paul revived him, what did Paul do next? Write a fake that sounds real!",
        },
        {
            "id": "tob_elisha_axe",
            "prompt": "In 2 Kings 6, when a young prophet accidentally dropped a borrowed iron axe head into the Jordan River, how did Elisha get it back? ______",
            "truth": "He threw a cut stick into the water and the iron axe head floated to the surface",
            "truth_keywords": ["floated", "iron float", "threw a stick", "stick in", "axe head floated"],
            "decoy": "He struck the river with his cloak, and the water parted around the axe",
            "bot_answers": [
                "He prayed, and a big fish swam up with the axe head in its mouth",
                "He had the prophets form a human chain and feel along the riverbed",
                "He lowered a basket on a long rope and scooped it out of the mud",
            ],
            "scripture_note": "2 Kings 6:5-7 - Elisha cut off a stick, threw it in where the borrowed axe sank, and made the heavy iron float so the student could reach out and take it!",
            "announcer": "Truth or Bluff! A student prophet dropped a borrowed axe head in the Jordan. How did Elisha get it back?",
        },
        {
            "id": "tob_ehud_detail",
            "prompt": "In Judges 3, what specific physical detail does Scripture highlight about Ehud, the judge who delivered Israel from King Eglon of Moab? ______",
            "truth": "He was left-handed (allowing him to strap his sword to his right thigh undetected)",
            "truth_keywords": ["left-handed", "left handed", "lefthanded"],
            "decoy": "He had a voice so loud it carried across the Jordan",
            "bot_answers": [
                "He was the shortest warrior in the tribe of Benjamin",
                "He had bright red hair that he never once cut",
                "He could whistle so loudly it scattered the camels",
            ],
            "scripture_note": "Judges 3:15-16 - Ehud son of Gera, a Benjamite, was a left-handed man, so palace guards checking his left side missed the blade on his right thigh.",
            "announcer": "Truth or Bluff! What unusual detail does the Book of Judges record about Ehud? Write a fake good enough to fool your family!",
        },
        {
            "id": "tob_dagon_statue",
            "prompt": "When the Philistines placed the captured Ark of the Covenant beside their idol Dagon in 1 Samuel 5, what happened to the statue of Dagon on the second morning? ______",
            "truth": "It fell face down before the Ark with its head and both hands broken off on the threshold",
            "truth_keywords": ["head and both hands", "head and hands", "hands broken", "fell face down"],
            "decoy": "It turned around overnight to face the wall, like it was in time-out",
            "bot_answers": [
                "It crumbled into a pile of dust from top to bottom",
                "Its crown cracked in half and rolled under the Ark",
                "It was found leaning against the door, as if trying to leave",
            ],
            "scripture_note": "1 Samuel 5:3-4 - Dagon fell face down before the Ark of the Lord, and its head and both hands were lying cut off on the threshold - showing the Lord alone is God.",
            "announcer": "Truth or Bluff! The Philistines left their idol Dagon overnight with the Ark of the Covenant. What happened by the second morning?",
        },
        {
            "id": "tob_peter_fish_count",
            "prompt": "In John 21, when the risen Jesus cooked breakfast on the beach for His disciples, exactly how many large fish did Simon Peter haul ashore without the net tearing? ______",
            "truth": "153 large fish",
            "truth_keywords": ["153", "one hundred fifty-three", "one hundred and fifty-three"],
            "decoy": "144 large fish (twelve for each tribe of Israel)",
            "bot_answers": [
                "77 large fish",
                "120 large fish",
                "100 large fish, exactly",
            ],
            "scripture_note": "John 21:11 - John records the exact eyewitness count: 'full of large fish, 153 of them; and although there were so many, the net was not torn.'",
            "announcer": "Truth or Bluff! John chapter 21 gives an exact count of the fish in the net. Can you bluff your family with a better number?",
        },
        {
            "id": "tob_absalom_hair",
            "prompt": "According to 2 Samuel 14:26, why did King David's son Absalom have to cut his hair once at the end of every year? ______",
            "truth": "Because it grew so heavy on him that the cut hair weighed 200 shekels (about 5 pounds)",
            "truth_keywords": ["heavy", "200 shekels", "weighed", "five pounds", "5 pounds"],
            "decoy": "Because royal princes renewed a special vow every Passover",
            "bot_answers": [
                "Because his helmet no longer fit over all those curls",
                "Because it kept getting tangled in his chariot wheels",
                "Because the palace barber only worked once a year",
            ],
            "scripture_note": "2 Samuel 14:26 & 18:9 - Absalom's prized five-pound mane became his downfall when his head caught fast in the branches of a great oak!",
            "announcer": "Truth or Bluff! Why did Absalom cut his hair only once a year? Write a fake that sounds totally real!",
        },
        {
            "id": "tob_paul_trade",
            "prompt": "According to Acts 18:3, what hands-on trade did the Apostle Paul practice alongside Aquila and Priscilla in Corinth to pay his own way? ______",
            "truth": "Tentmaking (working with leather and goat-hair cloth)",
            "truth_keywords": ["tentmaking", "tentmaker", "tent makers", "tents"],
            "decoy": "Making scrolls and mixing ink for scribes",
            "bot_answers": [
                "Fixing sandals for Roman travelers",
                "Repairing chariot wheels by the city gate",
                "Weaving fishing nets for the Corinth harbor",
            ],
            "scripture_note": "Acts 18:2-3 - Paul stayed and worked with Aquila and Priscilla because they were tentmakers by trade, modeling diligent bi-vocational ministry.",
            "announcer": "Truth or Bluff! What day job did Paul share with Aquila and Priscilla in Corinth?",
        },
    ],
    "survey_says": [
        {
            "id": "ss_pastor_conclusion",
            "prompt": "WE ASKED 100 CHURCHGOERS: What is the #1 unspoken thought when the pastor says \"And in conclusion...\" for the third time?",
            "decoy": "Every other church in town is already at the restaurant taking our booth!",
            "bot_answers": [
                "My roast just went from medium to burnt offering!",
                "Third conclusion? That's not a sermon, that's a trilogy!",
                "I can hear my stomach singing the closing hymn.",
            ],
            "scripture_note": "Ecclesiastes 3:1 - For everything there is a season, and a time for every matter under heaven - including lunch!",
            "announcer": "Double-point finale! Survey says: what's the number one thought when the pastor says 'And in conclusion' for the third time?",
        },
        {
            "id": "ss_noah_question",
            "prompt": "WE ASKED 100 BIBLE READERS: If you could ask Noah ONE practical question about life inside the Ark, what would it be?",
            "decoy": "Where did you keep the woodpeckers, and why were they allowed near the boat?",
            "bot_answers": [
                "Whose turn was it to clean the elephant deck on day thirty-nine?",
                "How did you keep the lions from looking at the zebras like a menu?",
                "Was there anywhere on board to hide from the monkeys?",
            ],
            "scripture_note": "Genesis 6:14-22 - Noah did all that the Lord commanded him, down to every cubit and provision.",
            "announcer": "Double points on the line! Survey says: what's the number one practical question you'd ask Noah about life on the Ark?",
        },
        {
            "id": "ss_sunday_school",
            "prompt": "WE ASKED 100 SUNDAY SCHOOL TEACHERS: What is the funniest answer a kid gives when they don't know the answer to a Bible question?",
            "decoy": "Say 'Jesus' with total confidence, even when the question was about a donkey",
            "bot_answers": [
                "Raise their hand, then tell a ten-minute story about their dog",
                "Guess 'Moses' for the whole Old Testament and 'Paul' for everything else",
                "Ask 'Is it snack time?' with the confidence of a prophet",
            ],
            "scripture_note": "Proverbs 22:6 - Train up a child in the way he should go - and enjoy the hilarious detours along the way!",
            "announcer": "Lightning finale for double points! What's the classic Sunday school answer when a kid has no idea?",
        },
        {
            "id": "ss_without_grumbling",
            "prompt": "WE ASKED 100 FAMILIES: Philippians 2:14 says \"Do all things without grumbling.\" What everyday chore puts that verse to the ultimate test?",
            "decoy": "Stepping on a Lego barefoot at ten at night while carrying the laundry",
            "bot_answers": [
                "Unloading the dishwasher right after someone asks 'Is that clean or dirty?'",
                "Untangling Christmas lights that were 'put away neatly' last January",
                "Cleaning the back seat of the van and finding a french fry from last summer",
            ],
            "scripture_note": "Philippians 2:14-15 - Doing even ordinary tasks without grumbling makes us shine as lights in the world!",
            "announcer": "Double-point finale! What everyday family chore tests 'Do all things without grumbling' the hardest?",
        },
    ],
}

# The bigger bank and the extra round types live in party_prompts.py.
try:
    from arcade.party_prompts import EXTRA_PROMPTS, NEW_ROUND_META
except ImportError:  # the arcade image runs with arcade/ on sys.path
    try:
        from party_prompts import EXTRA_PROMPTS, NEW_ROUND_META  # type: ignore[no-redef]
    except ImportError:
        EXTRA_PROMPTS, NEW_ROUND_META = {}, {}
ROUND_META.update(NEW_ROUND_META)

# Race rounds (Quick Draw, Lightning, Pair Up, In Order) have their own engine and content.
try:
    from arcade import party_race
except ImportError:  # the arcade image runs with arcade/ on sys.path
    import party_race  # type: ignore[no-redef]
ROUND_META.update(party_race.RACE_ROUND_META)
for _kind, _items in EXTRA_PROMPTS.items():
    PROMPT_BANK.setdefault(_kind, []).extend(_items)

#: Every match opens easy and ends on the double-points finale; the five
#: rounds between are drawn fresh each match from every round type there is,
#: always including one Truth or Bluff.
MATCH_OPENER, MATCH_FINALE = "blank_o_matic", "survey_says"


def new_match_schedule() -> list[str]:
    middle_pool = [k for k in PROMPT_BANK if k not in (MATCH_FINALE,) and PROMPT_BANK[k] and k in ROUND_META]
    # Variety first: the opener's type isn't drawn again while others are left.
    others = [k for k in middle_pool if k not in ("truth_or_bluff", MATCH_OPENER)] or \
        [k for k in middle_pool if k != "truth_or_bluff"]
    picks = random.sample(others, k=min(4, len(others)))
    if "truth_or_bluff" in middle_pool:
        picks.append("truth_or_bluff")
    while len(picks) < TOTAL_ROUNDS - 2:
        picks.append(random.choice(middle_pool))
    # One race round every match, sometimes two, in place of a writing round:
    # no two evenings play the same.
    races = [k for k in party_race.RACE_KINDS if party_race.POOLS.get(k) and k in ROUND_META]
    if races:
        swappable = [i for i, k in enumerate(picks) if k != "truth_or_bluff"]
        n = min(len(swappable), len(races), 2 if random.random() < 0.4 else 1)
        for i, kind in zip(random.sample(swappable, n), random.sample(races, n), strict=True):
            picks[i] = kind
    random.shuffle(picks)
    return [MATCH_OPENER, *picks[: TOTAL_ROUNDS - 2], MATCH_FINALE]


#: Prompts used recently in any room: a new room should not replay last night's game.
# Long enough to remember every writing prompt and race clue there is, so nothing comes back early.
_recent_prompt_ids: collections.deque[str] = collections.deque(maxlen=2000)


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
    """True when a player's bluff in truth_or_bluff gives away the real answer.

    Whole words only: substring matching refused honest bluffs ("tents" is in
    "contents", "because it" is in most truths).
    """
    if prompt_item.get("kind") != "truth_or_bluff":
        return False
    words = _normalize_compare(text).split()
    if not words:
        return False
    truth_words = _normalize_compare(str(prompt_item.get("truth") or "")).split()
    if truth_words and words == truth_words:
        return True
    # The bluff is the truth with a word or two changed: most of its words, in a long enough answer.
    if truth_words and len(truth_words) >= 3:
        shared = sum(1 for w in set(truth_words) if w in set(words))
        if shared / len(set(truth_words)) >= 0.8:
            return True
    joined = " " + " ".join(words) + " "
    for kw in prompt_item.get("truth_keywords") or []:
        kw_words = _normalize_compare(str(kw)).split()
        if kw_words and (" " + " ".join(kw_words) + " ") in joined:
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
        _release_screen_async(_rooms.pop(code, None))


def remember_screen(code: str, screen_id: str | None) -> None:
    """The TV a room was cast to (``None`` once it is taken off), so ending the room can free it."""
    screen = _clean_text(screen_id, 160) or None
    with _lock:
        if screen:
            # A TV shows one room: an older room ending must not take the new one off it.
            for other in _rooms.values():
                if other.get("cast_screen") == screen:
                    other["cast_screen"] = None
        room = _rooms.get(_clean_code(code))
        if room is not None:
            room["cast_screen"] = screen


def _release_screen_async(room: dict[str, Any] | None) -> None:
    """Take a finished room off its TV, without holding the lock for the network call."""
    screen = (room or {}).get("cast_screen")
    if not screen:
        return
    code = room["code"]  # type: ignore[index]
    threading.Thread(target=lambda: show_stage_on_screen(screen, "", room_code=code, stop=True),
                     name=f"party-release-{code}", daemon=True).start()


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
    # Never twice in this room's matches if avoidable; among the rest, the one
    # used least recently anywhere (a set test stopped helping once every
    # prompt had been used once).
    last_seen = {pid: i for i, pid in enumerate(_recent_prompt_ids)}
    candidates = [p for p in pool if p["id"] not in used] or pool
    oldest = min(last_seen.get(p["id"], -1) for p in candidates)
    fresh = [p for p in candidates if last_seen.get(p["id"], -1) == oldest]
    chosen = dict(random.choice(fresh))
    chosen["kind"] = kind
    room.setdefault("used_prompt_ids", []).append(chosen["id"])
    _recent_prompt_ids.append(chosen["id"])
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


def _top_up_bots_unlocked(room: dict[str, Any]) -> None:
    """Mid-match, bring the table back to three contestants after someone leaves.

    A scribe who sits down mid-round joins in: during writing it gets an
    answer, during voting it casts a vote. It is never removed mid-round.
    """
    before = {p["id"] for p in room["players"].values() if p.get("is_bot")}
    humans = [p for p in room["players"].values() if not p.get("is_bot") and not p.get("spectator")]
    if len(humans) + len(before) >= 3:
        return
    _ensure_bots_unlocked(room)
    added = [p for p in room["players"].values() if p.get("is_bot") and p["id"] not in before]
    prompt = room.get("current_prompt") or {}
    now = time.time()
    for bot in added:
        if room.get("phase") == "writing":
            taken = {s.get("text") for s in room["submissions"].values()}
            pool = [a for a in prompt.get("bot_answers") or [] if a not in taken] or [f"{bot['name']}'s wise quip"]
            room["submissions"][bot["id"]] = {"player_id": bot["id"], "text": random.choice(pool), "is_decoy": False,
                                             "submitted_at": now + random.uniform(4.0, 20.0)}
        elif room.get("phase") == "voting":
            valid = [o for o in room.get("options") or [] if o.get("kind") != "decoy"] or room.get("options") or []
            if valid:
                room["votes"][bot["id"]] = random.choice(valid)["option_id"]
                room.setdefault("bot_vote_at", {})[bot["id"]] = now + random.uniform(3.0, 12.0)


def _active_contestants_unlocked(room: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in room["players"].values() if not p.get("spectator")]


def _start_round_unlocked(room: dict[str, Any], round_num: int, custom_prompt: dict[str, Any] | None = None) -> None:
    now = time.time()
    if round_num == 1 or not room.get("schedule"):
        room["schedule"] = new_match_schedule()
    kind = round_kind(room, round_num)
    meta = ROUND_META[kind]
    if party_race.is_race(kind):
        _start_race_round_unlocked(room, round_num, kind, meta, now)
        return
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
            # Like a person: done at some point while everyone writes, not the instant it starts.
            "submitted_at": now + PROMPT_READ_SECONDS + random.uniform(6.0, WRITE_SECONDS * 0.8),
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
        "bot_answers": list(prompt_item.get("bot_answers") or []),  # the house's spare answers; never serialized
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


RACE_INTROS = {
    "quick_draw": ("Quick Draw! Fingers on buttons. The first right answer scores the most... "
                   "and a wrong tap costs you. Choose your confidence carefully."),
    "lightning": "Lightning Round! True or false, as fast as you can. Thinking is optional. Being right is not.",
    "pair_up": ("Pair Up! Match every pair on your phone. Perfect and fast wins big. "
                "Perfect and slow wins... a warm feeling."),
    "in_order": ("In Order! Put them in the right order on your phone. Perfect and fast wins big. "
                 "I believe in you. Mostly."),
}
RACE_REVEAL_SECONDS = 24.0


def _race_bots(room: dict[str, Any]) -> list[str]:
    return [p["id"] for p in room["players"].values() if p.get("is_bot") and not p.get("spectator")]


def _race_humans(room: dict[str, Any]) -> list[str]:
    return [p["id"] for p in _active_contestants_unlocked(room) if not p.get("is_bot")]


def _start_race_round_unlocked(room: dict[str, Any], round_num: int, kind: str, meta: dict[str, Any], now: float) -> None:
    rng = random.Random(secrets.randbits(64))
    last_seen = {pid: i for i, pid in enumerate(_recent_prompt_ids)}
    used = set(room.get("used_prompt_ids") or [])
    raws = party_race.choose(kind, last_seen, used, rng)
    for raw in raws:
        room.setdefault("used_prompt_ids", []).append(raw["id"])
        _recent_prompt_ids.append(raw["id"])
    race = party_race.new_race(kind, raws, rng)
    party_race.start_item(race, now, _stage_delay(room), _race_bots(room), rng)
    for p in room["players"].values():
        p["round_delta"] = 0
    room["race"] = race
    room["round"] = round_num
    room["total_rounds"] = TOTAL_ROUNDS
    room["phase"] = "racing"
    room["phase_started_at"] = now
    room["phase_deadline"] = race["deadline"]
    room["current_prompt"] = {
        "id": f"race:{kind}:{round_num}",  # not a clue id: those name the answer
        "kind": kind,
        "title": meta["title"],
        "badge": meta["badge"],
        "icon": meta["icon"],
        "multiplier": meta["multiplier"],
        "instructions": meta["instructions"],
        "prompt": meta["instructions"],
        "subprompt": "",
        "scene": "",
        "speaker": "",
        "decoy": "",
        "bot_answers": [],
        "truth": "",
        "truth_keywords": [],
        "scripture_note": "",
        "announcer": RACE_INTROS.get(kind, meta["title"]),
    }
    room["submissions"] = {}
    room["options"] = []
    room["votes"] = {}
    room["reveal"] = None
    room["updated_at"] = now


def _tick_race_unlocked(room: dict[str, Any], now: float) -> None:
    race = room.get("race") or {}
    if not race:
        return
    before = (race.get("index"), bool(race.get("flash_until")))
    done = party_race.tick(race, now, _race_humans(room), _stage_delay(room, now), _race_bots(room),
                           random.Random(secrets.randbits(64)))
    if done:
        _score_race_unlocked(room)
        return
    room["phase_deadline"] = race["flash_until"] or race["deadline"]
    if (race.get("index"), bool(race.get("flash_until"))) != before:
        room["updated_at"] = now


def _score_race_unlocked(room: dict[str, Any]) -> None:
    now = time.time()
    race = room["race"]
    race["done"] = True
    prompt = room["current_prompt"]
    players = room["players"]
    deltas, raw_events = party_race.score(race, int(prompt.get("multiplier") or 1))
    for p in players.values():
        p["round_delta"] = 0
    for pid, delta in deltas.items():
        p = players.get(pid)
        if not p:
            continue
        before = int(p.get("score") or 0)
        p["score"] = max(0, before + delta)
        p["round_delta"] = p["score"] - before
    events: list[dict[str, Any]] = []
    firsts: dict[str, int] = {}
    for e in raw_events:
        p = players.get(e["player_id"])
        if not p:
            continue
        name = p["name"]
        if e["type"] == "race_first":
            firsts[name] = firsts.get(name, 0) + 1
            text = f"{name} was fastest on question {e['item'] + 1}! (+{e['delta']:,} pts)"
        elif e["type"] == "race_wrong":
            text = f"{name} buzzed in wrong on question {e['item'] + 1}. ({e['delta']:,} pts)"
        else:
            text = f"{name} got it perfect{' and fastest' if e.get('place') == 0 else ''}! (+{e['delta']:,} pts)"
        events.append({**{k: v for k, v in e.items() if k != "item"}, "player_name": name, "text": text})
    ranked = sorted((p for p in players.values() if not p.get("spectator")), key=lambda p: -int(p.get("round_delta") or 0))
    top = ranked[0] if ranked and int(ranked[0].get("round_delta") or 0) > 0 else None
    if top is None:
        line = "Well... nobody cracked that one. The Scribe is laughing in the back. I'm trying very hard not to join him."
    elif race["kind"] in ("pair_up", "in_order") and any(e["type"] == "race_perfect" for e in raw_events):
        line = f"{top['name']} takes the round! Fast hands, a sharp mind... and frankly, a little showing off."
    else:
        line = (f"{top['name']} wins the race with {int(top['round_delta']):,} points! "
                "Those thumbs deserve their own trophy.")
    room["reveal"] = {
        "events": events,
        "round_winners": [top["name"]] if top else [],
        "scripture_note": "",
        "announcer": line,
        "race": [party_race.public_item(it, reveal=True)
                 | {"right": [players[pid]["name"] for pid in party_race.item_result(race, i)["right"] if pid in players]}
                 for i, it in enumerate(race["items"])],
    }
    room["phase"] = "reveal"
    room["phase_started_at"] = now
    room["phase_deadline"] = now + RACE_REVEAL_SECONDS + _stage_delay(room)
    room["updated_at"] = now
    _prefetch_next_prompt_unlocked(room)


def race_answer(code: str, player_id: str, item: Any, choice: Any) -> dict[str, Any]:
    """A player's tap in a race round. The first answer to each clue stands."""
    code = _clean_code(code)
    with _lock:
        room = _rooms.get(code)
        if room is None:
            raise KeyError(code)
        p = room["players"].get(player_id)
        if p is None:
            raise KeyError(player_id)
        if p.get("is_bot"):
            raise PermissionError("That seat belongs to a scribe.")
        if p.get("spectator"):
            raise ValueError("You're watching this round.")
        if room.get("phase") != "racing" or not room.get("race"):
            raise ValueError("There's no race on right now.")
        now = time.time()
        p["last_seen"] = now
        try:
            index = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError("Which question?") from exc
        party_race.answer(room["race"], player_id, index, choice, now)
        _tick_race_unlocked(room, now)
        room["updated_at"] = now
        return serialize_room(room, player_id)


def _public_race(room: dict[str, Any], viewer_id: str | None, now: float) -> dict[str, Any] | None:
    race = room.get("race")
    if not race or room.get("phase") != "racing":
        return None
    idx = race["index"]
    flash = bool(race.get("flash_until"))
    item = race["items"][idx]
    answers = race["answers"][idx]
    mine = answers.get(viewer_id) if viewer_id else None
    out: dict[str, Any] = {
        "kind": race["kind"],
        "index": idx,
        "count": len(race["items"]),
        "flash": flash,
        "item": party_race.public_item(item, reveal=flash),
        "answered": [public_id(room, pid) for pid in answers],
        "mine": {"choice": mine["choice"], "correct": mine["correct"] if flash else None} if mine else None,
    }
    if flash:
        res = party_race.item_result(race, idx)
        players = room["players"]
        out["right"] = [players[pid]["name"] for pid in res["right"] if pid in players]
        out["wrong"] = [players[pid]["name"] for pid in res["wrong"] if pid in players]
        if mine:
            place = res["right"].index(viewer_id) if viewer_id in res["right"] else -1
            out["mine"]["place"] = place
    return out


def _house_extra_answer(prompt: dict[str, Any], submissions: dict[str, dict[str, Any]]) -> str:
    """A second house answer: a pre-written line nobody at the table wrote, never the truth."""
    taken = {_normalize_compare(s.get("text") or "") for s in submissions.values()}
    taken.add(_normalize_compare(prompt.get("decoy") or ""))
    pool = [a for a in prompt.get("bot_answers") or []
            if a and _normalize_compare(a) not in taken and not _matches_truth(a, prompt)]
    return random.choice(pool) if pool else ""


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
            # Only someone who chose the decoy is its author. A seat that never
            # answered is not: its phone would call the trap "Yours" and it
            # could never fall for it.
            if not sub.get("auto_filled"):
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

    # With no AI players at the table, the house plays one more answer of its own
    # (one of the lines a scribe would have written), so there is always more
    # than one way to lose points. It scores exactly like the trap.
    if not any(p.get("is_bot") for p in room["players"].values()):
        extra = _house_extra_answer(prompt, submissions)
        if extra:
            options.append({"option_id": "opt_house_extra", "text": extra, "kind": "decoy",
                            "authors": [], "voters": []})

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

    # A fresh order every round: a seeded order put the trap and the truth in the
    # same slot every time a room code came round again. The order is stored on
    # the room, so it stays stable for everyone during the round.
    rng = random.Random(secrets.randbits(64))
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
            # Cast now (so the round never waits on a bot) but shown when a person might vote.
            room.setdefault("bot_vote_at", {})[bot["id"]] = now + random.uniform(4.0, VOTE_SECONDS * 0.7)

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
            before = int(voter["score"])
            voter["score"] = max(0, before - 250 * mult)
            penalty = voter["score"] - before  # never more than the player had
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
            # (One who never answered was only seated on the decoy, and earns nothing.)
            for author_id in opt.get("authors") or []:
                if author_id == voter_id or author_id not in players:
                    continue
                if (room["submissions"].get(author_id) or {}).get("auto_filled"):
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
                        o["points_awarded"] = int(o.get("points_awarded") or 0) + bonus
                        round_winner_names.append(players[author_id]["name"])
                        events.append({"type": "best_word", "player_id": author_id,
                                       "player_name": players[author_id]["name"], "delta": bonus,
                                       "text": f"{players[author_id]['name']} wrote the Best Word! (+{bonus:,} pts)"})

    # Build announcer summary for the reveal screen
    if round_winner_names:
        winner_str = " and ".join(round_winner_names)
        verb = "take" if len(round_winner_names) > 1 else "takes"
        announcer_reveal = f"The votes are in! {winner_str} {verb} the round bonus! A triumph of taste. Or at least, of votes."
    elif any(e["type"] == "truth_found" for e in events):
        announcer_reveal = "The truth is revealed. Scripture wins, as it tends to do. Now, let's see who spotted it."
    else:
        announcer_reveal = "The Scribe's Trap caught the room. Look at those scores. Actually... maybe don't."

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
    "Good evening, and welcome back. I'm told this is the fun part. Let's find out.",
    "Ah. Another round. My favorite kind of round... the next one.",
    "Thumbs ready, everyone. Try to impress me. I'm very easy to impress. I'm not.",
    "Settle in, friends. History is about to be made. Small history. Living room history.",
    "Welcome back, brilliant minds. And also the rest of you. Here we go.",
    "The stakes have never been higher. That isn't true. But doesn't it sound wonderful?",
    "Phones up. Snacks down. Well... snacks within reach. Here comes the next one.",
    "I have a very good feeling about this round. I said that last round, too. I was wrong.",
    "Ladies and gentlemen, and grown-ups who said they would only play one round... welcome back.",
    "Let's raise the bar. Gently. We don't want anybody tripping over it.",
)
HOST_VOTING = (
    "The answers are in. Some are brilliant. Some are... also answers. Vote for your favorite.",
    "Time to vote. Not for your own. I can tell. I can always tell.",
    "Read every one carefully. The Scribe hid a trap in there, and he is unbearably pleased with himself.",
    "Voting is open! Choose wisely. Or choose quickly. Ideally, both.",
    "Somewhere in that list is the Scribe's Trap. He worked on it all afternoon. Please don't reward him.",
    "Behold... the answers. I've read them. I need a moment. You need to vote.",
    "Pick the one that made you laugh. If none did, pick the one that tried the hardest.",
)
HOST_DRUMROLL = (
    "And now... the results. Try to act surprised.",
    "The votes have been counted. Twice. The second time was out of disbelief.",
    "Drumroll, please! A table works. A knee works. Somebody find a knee.",
    "Let's see who fooled whom... and who fooled themselves.",
    "I've seen the results. I won't say a word. My face, however, is saying several.",
    "Brace yourselves. Some of you are about to be very proud. Some of you are about to be very quiet.",
)
HOST_TRAPPED = (
    "Oh dear. The Scribe's Trap got you. And now he's taking a bow. Nobody asked him to.",
    "And there it is. The Scribe's Trap, working exactly as designed. He would like that noted.",
    "The Scribe's Trap strikes again! Somewhere, a very smug Scribe is writing this down.",
    "That one was the trap all along. Shake it off... and maybe read the next one twice.",
    "Well. That was the Scribe's Trap. He has asked me to thank his fans. I will not.",
    "Caught by the Scribe's Trap. It happens to the best of us. Tonight, it happened to you.",
    "The Scribe's Trap catches another one! He's already practicing his victory speech. It's very long.",
)
HOST_FINALE = (
    "Ladies and gentlemen... your champion! {name}, with {score} points! Hold your applause. No, go ahead.",
    "And the Golden Scroll goes to... {name}! {score} points! Everyone else, a truly respectable effort. Mostly.",
    "{name} takes the Golden Scroll with {score} points! Bragging rights are valid until the next family dinner.",
    "It's official. {name} wins with {score} points! And the sneaky Scribe goes home with nothing. As usual.",
    "With {score} points, {name} is tonight's champion! I'd say I saw it coming. I did not see it coming.",
    "The match is over. The scroll is golden. And {name} has {score} points to prove it. What a night!",
)


#: Clean dad jokes for all ages, Bible-flavoured: (setup, punchline).
DAD_JOKES: tuple[tuple[str, str], ...] = (
    ("What kind of man was Boaz before he married Ruth?", "Ruthless!"),
    ("When is baseball first mentioned in the Bible?", "In the big inning!"),
    ("What did Jonah's family say when he finally got home?", "Long time no sea!"),
    ("How do we know Moses wore a wig?", "Sometimes he was seen with Aaron... and sometimes without!"),
    ("Where was Solomon's temple?", "On the side of his head!"),
    ("Why did the Israelites wander the desert for forty years?", "Even back then, nobody would stop and ask for directions!"),
    ("Who broke all ten commandments at once?", "Moses. He dropped the tablets!"),
    ("What did the Red Sea say when Moses showed up?", "Nothing. It just waved!"),
    ("What did Noah say while he was loading the Ark?", "Now I've herd everything!"),
    ("What kind of car did the disciples drive?", "A Honda. They were all in one Accord!"),
    ("Why didn't anyone play cards on the Ark?", "Because Noah was standing on the deck!"),
    ("What did Adam say the day before Christmas?", "It's Christmas, Eve!"),
    ("How does Moses make his tea?", "Hebrews it!"),
    ("Who was the fastest runner in history?", "Adam. He was first in the human race!"),
    ("Who was the shortest man in the Bible?", "Nehemiah. Knee-high-miah!"),
    ("How do we know Peter was a rich fisherman?", "By his net income!"),
    ("Which animal could Noah never trust?", "The cheetah!"),
    ("Why didn't Noah go fishing much?", "He only had two worms!"),
    ("Which Bible character had no parents?", "Joshua, son of Nun!"),
    ("Who was the first tennis player in the Bible?", "Joseph. He served in Pharaoh's court!"),
    ("Why are church bells so well behaved?", "They only ring when they're tolled!"),
    ("Why did the church potluck hire Samson?", "Somebody had to open the pickle jar!"),
    ("What do you call a lamb covered in chocolate?", "A candy baa!"),
    ("What's the best way to study the Bible?", "Just Luke into it!"),
    ("Why did the man get so excited reading his Bible?", "He thought he'd found a Job!"),
    ("What's a sheep's favourite part of the Sermon on the Mount?", "The Baa-atitudes!"),
    ("Why did Noah scold the chickens on the Ark?", "They were using fowl language!"),
    ("Why did the deacon bring a ladder to church?", "He heard the service was going to be uplifting!"),
    ("What kind of lights did Noah put on the Ark?", "Flood lights!"),
    ("Why was Pharaoh the worst babysitter in Egypt?", "He would never let anybody go!"),
    ("Where do sheep get their hair cut?", "At the baa-baa shop!"),
    ("Why won't the youth group let Elijah drive the church van?", "He takes every corner like a chariot of fire!"),
    ("Why was the shepherd's phone buzzing all night?", "Ninety-nine new messages, and he was still looking for the one!"),
    ("Why was the camel doing stretches before church?", "He heard a sermon about the eye of a needle and wanted to be ready!"),
    ("Which bug has the best work ethic in the Bible?", "The ant. Proverbs sends lazy people to it for job training!"),
    ("Why did the church organist get locked out?", "He couldn't find the right key!"),
    ("What did Moses do with his very first tablet?", "He downloaded the Ten Commandments!"),
    ("What did Daniel say to the lions in the morning?", "Thanks for having me! And thanks for not having me!"),
    ("How do you know the church potluck is going really well?", "The Jell-O starts doing the wave!"),
)


#: Joke slots: each round tells one while players write and one after the result.
_JOKE_SLOTS = {"writing": 0, "between": 1}


def joke_for(room: dict[str, Any], salt: str) -> tuple[str, str]:
    """The next joke from a deck shuffled once per match, so none is told twice in a match."""
    if salt in _JOKE_SLOTS:
        deck_seed, idx = room.get("match_id", "") or room.get("code", ""), int(room.get("round") or 0) * 2 + _JOKE_SLOTS[salt]
    else:  # lobby rotation: its own deck, stepped by time bucket
        deck_seed = f"{room.get('code', '')}:lobby"
        idx = int(re.sub(r"\D", "", salt) or 0)
    deck = list(DAD_JOKES)
    random.Random(deck_seed).shuffle(deck)
    return deck[idx % len(deck)]


#: While players write and vote, the host keeps talking: a remark about the
#: round, a joke or a nudge, the way a game-show host fills dead air.
HOST_CHATTER: dict[str, tuple[str, ...]] = {
    "blank_o_matic": (
        "Fill in that blank. Funny beats correct. Funny and correct? Now you're just showing off.",
        "Somebody once answered one of these with the word potato. It won. Please aim higher.",
        "Remember, somebody in this room has to vote for it. Write accordingly.",
        "A little host advice. The oddly specific answer gets the laugh. The vague one gets a polite nod.",
        "The Scribe has already set his trap. He's been humming about it all day. It's unsettling.",
        "Take your time. Not too much time. I'm a patient host, but the clock is not.",
    ),
    "extra_extra": (
        "Every great headline needs a pun. It's not a rule. Fine... it's a rule.",
        "Short. Punchy. Front page. If it needs a second paragraph, it's not a headline. It's a book.",
        "Read all about it... just as soon as somebody writes it.",
        "The paper boy is waiting, and he has had far too much lemonade.",
        "Big letters, small words. That's journalism. I think. I've never actually checked.",
    ),
    "truth_or_bluff": (
        "Here's the trick. Make your fake sound truer than the truth. Easy. No, it isn't.",
        "One of these answers is the real one from Scripture. Your job is to write something sneakier.",
        "Bluffers, find your best poker face. Or your best... I definitely read all of Leviticus, face.",
        "The real answer is usually stranger than anything you'd guess. That's what makes this fun.",
        "The Scribe is bluffing too. He thinks he's very good at it. He's about average.",
    ),
    "sub_the_scroll": (
        "What are they saying? Only you know. Well... now you do.",
        "Put words in their mouths. Kind words, please. Funny, kind words.",
        "I would love to hear this conversation. Go on. Write it down for me.",
        "Look at that face on the screen. That is the face of someone with something to say.",
        "Keep it short. Speech bubbles are small, and so is my patience.",
    ),
    "survey_says": (
        "Double points, everyone. This is where champions are made... and snacks are earned.",
        "Think like the crowd. What would most folks say in the church parking lot?",
        "Final round. Everything is on the line. I may need to sit down for this.",
        "Behind on points? Comebacks are very biblical. Just ask Jonah. He came all the way back.",
        "Don't be clever this time. Be common. I know. It goes against everything you've done tonight.",
    ),
    "parable_remix": (
        "Same story, brand new setting. Make it make sense... or at least make us laugh.",
        "Jesus told the greatest stories ever told. So... no pressure.",
        "Modern day, timeless truth. Let's hear it.",
        "Imaginary bonus points if a minivan is involved. Double imaginary points for a drive-through.",
        "Keep the heart of the story. Change everything else. That's the whole trick.",
    ),
    "bible_bios": (
        "Keep it short and snappy. Nobody reads a long bio. Not even me, and I read everything.",
        "Picture the profile picture. Now write the bio to go with it.",
        "Hobbies, skills, and one strange fact. That's a winning bio.",
        "Methuselah's bio would just say... still here. Try to beat that.",
        "These are real people from Scripture. Funny is welcome. Kind is required.",
    ),
    "hymn_remix": (
        "Feel free to hum while you write. I won't tell anyone.",
        "The best hymn titles practically sing themselves. The rest need a little help.",
        "Somewhere, a church organist is already warming up. No pressure.",
        "Extra credit if it rhymes. There is no extra credit. Rhyme anyway.",
        "Make it something a choir could sing with a straight face. Or almost a straight face.",
    ),
    "church_bulletin": (
        "Every church bulletin has one announcement nobody expected. Write that one.",
        "Potluck sign-ups? Choir tryouts? A lost and found full of casserole dishes? Go.",
        "Keep it bulletin-sized. The deacons are watching. Lovingly.",
        "Somewhere, a church secretary is proofreading this with a red pen. Make her smile.",
        "It has to fit between the bake sale and the parking reminder. Choose your words wisely.",
    ),
    "voting": (
        "Read them carefully. The Scribe's Trap is hiding in there, and the Scribe is feeling very confident tonight.",
        "No voting for yourself. I'm watching. Warmly. But watching.",
        "Go with your gut. Or your funny bone. Whichever one is louder.",
        "If one made you laugh out loud, that's your vote. Don't overthink it. That's my job.",
        "A tip. The Scribe's Trap usually sounds a little too sure of itself. Much like the Scribe.",
        "Take your time. Some of these deserve a second look. Some deserve a second chance.",
    ),
}
HOST_HURRY = (
    "Ten seconds. No pressure. Some pressure.",
    "Ten seconds! Finish that thought. Any thought.",
    "Ten seconds left! Type like the potluck line is closing!",
    "Ten seconds, thumbs. This is your moment.",
    "Ten seconds. A brilliant answer would be lovely. Any answer would be fine.",
    "Hurry, hurry! Ten seconds on the clock!",
)


def reveal_result_at(room: dict[str, Any]) -> float:
    """Seconds into the reveal when the TV has unveiled its last answer (its revealStep pacing)."""
    n = len(room.get("options") or [])
    return round(1.0 + max(0, n - 1) * REVEAL_PER_ANSWER_SECONDS + 1.5 + _stage_delay(room), 1)


#: The scoreboard takes over the screen this long after the result is called.
SCOREBOARD_AFTER_RESULT_SECONDS = 5.0
RACE_SCOREBOARD_AT_SECONDS = 8.0


def scores_at(room: dict[str, Any]) -> float:
    """Seconds into the reveal when screens switch from the answers to the scoreboard."""
    if (room.get("reveal") or {}).get("race") is not None:
        return round(RACE_SCOREBOARD_AT_SECONDS + _stage_delay(room), 1)
    return round(reveal_result_at(room) + SCOREBOARD_AFTER_RESULT_SECONDS, 1)


def up_next(room: dict[str, Any]) -> dict[str, Any] | None:
    """The round after this one, for the scoreboard's "Up next"."""
    nxt = _next_kind(room)
    if nxt is None:
        return None
    meta = ROUND_META.get(nxt[1]) or {}
    return {"round": nxt[0], "title": meta.get("title", ""), "icon": meta.get("icon", ""),
            "badge": meta.get("badge", ""), "multiplier": int(meta.get("multiplier") or 1)}


def reveal_result_line(room: dict[str, Any]) -> str:
    reveal = room.get("reveal") or {}
    if reveal.get("race") is not None:
        result = str(reveal.get("announcer") or "")  # a race has no trap to blame
    elif reveal.get("round_winners") or any(e.get("type") == "truth_found" for e in reveal.get("events") or []):
        result = str(reveal.get("announcer") or "")
    else:
        result = _pick(room, HOST_TRAPPED, "trap")
    if int(room.get("round") or 0) < int(room.get("total_rounds") or TOTAL_ROUNDS):
        setup, punch = joke_for(room, "between")
        result += f" And while we catch our breath... {setup} ... {punch}"
    return result


def chatter_for(room: dict[str, Any]) -> list[dict[str, Any]]:
    """Lines to say during the current phase: ``[{"at": seconds into it, "text"}]``.

    Spaced through the phase and kept clear of its first lines (the prompt is
    still being read) and its last ten seconds (the hurry call).
    """
    phase = room.get("phase")
    if phase == "reveal":
        return [{"at": reveal_result_at(room), "text": reveal_result_line(room)}]
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
        if prompt.get("kind") == "extra_extra" and not line.lower().startswith("extra"):
            line = f"Extra! Extra! {line}"  # every headline round opens with the paper boy
        if line.lower().startswith("extra"):
            return line
        return f"{_pick(room, HOST_OPENERS, 'open')} {line}".strip()
    if phase == "voting":
        return _pick(room, HOST_VOTING, "vote")
    if phase == "racing" and room.get("race"):
        return race_line(room)
    if phase == "reveal":
        # The result waits for the TV to show the last answer (see chatter_for);
        # saying it now gave the winner away before the reveal.
        return _pick(room, HOST_DRUMROLL, "drum")
    if phase == "finished":
        players = [p for p in (room.get("players") or {}).values() if not p.get("spectator")]
        if not players:
            return "What a match!"
        best = max(players, key=lambda p: int(p.get("score") or 0))
        return _pick(room, HOST_FINALE, "finale").format(name=best.get("name", "Our winner"), score=best.get("score", 0))
    return ""


def race_line(room: dict[str, Any]) -> str:
    """The host during a race: the intro with the first clue, each clue, then who got it."""
    race = room["race"]
    idx, item = race["index"], race["items"][race["index"]]
    kind = race["kind"]
    if race.get("flash_until"):
        res = party_race.item_result(race, idx)
        players = room["players"]
        names = [players[pid]["name"] for pid in res["right"] if pid in players]
        answer = party_race.correct_text(item)
        if kind == "lightning":
            answer = "True" if item["answer"] == 0 else "False"
        answer = answer.rstrip(".!? ") + "!"
        if not names:
            return f"It's {answer} Nobody? Really? Nobody."
        return f"It's {answer} {names[0]} got it first. Naturally."
    if item["type"] == "pick":
        clue = f"{item['clue']} True or false?" if kind == "lightning" else item["clue"]
        lead = f"Question {idx + 1}!" if idx else str((room.get("current_prompt") or {}).get("announcer") or "")
        return f"{lead} {clue}".strip()
    return f"{(room.get('current_prompt') or {}).get('announcer') or ''} {item['clue']}".strip()


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
        if all_submitted or (deadline and now >= deadline + SUBMIT_GRACE_SECONDS):
            _build_voting_options_unlocked(room)
    elif phase == "voting":
        humans = [p for p in _active_contestants_unlocked(room) if not p.get("is_bot")]
        all_voted = bool(humans) and all(p["id"] in room["votes"] for p in humans)
        if all_voted or (deadline and now >= deadline + SUBMIT_GRACE_SECONDS):
            _score_round_unlocked(room)
    elif phase == "racing":
        _tick_race_unlocked(room, now)
    elif phase == "reveal" and deadline and now >= deadline:
        # The scores have been up long enough: the show goes on without waiting for the host.
        _advance_unlocked(room)


AI_PROMPT_KINDS = ("blank_o_matic", "survey_says")


def round_kind(room: dict[str, Any], round_num: int) -> str:
    schedule = room.get("schedule") or list(MATCH_ROUND_SCHEDULE)
    return schedule[max(0, min(len(schedule) - 1, round_num - 1))]


def _next_kind(room: dict[str, Any]) -> tuple[int, str] | None:
    nxt = int(room.get("round") or 0) + 1
    if nxt > int(room.get("total_rounds") or TOTAL_ROUNDS):
        return None
    return nxt, round_kind(room, nxt)


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
    elif phase == "racing":
        _score_race_unlocked(room)
    elif phase == "reveal":
        nxt = _next_kind(room)
        if nxt is None:
            room["phase"] = "finished"
            room["updated_at"] = time.time()
        else:
            ready = room.pop("next_prompt", None) or {}
            if custom is None and ready.get("round") == nxt[0]:
                custom = ready.get("prompt")
            _ensure_bots_unlocked(room)  # who is at the table now, between rounds
            _start_round_unlocked(room, nxt[0], custom_prompt=custom)
    elif phase == "finished":
        for p in room["players"].values():
            p["score"] = 0
            p["round_delta"] = 0
        _ensure_bots_unlocked(room)
        room["match_id"] = f"{room['code']}-{int(time.time())}"
        _start_round_unlocked(room, 1, custom_prompt=custom)


def _public_reveal(room: dict[str, Any]) -> dict[str, Any] | None:
    reveal = room.get("reveal")
    if not reveal:
        return None
    return {**reveal, "events": [{**e, "player_id": public_id(room, e.get("player_id"))} for e in reveal.get("events") or []]}


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
            "id": public_id(room, p["id"]),
            "name": p["name"],
            "initials": p["initials"],
            "avatar": p.get("avatar", "⭐"),
            "score": int(p.get("score") or 0),
            "round_delta": int(p.get("round_delta") or 0),
            "is_bot": bool(p.get("is_bot")),
            "spectator": bool(p.get("spectator")),
            "submitted": (p["id"] in room["race"]["answers"][room["race"]["index"]]) if phase == "racing" and room.get("race")
            else p["id"] in submissions
            and (not p.get("is_bot") or now >= float(submissions[p["id"]].get("submitted_at") or 0)),
            "voted": p["id"] in votes
            and (not p.get("is_bot") or now >= float((room.get("bot_vote_at") or {}).get(p["id"]) or 0)),
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
                    "authors": [public_id(room, a) for a in opt.get("authors") or []],
                    "author_names": author_names,
                    "voters": [{**v, "id": public_id(room, v.get("id"))} for v in opt.get("voters") or []],
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
    timed = ("writing", "voting", "reveal", "racing")
    race = room.get("race") if phase == "racing" else None
    # In a race the clock, and the phones' hold for a streamed TV, are per clue.
    started = float(room.get("phase_started_at") or now)
    if race:
        started = (float(race.get("flash_at") or race["flash_until"]) if race.get("flash_until")
                   else float(race.get("started_at") or started))
    remaining = max(0, round(float(room.get("phase_deadline") or now) - now, 1)) if phase in timed else 0.0

    return {
        "code": room["code"],
        "host_id": public_id(room, room.get("host_id")),
        "host_name": room.get("host_name", "Host"),
        "phase": phase,
        "round": int(room.get("round") or 0),
        "total_rounds": int(room.get("total_rounds") or TOTAL_ROUNDS),
        "seconds_remaining": remaining,
        "phase_seconds": round(float(room.get("phase_deadline") or now) - (started if race else float(room.get("phase_started_at") or now)), 1)
        if phase in timed else 0.0,
        "race": _public_race(room, viewer_id, now),
        # Changes with every clue and every answer flash, so screens know to redraw and speak.
        "step": f"{race['index']}{'f' if race.get('flash_until') else ''}" if race else "",
        "reveal_step_seconds": REVEAL_PER_ANSWER_SECONDS,
        "players": public_players,
        "prompt": public_prompt,
        "options": public_options,
        "reveal": _public_reveal(room) if phase in ("reveal", "finished") else None,
        # The reveal ends on a full-screen scoreboard that leads into the next round.
        "scores_at_s": scores_at(room) if phase == "reveal" else 0.0,
        "up_next": up_next(room) if phase == "reveal" else None,
        "my_submission": {"text": my_sub["text"], "is_decoy": bool(my_sub.get("is_decoy"))} if my_sub else None,
        "my_vote": my_vote,
        "you": public_id(room, viewer_id) if viewer_id and viewer_id in room["players"] else None,
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
        "phase_age_s": round(max(0.0, now - (started if race else float(room.get("phase_started_at") or now))), 2)
        if phase in timed else 0.0,
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


MAX_CONTESTANTS = 8


def _seat_free_unlocked(room: dict[str, Any]) -> bool:
    humans = [p for p in room["players"].values() if not p.get("is_bot") and not p.get("spectator")]
    return len(humans) < MAX_CONTESTANTS


def _free_avatar_unlocked(room: dict[str, Any], pid: str) -> str:
    taken = {p.get("avatar") for p in room["players"].values() if p["id"] != pid}
    return next((a for a in PLAYER_AVATARS if a not in taken), PLAYER_AVATARS[0])


def public_id(room: dict[str, Any], pid: str | None) -> str | None:
    """What other phones see instead of a player's id.

    A player's own id is what authorises their actions (submit, vote, leave,
    host controls), so it is never sent to anyone else; screens get this
    per-room stand-in, the same for everyone.
    """
    if not pid:
        return None
    secret = room.setdefault("secret", secrets.token_hex(16))
    return "u" + hashlib.sha256(f"{secret}:{pid}".encode()).hexdigest()[:12]


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
        existing = (_rooms.get(clean_code) or {}).get("players", {}).get(pid) if clean_code else None
        if existing is not None and existing.get("is_bot"):
            raise PermissionError("That seat belongs to an AI scribe.")
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
            if not spectator and players[pid].get("spectator") and _seat_free_unlocked(room):
                players[pid]["spectator"] = False
                if players[pid].get("avatar") == "👀":
                    players[pid]["avatar"] = _free_avatar_unlocked(room, pid)

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
        if not p.get("is_bot") and not p.get("spectator") and now - float(p.get("last_seen") or 0) <= PLAYER_AWAY_SECONDS
    ]
    if here:
        heir = min(here, key=lambda p: float(p.get("joined_at") or 0))
        room["host_id"] = heir["id"]
        room["host_name"] = heir["name"]


def _hand_off_host_unlocked(room: dict[str, Any]) -> None:
    now = time.time()
    # Someone whose phone is still here first, then anyone seated.
    humans = sorted(_humans_unlocked(room), key=lambda p: (now - float(p.get("last_seen") or 0) > PLAYER_AWAY_SECONDS,
                                                           float(p.get("joined_at") or 0)))
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
        if (room["players"].get(pid) or {}).get("is_bot"):
            raise PermissionError("That is an AI scribe's seat.")
        room["players"].pop(pid, None)
        if room.get("phase") == "writing":
            (room.get("submissions") or {}).pop(pid, None)
        if room.get("phase") == "voting":
            (room.get("votes") or {}).pop(pid, None)
        if not _humans_unlocked(room):
            _release_screen_async(_rooms.pop(clean_code, None))
            return {"closed": True, "room": None}
        if room.get("host_id") == pid:
            _hand_off_host_unlocked(room)
        if room.get("phase") in ("writing", "voting", "reveal"):
            _top_up_bots_unlocked(room)  # never leave the last players with only their own answer and the trap
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
        _release_screen_async(_rooms.pop(clean_code, None))


def _require_host_unlocked(room: dict[str, Any], player_id: str | None) -> None:
    """Host-only controls. The timer advances rooms itself; every request must name its host."""
    pid = _clean_text(player_id, 64)
    if not pid or pid != room.get("host_id"):
        raise PermissionError("Only the host can do that.")


def set_room_voice(code: str, voice: str, *, player_id: str | None = None) -> dict[str, Any]:
    clean_code = _clean_code(code)
    clean_voice = _clean_text(voice, 64) or DEFAULT_ANNOUNCER_VOICE
    with _lock:
        room = _rooms.get(clean_code)
        if room is not None:
            _require_host_unlocked(room, player_id)
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
        if room["phase"] not in ("lobby", "finished"):
            return serialize_room(room, viewer_id=_clean_text(player_id, 64) or None)  # a match is running
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


def _take_late_answer_unlocked(room: dict[str, Any], pid: str, text: str) -> bool:
    """Swap a just-too-late answer in for the house answer its writer was given."""
    sub = (room.get("submissions") or {}).get(pid) or {}
    fresh = time.time() - float(room.get("phase_started_at") or 0) <= LATE_ANSWER_SECONDS
    # AI companions vote the moment voting opens; only a human's vote settles the options.
    players = room.get("players") or {}
    human_voted = any(not (players.get(v) or {}).get("is_bot") for v in (room.get("votes") or {}))
    if not (text and sub.get("auto_filled") and fresh and not human_voted):
        return False
    prompt = room["current_prompt"]
    if _matches_truth(text, prompt) or _normalize_compare(text) == _normalize_compare(prompt.get("decoy", "")):
        return False
    # The same words as an answer already on the ballot would be two identical options.
    if any(_normalize_compare(o.get("text") or "") == _normalize_compare(text) for o in room.get("options") or []):
        return False
    room["submissions"][pid] = {"player_id": pid, "text": text, "is_decoy": False, "submitted_at": time.time(),
                                "late": True}
    options = room.get("options") or []
    for o in options:
        if o.get("kind") == "decoy" and pid in (o.get("authors") or []):
            o["authors"] = [a for a in o["authors"] if a != pid]
    opt_hash = hashlib.sha1(f"{pid}:{text}".encode()).hexdigest()[:8]
    options.insert(random.randint(0, len(options)), {"option_id": f"opt_{opt_hash}", "text": text, "kind": "player",
                                                     "authors": [pid], "voters": []})
    room["options"] = options
    room["updated_at"] = time.time()
    return True


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
        if room["phase"] == "voting" and not use_decoy and _take_late_answer_unlocked(room, pid, clean_ans):
            return serialize_room(room, viewer_id=pid)
        if room["phase"] != "writing":
            raise ValueError("This round's writing phase has already closed.")
        if pid not in room["players"]:
            raise ValueError("Join the room before submitting an answer.")
        if room["players"][pid].get("is_bot"):
            raise PermissionError("That is an AI scribe's seat.")
        if room["players"][pid].get("spectator"):
            if not _seat_free_unlocked(room):
                raise ValueError("The room is full. You're watching this match.")
            room["players"][pid]["spectator"] = False
            if room["players"][pid].get("avatar") == "👀":
                room["players"][pid]["avatar"] = _free_avatar_unlocked(room, pid)
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
        if room["players"][pid].get("is_bot"):
            raise PermissionError("That is an AI scribe's seat.")
        if room["players"][pid].get("spectator"):
            raise ValueError("You're watching this match, so you can't vote this round.")
        target = next((o for o in room["options"] if o["option_id"] == oid), None)
        if target is None:
            raise ValueError("Unknown answer option.")
        if pid in (target.get("authors") or []):
            raise ValueError("You cannot vote for your own answer!")
        room["votes"][pid] = oid
        room["updated_at"] = time.time()
        _check_auto_advance_unlocked(room)
        return serialize_room(room, viewer_id=pid)


def advance_phase(code: str, *, use_ai_prompt: bool = False, player_id: str | None = None,
                  expect: dict[str, Any] | None = None) -> dict[str, Any]:
    """The host moves the show on: writing -> voting -> reveal -> next round / finished.

    ``expect`` is the screen the host tapped on (``phase``, ``round``,
    ``match_id``). If the room already moved on (the timer advanced it a
    moment earlier), the tap is about a screen that is gone and does nothing;
    otherwise a late tap would skip the next round's writing, or wipe the
    final scores by starting a new match.
    """
    clean_code = _clean_code(code)
    with _lock:
        room = _rooms.get(clean_code)
        if room is None:
            raise LookupError(f"Room '{clean_code}' not found.")
        _require_host_unlocked(room, player_id)
        _check_auto_advance_unlocked(room)
        if expect and (
            (expect.get("phase") and expect["phase"] != room["phase"])
            or (expect.get("round") is not None and str(expect["round"]) != str(int(room.get("round") or 0)))
            or (expect.get("match_id") and expect["match_id"] != room.get("match_id"))
        ):
            return serialize_room(room, viewer_id=_clean_text(player_id, 64) or None)
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


def _cache_clip(clip_id: str, wav: bytes) -> None:
    with _lock:
        while _tts_clips and (len(_tts_clips) >= 96 or
                              sum(map(len, _tts_clips.values())) + len(wav) > TTS_CACHE_MAX_BYTES):
            _tts_clips.pop(next(iter(_tts_clips)))
        _tts_clips[clip_id] = wav


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
            _cache_clip(clip_id, wav)
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
    parts = announcer.split_show_line((text or "")[:TTS_MAX_CHARS])
    if not parts:
        return None
    pieces: list[tuple[str, str, bytes, float]] = []
    for role, line in parts:
        # The announcer goes sentence by sentence, each at its own pace with a
        # breath after it. The newsboy's call is one breath ("Extra! Extra!"
        # voiced twice from one clip sounded like a stutter).
        for sentence in ([line] if role == "paperboy" else announcer.sentences(line)):
            if role == "paperboy":
                speed, gap = announcer.PAPERBOY_SPEED, 180.0
                clip = synthesize_announcer_clip(sentence, announcer.PAPERBOY_VOICE, speed=speed)
            else:
                speed, gap = announcer.cadence(sentence)
                clip = synthesize_announcer_clip(announcer.pronounce(sentence), voice, clone=clone, speed=speed)
                if clip is not None and not sentence.rstrip().endswith("!"):
                    # The shouts land: everything else sits a little under them.
                    clip = (clip[0] + "q", announcer.quieter(clip[1], announcer.SHOUT_LIFT_DB))
            if clip is None:
                if role == "paperboy":
                    continue  # no newsboy today: the announcer still reads the line
                return None
            pieces.append((role, clip[0], clip[1], gap))
    if not pieces:
        return None
    show_id = hashlib.sha256(("show:v6:" + ":".join(f"{r}={c}" for r, c, _, _ in pieces)).encode()).hexdigest()[:16]
    with _lock:
        cached = _tts_clips.get(show_id)
    if cached:
        return show_id, cached
    try:
        wav = announcer.assemble([(r, w, g) for r, _, w, g in pieces])
    except Exception:
        wav = pieces[-1][2]  # an unreadable clip is still better spoken plainly
    _cache_clip(show_id, wav)
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
