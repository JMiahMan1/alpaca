"""Extra prompt bank for Parable Party (``arcade/party.py``).

``EXTRA_PROMPTS`` adds new items to the five existing round kinds and fills four new
kinds, whose round metadata is ``NEW_ROUND_META`` (same keys as ``party.ROUND_META``).
Every item follows the ``party.PROMPT_BANK`` schema; ``tests/test_party_prompts.py`` checks it.

Content rules: clean, family-friendly (ages 7+), respectful of Scripture and of every
Christian tradition. Each ``scripture_note`` cites a real verse that relates to the prompt, and
every ``truth_or_bluff`` truth is a well-known, unambiguous fact from the cited passage.
"""

from __future__ import annotations

from typing import Any

NEW_ROUND_META: dict[str, dict[str, Any]] = {
    "parable_remix": {
        "title": "Parable Remix",
        "badge": "MODERN RETELLING",
        "icon": "🔄",
        "multiplier": 1,
        "instructions": (
            "Retell or retitle this Bible story as if it happened today, in your funniest everyday setting!"
        ),
    },
    "bible_bios": {
        "title": "Bible Bios",
        "badge": "PROFILE PAGE",
        "icon": "📱",
        "multiplier": 1,
        "instructions": "Write a short bio, headline, or job-listing line for this Bible character!",
    },
    "hymn_remix": {
        "title": "Hymn Remix",
        "badge": "SING IT OUT",
        "icon": "🎵",
        "multiplier": 1,
        "instructions": "Write a brand-new hymn title or one singable line for this everyday family moment!",
    },
    "church_bulletin": {
        "title": "Church Bulletin",
        "badge": "SUNDAY ANNOUNCEMENTS",
        "icon": "📋",
        "multiplier": 1,
        "instructions": "Write the church bulletin announcement nobody expected to read this Sunday!",
    },
}


EXTRA_PROMPTS: dict[str, list[dict[str, Any]]] = {
    "blank_o_matic": [
        {
            "id": "bom_adam_naming",
            "prompt": (
                "When God brought every animal to Adam to be named, the one name Adam secretly regretted was "
                "\"______.\""
            ),
            "decoy": "Sir Hops-a-Lot, for the kangaroo, in front of everybody",
            "bot_answers": [
                "Long Neck Larry, which the giraffe still brings up",
                "Mr. Stripes, before he noticed the zebra AND the tiger",
                "Bob. He named the first forty animals Bob",
            ],
            "scripture_note": (
                "Genesis 2:19-20 - God brought the animals to Adam to see what he would call them, and whatever he "
                "called them, that was their name."
            ),
            "announcer": "Fill in the blank! Adam had to name every animal on Earth. Which name did he regret?",
        },
        {
            "id": "bom_joshua_sun",
            "prompt": "When the sun stood still for Joshua, the people in the next town over all said: \"______!\"",
            "decoy": "This is the longest Tuesday of my entire life",
            "bot_answers": [
                "Great, now my bread is extra overbaked",
                "Somebody please tell the rooster to wait",
                "Finally, enough daylight to finish the laundry",
            ],
            "scripture_note": (
                "Joshua 10:13 - The sun stood still in the midst of heaven and did not hurry to go down for about a "
                "whole day."
            ),
            "announcer": "The sun stood still for Joshua! What were the neighbors saying? Fill in the blank!",
        },
        {
            "id": "bom_manna_menu",
            "prompt": (
                "After forty years of manna for breakfast, lunch, and dinner, the Israelites' new favorite recipe was "
                "______."
            ),
            "decoy": "manna with a side of manna, garnished with a little more manna",
            "bot_answers": [
                "Manna-cotti, which is just manna rolled up and hopeful",
                "manna pancakes with quail syrup",
                "Grandma's famous 'surprise, it's manna' casserole",
            ],
            "scripture_note": (
                "Exodus 16:31,35 - Manna tasted like wafers made with honey, and Israel ate it for forty years."
            ),
            "announcer": "Forty years of the same menu! What was the hottest manna recipe in the wilderness?",
        },
        {
            "id": "bom_jacob_ladder",
            "prompt": (
                "When Jacob woke up from his dream of a stairway to heaven, his very first thought was: \"______.\""
            ),
            "decoy": "Next time I'm packing a real pillow, this rock was NOT memory foam",
            "bot_answers": [
                "I need one of those for the attic",
                "Surely the Lord is in this place, and also my neck hurts",
                "Were the angels taking the stairs or the elevator?",
            ],
            "scripture_note": (
                "Genesis 28:12,16 - Jacob dreamed of a ladder with angels going up and down, and woke saying, 'Surely "
                "the Lord is in this place.'"
            ),
            "announcer": "Jacob just woke up from the most amazing dream ever! What was his first thought?",
        },
        {
            "id": "bom_joseph_coat",
            "prompt": "Joseph's coat of many colors came with a care tag that read: \"Wash separately, ______.\"",
            "decoy": "and do not, under any circumstances, wear it in front of your brothers",
            "bot_answers": [
                "or it turns all your other robes rainbow",
                "tumble dry low, avoid deep pits",
                "hand-stitched with love by Dad, sorry to the other eleven sons",
            ],
            "scripture_note": (
                "Genesis 37:3 - Israel loved Joseph more than all his children and made him a coat of many colors."
            ),
            "announcer": "Check the tag on Joseph's famous coat! Fill in the care instructions!",
        },
        {
            "id": "bom_samson_barber",
            "prompt": "Samson's neighborhood barber shop had a sign on the door that read: \"______.\"",
            "decoy": "Trims available. Strength not included.",
            "bot_answers": [
                "Absolutely NO walk-ins named Delilah",
                "We are not responsible for collapsing pillars",
                "Please stop asking for 'just a little off the top'",
            ],
            "scripture_note": (
                "Judges 16:17 - Samson told Delilah that if his head were shaved, his strength would leave him."
            ),
            "announcer": "Samson walks past a barber shop! What does the sign on the door say?",
        },
        {
            "id": "bom_ruth_gleaning",
            "prompt": "Ruth's number one tip for anyone gleaning in a barley field all day is ______.",
            "decoy": "always pick the field owned by the kind man who drops extra barley on purpose",
            "bot_answers": [
                "wear comfy sandals and bring your mother-in-law snacks",
                "stay humble, stay hydrated, stay near the reapers",
                "never argue with Naomi about where to go next",
            ],
            "scripture_note": (
                "Ruth 2:3 - Ruth went out to glean and happened to come to the part of the field belonging to Boaz."
            ),
            "announcer": "Ruth is the expert gleaner of Bethlehem! What's her number one tip?",
        },
        {
            "id": "bom_elijah_baal",
            "prompt": "On Mount Carmel, Elijah teased the prophets of Baal that maybe their god was ______.",
            "decoy": "stuck in traffic somewhere on the Damascus Road",
            "bot_answers": [
                "on hold with customer service",
                "taking a really long nap after lunch",
                "still looking for the matches",
            ],
            "scripture_note": (
                "1 Kings 18:27 - Elijah mocked them: perhaps their god was busy, on a journey, or asleep and needed "
                "waking!"
            ),
            "announcer": (
                "Showdown on Mount Carmel! Elijah has a few jokes for the prophets of Baal. Fill in the blank!"
            ),
        },
        {
            "id": "bom_abraham_stars",
            "prompt": "When God told Abraham to count the stars, Abraham got as far as ______ before he dozed off.",
            "decoy": "number 7,000 and one very distracting shooting star",
            "bot_answers": [
                "eleventy-twelve, which he's pretty sure is a real number",
                "the Big Dipper, twice",
                "forty-three, then he started counting sheep instead",
            ],
            "scripture_note": (
                "Genesis 15:5 - God took Abraham outside and said, 'Look toward heaven and count the stars... so "
                "shall your offspring be.'"
            ),
            "announcer": "Abraham is outside counting stars! How far did he get? Fill in the blank!",
        },
        {
            "id": "bom_nehemiah_wall",
            "prompt": "Nehemiah's crew rebuilt the whole wall of Jerusalem in just 52 days, mainly because ______.",
            "decoy": "nobody stopped for a coffee break or a committee meeting",
            "bot_answers": [
                "the neighbors kept heckling and that's very motivating",
                "they held a trowel in one hand and a sandwich in the other",
                "Nehemiah promised pizza when they finished",
            ],
            "scripture_note": (
                "Nehemiah 6:15 - The wall was finished in fifty-two days, because the people had a mind to work."
            ),
            "announcer": "Fifty-two days to rebuild a city wall! What was Nehemiah's secret? Fill in the blank!",
        },
        {
            "id": "bom_esther_reminder",
            "prompt": "Queen Esther's phone reminder the night before she visited the king said: \"______.\"",
            "decoy": "Pick the outfit. Practice the bow. For such a time as this!",
            "bot_answers": [
                "Make enough banquet food, Haman's staying for two",
                "Do NOT trip on the throne room steps",
                "Three days of fasting done. Don't eat ALL the banquet snacks",
            ],
            "scripture_note": (
                "Esther 4:14 - 'Who knows whether you have come to the kingdom for such a time as this?'"
            ),
            "announcer": "Queen Esther set a reminder on her phone! What did it say? Fill in the blank!",
        },
        {
            "id": "bom_lost_sheep",
            "prompt": "When the shepherd finally found his one lost sheep, the sheep's excuse was: \"______.\"",
            "decoy": "I wasn't lost, I was taking the scenic route",
            "bot_answers": [
                "The grass really WAS greener over here",
                "I followed a butterfly and then another butterfly",
                "I thought we were playing hide and seek",
            ],
            "scripture_note": (
                "Luke 15:4-6 - The shepherd leaves the ninety-nine, finds the lost sheep, and carries it home on his "
                "shoulders, rejoicing."
            ),
            "announcer": "The shepherd found his lost sheep! What excuse did the sheep give? Fill in the blank!",
        },
        {
            "id": "bom_samaritan_review",
            "prompt": "The Good Samaritan's online review of the roadside inn said: \"______.\"",
            "decoy": "Clean rooms, kind staff, and they'll put the rest on my tab",
            "bot_answers": [
                "Two denarii well spent, would rescue again",
                "Great place to recover from a really bad trip to Jericho",
                "Innkeeper takes excellent care of guests, highly recommend",
            ],
            "scripture_note": (
                "Luke 10:34-35 - The Samaritan brought the hurt man to an inn, paid two denarii, and promised to "
                "cover the rest."
            ),
            "announcer": "The Good Samaritan left a review for the inn! What did it say?",
        },
        {
            "id": "bom_martha_kitchen",
            "prompt": "While Mary sat listening at Jesus' feet, Martha was in the kitchen muttering: \"______.\"",
            "decoy": "Fine, I'll just make thirteen dinners by myself, no big deal",
            "bot_answers": [
                "Somebody is getting the burnt pita tonight",
                "If anyone needs me, I'll be loudly stacking dishes",
                "Lord, do You SEE how many dishes there are?",
            ],
            "scripture_note": (
                "Luke 10:40-42 - Martha was distracted with much serving, but Jesus said Mary had chosen the good "
                "portion."
            ),
            "announcer": "Martha is busy in the kitchen while Mary listens to Jesus! What is she muttering?",
        },
        {
            "id": "bom_sand_house",
            "prompt": (
                "The wise man built his house upon the rock. The foolish man built his house upon the sand because "
                "the sand ______."
            ),
            "decoy": "had a gorgeous ocean view and a beach volleyball net",
            "bot_answers": [
                "was half price and came with a free bucket",
                "was closer to the snack shack",
                "looked very solid at low tide",
            ],
            "scripture_note": (
                "Matthew 7:24-27 - Whoever hears Jesus' words and does them is like a wise man who built his house on "
                "the rock."
            ),
            "announcer": "Everybody knows the song! But why did the foolish man pick the sand? Fill in the blank!",
        },
        {
            "id": "bom_lazarus_normal",
            "prompt": "After Jesus raised Lazarus, the hardest part of getting back to normal life was ______.",
            "decoy": "explaining to his boss why he missed four days of work",
            "bot_answers": [
                "unwrapping all those linen strips before breakfast",
                "everybody wanting to eat dinner with him",
                "his sisters hugging him every five minutes",
            ],
            "scripture_note": (
                "John 11:43-44 - Jesus cried, 'Lazarus, come out!' and the man who had died came out, still wrapped "
                "in linen strips."
            ),
            "announcer": "Lazarus is back! What was the hardest part of getting back to normal? Fill in the blank!",
        },
        {
            "id": "bom_babel_hammer",
            "prompt": (
                "The first sign of trouble at the Tower of Babel was when somebody asked for a hammer and got ______."
            ),
            "decoy": "a sandwich, a llama, and a very confused look",
            "bot_answers": [
                "a pickle and a polite bow",
                "a song in a language nobody had heard",
                "a brick, a nod, and a very long speech",
            ],
            "scripture_note": (
                "Genesis 11:7-9 - The Lord confused their language so they could not understand one another."
            ),
            "announcer": "Construction trouble at the Tower of Babel! What did the worker get instead of a hammer?",
        },
        {
            "id": "bom_gideon_fleece",
            "prompt": (
                "Gideon put out a fleece twice to double-check God's answer. Today he would probably check by ______."
            ),
            "decoy": "asking for a sign and then asking for the sign's receipt",
            "bot_answers": [
                "checking the weather app three times",
                "flipping a coin in two different rooms",
                "texting the angel 'r u sure?'",
            ],
            "scripture_note": (
                "Judges 6:37-40 - Gideon asked for dew on the fleece only, then dew everywhere but the fleece, and "
                "God answered both times."
            ),
            "announcer": "Gideon loves a second opinion! How would he double-check today? Fill in the blank!",
        },
        {
            "id": "bom_disciples_sticker",
            "prompt": "The disciples' fishing boat had a bumper sticker that said: \"______.\"",
            "decoy": "I'd rather be fishing for people",
            "bot_answers": [
                "Honk if you've walked on water (Peter only)",
                "Caution: may stop suddenly for storms",
                "My other boat is full of 153 fish",
            ],
            "scripture_note": "Matthew 4:19 - 'Follow me, and I will make you fishers of men.'",
            "announcer": "Bumper sticker time! What's on the back of the disciples' fishing boat?",
        },
        {
            "id": "bom_furnace_advice",
            "prompt": (
                "After walking out of the fiery furnace without even smelling like smoke, Shadrach, Meshach and "
                "Abednego's advice was ______."
            ),
            "decoy": "always travel with a fourth friend",
            "bot_answers": [
                "stand firm and stay cool",
                "never bow to a giant gold statue, even if the band is great",
                "pack sunscreen, just in case",
            ],
            "scripture_note": (
                "Daniel 3:25,27 - Four men were seen walking in the fire, and not even the smell of smoke was on "
                "them."
            ),
            "announcer": (
                "Three friends walked out of a furnace without a scratch! What's their advice? Fill in the blank!"
            ),
        },
        {
            "id": "bom_vbs_memory",
            "prompt": (
                "The real reason every kid remembers Vacation Bible School for the rest of their life is ______."
            ),
            "decoy": "the snack lady who gave out double cookies on Thursday",
            "bot_answers": [
                "the song with hand motions that never leaves your brain",
                "the youth pastor getting soaked in the water balloon finale",
                "the craft still glued to Grandma's fridge",
            ],
            "scripture_note": "Psalm 78:4 - We will tell the coming generation the glorious deeds of the Lord.",
            "announcer": "Vacation Bible School memories! Why does every kid remember it forever?",
        },
        {
            "id": "bom_choir_member",
            "prompt": "Every church choir has that one wonderful member who ______.",
            "decoy": "sings the loudest during the one song they don't know",
            "bot_answers": [
                "holds the last note three seconds after everyone else",
                "brings cough drops for the whole alto section",
                "starts swaying and accidentally leads the whole row",
            ],
            "scripture_note": "Psalm 98:4 - Make a joyful noise unto the Lord, all the earth!",
            "announcer": "Every choir has one! Fill in the blank about that special choir member!",
        },
        {
            "id": "bom_family_devotions",
            "prompt": "Family devotions at our house always end exactly the same way: ______.",
            "decoy": "with the dog howling all the way through the closing prayer",
            "bot_answers": [
                "everyone arguing over who gets to blow out the candle",
                "the toddler asking if dinosaurs were on the Ark",
                "the youngest praying for every single stuffed animal by name",
            ],
            "scripture_note": (
                "Deuteronomy 6:6-7 - Talk of God's words when you sit in your house, when you walk by the way, and "
                "when you lie down."
            ),
            "announcer": "Family devotion time! How does it always end at your house?",
        },
        {
            "id": "bom_nursery_medal",
            "prompt": "The volunteers in the church nursery deserve a gold medal for ______.",
            "decoy": "finding the matching tiny shoe in under four minutes",
            "bot_answers": [
                "singing 'Jesus Loves Me' forty times in a row",
                "settling a goldfish cracker dispute with total wisdom",
                "returning every baby to the right parent",
            ],
            "scripture_note": "Mark 10:14 - 'Let the little children come to me, and do not hinder them.'",
            "announcer": "Let's hear it for the nursery volunteers! What do they deserve a medal for?",
        },
        {
            "id": "bom_widow_oil",
            "prompt": "As the widow's oil kept pouring into every borrowed jar, her son whispered: \"______.\"",
            "decoy": "Mom, should we borrow the neighbor's bathtub too?",
            "bot_answers": [
                "We are going to need a bigger kitchen",
                "Can it pour chocolate milk next?",
                "I'll go knock on every door on the street!",
            ],
            "scripture_note": (
                "2 Kings 4:3-6 - Elisha told the widow to borrow many empty jars, and the oil kept flowing until "
                "every jar was full."
            ),
            "announcer": "The oil just won't stop pouring! What did the widow's son whisper?",
        },
        {
            "id": "bom_malta_snake",
            "prompt": "When Paul was shipwrecked on Malta and a snake bit his hand, he simply ______.",
            "decoy": "shook it off and asked if anyone else wanted s'mores",
            "bot_answers": [
                "added 'snakebite' to his very long list of travel stories",
                "wrote it into his next letter as a P.S.",
                "asked the islanders to please stop staring",
            ],
            "scripture_note": "Acts 28:5 - Paul shook the creature off into the fire and suffered no harm.",
            "announcer": "Shipwrecked on Malta and now a snake! What did Paul do? Fill in the blank!",
        },
        {
            "id": "bom_nine_lepers",
            "prompt": (
                "Only one of the ten healed lepers came back to say thank you. The other nine were probably ______."
            ),
            "decoy": "writing thank-you notes that they never actually mailed",
            "bot_answers": [
                "showing off their new skin to their moms",
                "planning a big party and forgot to invite the guest of honor",
                "stuck in a really long line at the temple",
            ],
            "scripture_note": (
                "Luke 17:15-17 - One returned, praising God with a loud voice, and Jesus asked, 'Where are the nine?'"
            ),
            "announcer": "Ten were healed but only one came back! Where were the other nine? Fill in the blank!",
        },
        {
            "id": "bom_bethlehem_inn",
            "prompt": "The Bethlehem innkeeper's sign on that famous night read: \"No vacancy, but ______.\"",
            "decoy": "the stable has excellent reviews from the cows",
            "bot_answers": [
                "free hay with every stay",
                "the shepherds say the view of the angels is amazing",
                "donkey parking is free",
            ],
            "scripture_note": (
                "Luke 2:7 - Mary laid her baby in a manger, because there was no room for them in the inn."
            ),
            "announcer": "The inn in Bethlehem is completely full! What did the sign say? Fill in the blank!",
        },
        {
            "id": "bom_church_parking",
            "prompt": "The holiest, most sought-after spot in the whole church parking lot is ______.",
            "decoy": "the space closest to the door that nobody ever leaves",
            "bot_answers": [
                "the shady spot under the big oak tree",
                "the 'Visitor' space the greeters guard with love",
                "the spot right next to the donut table door",
            ],
            "scripture_note": "Hebrews 10:25 - Do not neglect meeting together, but encourage one another.",
            "announcer": "Church parking lot showdown! What's the most sought-after spot? Fill in the blank!",
        },
        {
            "id": "bom_thomas_proof",
            "prompt": (
                "Thomas needed to see it to believe it. Today, Thomas would also need ______ before believing "
                "anything."
            ),
            "decoy": "three reviews, a receipt, and a slow-motion replay",
            "bot_answers": [
                "a fact-checker and a second opinion from his twin",
                "a selfie with a timestamp",
                "his mom to call and confirm it",
            ],
            "scripture_note": (
                "John 20:27-29 - Jesus said to Thomas, 'Blessed are those who have not seen and yet have believed.'"
            ),
            "announcer": "Thomas wants proof! What would he need before believing anything today?",
        },
    ],
    "extra_extra": [
        {
            "id": "ee_red_sea_crossing",
            "scene": "red_sea_crossing",
            "subprompt": (
                "SCENE: Two towering walls of water with curious fish peering out as a huge crowd of families strolls "
                "across a dry seabed with goats and luggage."
            ),
            "prompt": "Write the front-page headline for THE GOSHEN GAZETTE!",
            "decoy": "COMMUTE TIMES IMPROVE AS NEW SEABED EXPRESSWAY OPENS WITH ZERO TOLL BOOTHS",
            "bot_answers": [
                "FISH REPORT UNUSUAL FOOT TRAFFIC IN THEIR LIVING ROOM",
                "MOSES RAISES STAFF; SEA TAKES THE DAY OFF",
                "MILLIONS CROSS SEA, NOT ONE WET SANDAL REPORTED",
            ],
            "scripture_note": (
                "Exodus 14:29 - The people of Israel walked on dry ground through the sea, the waters a wall on their "
                "right and left."
            ),
            "announcer": (
                "Extra! Extra! The sea has split wide open! Write the front-page headline for The Goshen Gazette!"
            ),
        },
        {
            "id": "ee_manna_morning",
            "scene": "manna_morning",
            "subprompt": (
                "SCENE: Desert families step out of their tents at sunrise to find the ground covered in white flaky "
                "wafers like frost, as kids run out with baskets."
            ),
            "prompt": "Write the breakfast-time headline for THE WILDERNESS MORNING POST!",
            "decoy": "BREAKFAST FALLS FROM SKY; LOCAL MOMS DECLARE 'NO COOKING TODAY'",
            "bot_answers": [
                "MYSTERY FLAKES TASTE LIKE HONEY WAFERS; CROWD NAMES THEM 'WHAT IS IT?'",
                "DESERT GROCERY STORES REPORT ZERO SALES FOR FORTIETH STRAIGHT YEAR",
                "FORECAST: 100% CHANCE OF BREAD SHOWERS",
            ],
            "scripture_note": (
                "Exodus 16:14-15 - Fine flakes like frost covered the ground, and the people asked, 'What is it?' It "
                "was bread from the Lord."
            ),
            "announcer": (
                "Extra! Extra! Breakfast has fallen from the sky! Write the headline for The Wilderness Morning Post!"
            ),
        },
        {
            "id": "ee_babel_tower",
            "scene": "babel_tower",
            "subprompt": (
                "SCENE: A half-built brick tower rises into the clouds while workers on wobbly scaffolding wave their "
                "arms at each other, clearly speaking different languages."
            ),
            "prompt": "Write the headline for THE SHINAR CONSTRUCTION NEWS!",
            "decoy": "SKYSCRAPER PROJECT HALTED AFTER FOREMAN ORDERS 'BRICKS' AND RECEIVES 'BANANAS'",
            "bot_answers": [
                "TOWER CREW REPORTS 'COMMUNICATION ISSUES' IN DOZENS OF LANGUAGES",
                "TRANSLATORS SUDDENLY IN VERY HIGH DEMAND",
                "WORLD'S FIRST GROUP PROJECT ENDS EXACTLY LIKE ALL THE OTHERS",
            ],
            "scripture_note": (
                "Genesis 11:9 - It was called Babel, because there the Lord confused the language of all the earth."
            ),
            "announcer": (
                "Extra! Extra! Nobody on the tower can understand each other! Write the headline for The Shinar "
                "Construction News!"
            ),
        },
        {
            "id": "ee_burning_bush",
            "scene": "burning_bush",
            "subprompt": (
                "SCENE: A barefoot shepherd kneels before a desert bush blazing with fire that isn't burning a single "
                "leaf, while his sheep stare in confusion."
            ),
            "prompt": "Write the headline for THE MIDIAN SHEPHERD'S JOURNAL!",
            "decoy": "LOCAL SHRUB ON FIRE FOR HOURS, STILL HAS ALL ITS LEAVES; FIRE DEPARTMENT BAFFLED",
            "bot_answers": [
                "SHEPHERD TAKES OFF SANDALS, RECEIVES MAJOR CAREER CHANGE",
                "BUSH SETS WORLD RECORD FOR LONGEST BURN WITH ZERO DAMAGE",
                "SHEEP REPORT OWNER 'TALKING TO PLANTS' AGAIN",
            ],
            "scripture_note": (
                "Exodus 3:2,5 - The bush was burning yet not consumed, and God said, 'Take off your sandals, for this "
                "is holy ground.'"
            ),
            "announcer": (
                "Extra! Extra! A bush is on fire but not burning up! Write the headline for The Midian Shepherd's "
                "Journal!"
            ),
        },
        {
            "id": "ee_nehemiah_wall",
            "scene": "nehemiah_wall",
            "subprompt": (
                "SCENE: Families stack stones and swing trowels along a nearly finished city wall while a man with a "
                "clipboard checks off day 52 on a calendar."
            ),
            "prompt": "Write the headline for THE JERUSALEM BUILDERS BULLETIN!",
            "decoy": "CITY WALL DONE IN 52 DAYS; LOCAL CONTRACTORS DEMAND TO KNOW HIS SECRET",
            "bot_answers": [
                "NEIGHBORHOOD DIY PROJECT ACTUALLY FINISHED ON TIME, EXPERTS STUNNED",
                "MOCKERS SPEECHLESS AS WALL GOES UP ANYWAY",
                "NEHEMIAH: 'THE PEOPLE HAD A MIND TO WORK'",
            ],
            "scripture_note": "Nehemiah 4:6 - So we built the wall, for the people had a mind to work.",
            "announcer": (
                "Extra! Extra! Jerusalem's wall is finished in record time! Write the headline for The Jerusalem "
                "Builders Bulletin!"
            ),
        },
        {
            "id": "ee_elijah_chariot",
            "scene": "elijah_chariot",
            "subprompt": (
                "SCENE: A blazing chariot pulled by horses of fire swoops down in a whirlwind as Elisha stands below, "
                "reaching up to catch a falling cloak."
            ),
            "prompt": "Write the headline for THE JORDAN VALLEY EVENING NEWS!",
            "decoy": "PROPHET TAKES EXPRESS RIDE HOME; SUCCESSOR GETS TO KEEP THE JACKET",
            "bot_answers": [
                "WHIRLWIND TRAFFIC ADVISORY: FLAMING CHARIOT SPOTTED NEAR THE JORDAN",
                "ELISHA INHERITS CLOAK AND DOUBLE PORTION, ASKS ABOUT THE CHARIOT NEXT",
                "WITNESS SAYS 'MY FATHER, MY FATHER!'",
            ],
            "scripture_note": (
                "2 Kings 2:11-13 - Chariots and horses of fire appeared, Elijah went up by a whirlwind into heaven, "
                "and Elisha took up his cloak."
            ),
            "announcer": (
                "Extra! Extra! A chariot of fire just swooped down by the Jordan! Write the evening news headline!"
            ),
        },
        {
            "id": "ee_handwriting_wall",
            "scene": "handwriting_wall",
            "subprompt": (
                "SCENE: At a glittering palace feast, a glowing hand writes strange words on the plaster wall while "
                "the king's knees knock and guests drop their goblets."
            ),
            "prompt": "Write the headline for THE BABYLON TIMES!",
            "decoy": "KING'S PARTY CRASHED BY MYSTERY GRAFFITI ARTIST; NOBODY SAW THE REST OF HIM",
            "bot_answers": [
                "WRITING ON THE WALL: ROYAL BANQUET ENDS EARLY",
                "PALACE PAINTERS DENY ANY INVOLVEMENT",
                "ONLY ONE MAN CAN READ IT, AND HE WASN'T EVEN INVITED",
            ],
            "scripture_note": (
                "Daniel 5:5-6 - The fingers of a man's hand wrote on the palace wall, and the king's knees knocked "
                "together."
            ),
            "announcer": (
                "Extra! Extra! Mysterious writing has appeared on the palace wall! Write the headline for The Babylon "
                "Times!"
            ),
        },
        {
            "id": "ee_storm_calmed",
            "scene": "storm_calmed",
            "subprompt": (
                "SCENE: A small fishing boat floats on a glassy-calm lake right after a huge storm; soaked disciples "
                "stare open-mouthed at Jesus, just up from a nap on a cushion."
            ),
            "prompt": "Write the headline for THE GALILEE MARINE WEATHER REPORT!",
            "decoy": "STORM CANCELLED BY PASSENGER WHO WAS ASLEEP FOR MOST OF IT",
            "bot_answers": [
                "FORECAST UPDATE: WIND AND WAVES OBEY ONE MAN; WEATHERMEN SPEECHLESS",
                "FISHERMEN ASK: 'WHO IS THIS?'",
                "SEA GOES FROM HURRICANE TO BATHTUB IN ONE SENTENCE",
            ],
            "scripture_note": (
                "Mark 4:39-41 - Jesus said to the sea, 'Peace! Be still!' and there was a great calm. 'Who then is "
                "this, that even the wind and the sea obey him?'"
            ),
            "announcer": (
                "Extra! Extra! The storm on Galilee stopped in an instant! Write the marine weather headline!"
            ),
        },
        {
            "id": "ee_palm_parade",
            "scene": "palm_parade",
            "subprompt": (
                "SCENE: Crowds wave palm branches and lay cloaks on the road as Jesus rides a young donkey through "
                "the city gate, with kids shouting Hosanna."
            ),
            "prompt": "Write the front-page headline for THE JERUSALEM DAILY!",
            "decoy": "DONKEY GETS RED-CARPET WELCOME; HORSES DEMAND EQUAL TREATMENT",
            "bot_answers": [
                "PALM BRANCH SALES SKYROCKET AS HOSANNA PARADE FILLS CITY",
                "COATS ON THE ROAD: CITY LAUNDRY DAY POSTPONED",
                "LOCAL DONKEY HAS BEST DAY OF ITS LIFE",
            ],
            "scripture_note": (
                "Matthew 21:8-9 - The crowds spread their cloaks and branches on the road, shouting, 'Hosanna to the "
                "Son of David!'"
            ),
            "announcer": (
                "Extra! Extra! A parade of palm branches fills Jerusalem! Write the front-page headline for The "
                "Jerusalem Daily!"
            ),
        },
        {
            "id": "ee_empty_tomb",
            "scene": "empty_tomb",
            "subprompt": (
                "SCENE: At sunrise, a huge stone sits rolled away from an empty garden tomb as women holding spice "
                "jars gaze at a shining angel in dazzling white."
            ),
            "prompt": "Write the headline for THE GARDEN TOMB GAZETTE!",
            "decoy": "BIGGEST STONE IN TOWN ROLLED AWAY; ANGEL TAKES A SEAT ON IT",
            "bot_answers": [
                "HE IS NOT HERE: GREATEST NEWS IN HISTORY BREAKS AT DAWN",
                "WOMEN ARRIVE WITH SPICES, LEAVE WITH GOOD NEWS",
                "GUARDS STUNNED; GARDEN TOMB OFFICIALLY EMPTY",
            ],
            "scripture_note": (
                "Matthew 28:2,6 - An angel rolled back the stone and sat on it, and said, 'He is not here, for he has "
                "risen, as he said.'"
            ),
            "announcer": (
                "Extra! Extra! The best news ever just broke at sunrise! Write the headline for The Garden Tomb "
                "Gazette!"
            ),
        },
        {
            "id": "ee_pentecost_tongues",
            "scene": "pentecost_tongues",
            "subprompt": (
                "SCENE: Disciples stand in a crowded street with small flames resting over their heads as visitors "
                "from many nations listen in astonishment."
            ),
            "prompt": "Write the headline for THE JERUSALEM WORLD NEWS!",
            "decoy": "GALILEAN FISHERMEN SUDDENLY FLUENT IN A DOZEN LANGUAGES, NO APP REQUIRED",
            "bot_answers": [
                "3,000 JOIN IN ONE DAY; CHURCH NEEDS A MUCH BIGGER POTLUCK",
                "FLAMES ON HEADS, NOT ONE HAIR SINGED",
                "VISITORS FROM EVERY NATION HEAR SERMON IN THEIR OWN LANGUAGE",
            ],
            "scripture_note": (
                "Acts 2:3-6 - Tongues as of fire rested on each of them, and every visitor heard them speaking in his "
                "own language."
            ),
            "announcer": (
                "Extra! Extra! Something amazing is happening in the streets of Jerusalem! Write the world news "
                "headline!"
            ),
        },
        {
            "id": "ee_paul_basket",
            "scene": "paul_basket",
            "subprompt": (
                "SCENE: Under the moonlight, friends lower a nervous Paul in a large woven basket over the Damascus "
                "city wall on a long rope."
            ),
            "prompt": "Write the headline for THE DAMASCUS DISPATCH!",
            "decoy": "LOCAL MAN DELIVERED LIKE A LOAF OF BREAD; ESCAPES CITY IN STYLE",
            "bot_answers": [
                "BASKET WEAVERS REPORT RECORD WEIGHT TEST",
                "NEW BELIEVER TAKES THE 'SCENIC ROUTE' OUT OF TOWN",
                "GATE GUARDS WATCH FRONT DOOR ALL NIGHT; MISS THE WALL",
            ],
            "scripture_note": (
                "Acts 9:25 - His disciples took him by night and let him down through an opening in the wall, "
                "lowering him in a basket."
            ),
            "announcer": (
                "Extra! Extra! A daring basket escape over the Damascus wall! Write the headline for The Damascus "
                "Dispatch!"
            ),
        },
        {
            "id": "ee_peter_prison",
            "scene": "peter_prison",
            "subprompt": (
                "SCENE: A shining angel leads a sleepy Peter past snoring guards as a big iron gate swings open by "
                "itself and broken chains lie on the floor."
            ),
            "prompt": "Write the headline for THE JERUSALEM POLICE BLOTTER!",
            "decoy": "PRISON SECURITY FAILS; INMATE STROLLS OUT WITH ANGEL ESCORT",
            "bot_answers": [
                "IRON GATE OPENS ALL BY ITSELF; LOCKSMITHS BAFFLED",
                "GUARDS SLEEP THROUGH THE WHOLE THING",
                "PRAYER MEETING ANSWERED BEFORE IT EVEN ENDS",
            ],
            "scripture_note": (
                "Acts 12:7,10 - An angel woke Peter, the chains fell off his hands, and the iron gate opened for them "
                "of its own accord."
            ),
            "announcer": (
                "Extra! Extra! A prisoner has walked right out of the Jerusalem jail! Write the headline for The "
                "Police Blotter!"
            ),
        },
        {
            "id": "ee_ark_rainbow",
            "scene": "ark_rainbow",
            "subprompt": (
                "SCENE: Animals pour out of the Ark two by two onto a muddy mountainside as an enormous rainbow "
                "stretches across the clearing sky."
            ),
            "prompt": "Write the headline for THE ARARAT MOUNTAIN NEWS!",
            "decoy": "ARK DOCKS ON MOUNTAINTOP; THOUSANDS OF ANIMALS SPRINT FOR THE NEAREST BUSH",
            "bot_answers": [
                "SKY GETS A NEW ACCESSORY: GOD'S PROMISE IN SEVEN COLORS",
                "FAMILY OF EIGHT ENDS YEAR-LONG CRUISE, NEEDS A BATH",
                "NO MORE WORLDWIDE FLOODS, SAYS RAINBOW; UMBRELLA SALES DROP",
            ],
            "scripture_note": (
                "Genesis 9:13 - 'I have set my bow in the cloud, and it shall be a sign of the covenant between me "
                "and the earth.'"
            ),
            "announcer": (
                "Extra! Extra! The Ark has landed and the sky has a rainbow! Write the headline for The Ararat "
                "Mountain News!"
            ),
        },
        {
            "id": "ee_david_harp",
            "scene": "david_harp",
            "subprompt": (
                "SCENE: A young shepherd plays a harp in the royal court as a very grumpy King Saul slowly relaxes on "
                "his throne."
            ),
            "prompt": "Write the entertainment headline for THE GIBEAH COURT CIRCULAR!",
            "decoy": "SHEPHERD BOY'S PLAYLIST CALMS ROYAL MOOD; PALACE BAND ASKS FOR TIPS",
            "bot_answers": [
                "PALACE HIRES TEEN HARPIST; KING FEELS 'REFRESHED AND WELL'",
                "LOCAL SHEEP UPSET TO MISS THE CONCERT",
                "SHEPHERD'S HARP TOPS PALACE CHARTS FOR TWELVE STRAIGHT WEEKS",
            ],
            "scripture_note": "1 Samuel 16:23 - David took the harp and played, and Saul was refreshed and was well.",
            "announcer": "Extra! Extra! A shepherd boy is playing for the king! Write the entertainment headline!",
        },
        {
            "id": "ee_baby_moses",
            "scene": "baby_moses",
            "subprompt": (
                "SCENE: Pharaoh's daughter and her maidens find a baby in a little waterproof basket among the reeds "
                "while his big sister peeks out from behind tall papyrus."
            ),
            "prompt": "Write the headline for THE NILE RIVER REPORT!",
            "decoy": "PRINCESS FINDS BABY IN BASKET; DECIDES TO KEEP THE BABY, NOT THE BASKET",
            "bot_answers": [
                "WATERPROOF BASKET PASSES FIRST RIVER TEST",
                "BIG SISTER FINDS PERFECT BABYSITTER: IT'S MOM",
                "ROYAL BATH TIME INTERRUPTED BY SMALLEST BOAT ON THE NILE",
            ],
            "scripture_note": (
                "Exodus 2:5-8 - Pharaoh's daughter saw the basket among the reeds, and the baby's sister offered to "
                "fetch a nurse, his own mother."
            ),
            "announcer": (
                "Extra! Extra! A baby has been found floating on the Nile! Write the headline for The Nile River "
                "Report!"
            ),
        },
        {
            "id": "ee_full_nets",
            "scene": "full_nets",
            "subprompt": (
                "SCENE: Tired fishermen strain to haul two boats so full of flopping fish they're starting to sink, "
                "after a teacher told them to try the deep water."
            ),
            "prompt": "Write the headline for THE CAPERNAUM FISHING NEWS!",
            "decoy": "TIRED FISHERMEN TRY ONE MORE CAST; NOW NEED A MUCH BIGGER BOAT",
            "bot_answers": [
                "NETS BREAK UNDER RECORD CATCH; SIMON CALLS FOR BACKUP",
                "ALL-NIGHT SHIFT CATCHES NOTHING, ONE MORNING CAST CATCHES EVERYTHING",
                "LOCAL FISH MARKET OVERWHELMED",
            ],
            "scripture_note": (
                "Luke 5:5-7 - 'We toiled all night and took nothing! But at your word I will let down the nets.' The "
                "boats filled until they began to sink."
            ),
            "announcer": (
                "Extra! Extra! The biggest catch Galilee has ever seen! Write the headline for The Capernaum Fishing "
                "News!"
            ),
        },
        {
            "id": "ee_children_welcome",
            "scene": "children_welcome",
            "subprompt": (
                "SCENE: A crowd of laughing kids climbs into Jesus' lap while the disciples, holding up a 'SCHEDULE "
                "FULL' sign, look very surprised."
            ),
            "prompt": "Write the headline for THE JUDEA FAMILY PAGES!",
            "decoy": "DISCIPLES TRY TO CANCEL KIDS' MEET-AND-GREET; JESUS OVERRULES",
            "bot_answers": [
                "KIDS SKIP THE LINE; TEACHER SAYS 'LET THEM COME'",
                "BIGGEST GROUP HUG IN JUDEAN HISTORY",
                "DISCIPLES' CALENDAR CLEARED FOR STORY TIME",
            ],
            "scripture_note": (
                "Mark 10:14 - 'Let the little children come to me; do not hinder them, for to such belongs the "
                "kingdom of God.'"
            ),
            "announcer": (
                "Extra! Extra! The kids have taken over the meeting! Write the headline for The Judea Family Pages!"
            ),
        },
        {
            "id": "ee_frog_plague",
            "scene": "frog_plague",
            "subprompt": (
                "SCENE: Frogs hop all over Pharaoh's palace: one in the royal bed, one in the bread dough, and one "
                "sitting on the crown while servants chase them with brooms."
            ),
            "prompt": "Write the headline for THE MEMPHIS MORNING NEWS!",
            "decoy": "PALACE DECLARES ITSELF A POND; PHARAOH ASKS FOR 'ONE MORE NIGHT WITH THE FROGS'",
            "bot_answers": [
                "FROGS FOUND IN KING'S BED, OVENS AND KNEADING BOWLS",
                "EGYPT'S FROG POPULATION UP TEN THOUSAND PERCENT",
                "ROYAL CHEF FINDS UNEXPECTED GUEST IN BREAD DOUGH",
            ],
            "scripture_note": (
                "Exodus 8:3,10 - Frogs came into Pharaoh's bedroom, ovens and kneading bowls, and when asked when to "
                "remove them, Pharaoh said, 'Tomorrow.'"
            ),
            "announcer": (
                "Extra! Extra! Frogs have taken over Pharaoh's palace! Write the headline for The Memphis Morning "
                "News!"
            ),
        },
        {
            "id": "ee_shepherds_angels",
            "scene": "shepherds_angels",
            "subprompt": (
                "SCENE: Night-shift shepherds shield their eyes as the sky fills with a shining choir of angels "
                "singing over their very startled sheep."
            ),
            "prompt": "Write the headline for THE BETHLEHEM HILLS HERALD!",
            "decoy": "NIGHT SHIFT INTERRUPTED BY CHOIR OF THOUSANDS; SHEEP DEMAND ENCORE",
            "bot_answers": [
                "SHEPHERDS LEAVE FLOCK TO VISIT NEWBORN IN A MANGER",
                "GLORY TO GOD IN THE HIGHEST: SKY LIGHTS UP OVER FIELDS",
                "LOCAL SHEPHERDS TELL EVERYONE; NOBODY BELIEVES HOW BRIGHT IT WAS",
            ],
            "scripture_note": (
                "Luke 2:13-14 - Suddenly there was with the angel a multitude of the heavenly host praising God: "
                "'Glory to God in the highest!'"
            ),
            "announcer": (
                "Extra! Extra! The night sky over Bethlehem is full of angels! Write the headline for The Bethlehem "
                "Hills Herald!"
            ),
        },
        {
            "id": "ee_boy_jesus_temple",
            "scene": "boy_jesus_temple",
            "subprompt": (
                "SCENE: Twelve-year-old Jesus sits among amazed gray-bearded teachers in the Temple while Mary and "
                "Joseph rush in looking relieved and a little frazzled."
            ),
            "prompt": "Write the headline for THE TEMPLE COURTS CHRONICLE!",
            "decoy": "PARENTS SEARCH THREE DAYS; FIND SON AT TEMPLE ACING THE TEACHERS' QUIZ",
            "bot_answers": [
                "TWELVE-YEAR-OLD AMAZES SCHOLARS WITH HIS ANSWERS",
                "FAMILY ROAD-TRIP HEADCOUNT PROCEDURES UNDER REVIEW",
                "BOY ASKS PARENTS: 'DIDN'T YOU KNOW I'D BE IN MY FATHER'S HOUSE?'",
            ],
            "scripture_note": (
                "Luke 2:46-47,49 - After three days they found him in the temple, and all who heard him were amazed "
                "at his understanding."
            ),
            "announcer": (
                "Extra! Extra! A twelve-year-old is amazing the Temple teachers! Write the headline for The Temple "
                "Courts Chronicle!"
            ),
        },
        {
            "id": "ee_quiet_temple",
            "scene": "quiet_temple",
            "subprompt": (
                "SCENE: Workers polish golden doors and cedar walls inside King Solomon's magnificent new Temple, and "
                "the construction site is strangely, totally silent."
            ),
            "prompt": "Write the headline for THE JERUSALEM HOME AND GARDEN!",
            "decoy": "WORLD'S QUIETEST CONSTRUCTION SITE: NOT ONE HAMMER HEARD",
            "bot_answers": [
                "GOLD EVERYWHERE: TEMPLE INTERIOR DESIGN REACHES NEW HEIGHTS",
                "CEDARS OF LEBANON ARRIVE; SOME ASSEMBLY REQUIRED",
                "NEIGHBORS THANKFUL FOR 'PRE-CUT STONE' BUILDING POLICY",
            ],
            "scripture_note": (
                "1 Kings 6:7 - The stone was prepared at the quarry, so neither hammer nor axe nor any tool of iron "
                "was heard in the house while it was being built."
            ),
            "announcer": (
                "Extra! Extra! Solomon's Temple is going up without a sound! Write the headline for The Jerusalem "
                "Home and Garden!"
            ),
        },
        {
            "id": "ee_little_cloud",
            "scene": "little_cloud",
            "subprompt": (
                "SCENE: On a mountaintop, a servant points at a tiny cloud no bigger than a hand rising from the sea, "
                "while Elijah tucks up his robe to run."
            ),
            "prompt": "Write the headline for THE CARMEL WEATHER WATCH!",
            "decoy": "TINY CLOUD ENDS BIG DROUGHT; WEATHER APPS DELETE THEMSELVES",
            "bot_answers": [
                "PROPHET OUTRUNS ROYAL CHARIOT IN THE RAIN",
                "SERVANT CHECKS SKY SEVEN TIMES; SEVENTH TIME'S THE CHARM",
                "CLOUD THE SIZE OF A HAND BRINGS RAIN THE SIZE OF A NATION",
            ],
            "scripture_note": (
                "1 Kings 18:44-46 - 'A little cloud like a man's hand is rising from the sea!' Then heavy rain came, "
                "and Elijah ran ahead of Ahab's chariot."
            ),
            "announcer": (
                "Extra! Extra! A tiny cloud just appeared over Mount Carmel! Write the headline for The Carmel "
                "Weather Watch!"
            ),
        },
        {
            "id": "ee_philippi_jail",
            "scene": "philippi_jail",
            "subprompt": (
                "SCENE: At midnight, Paul and Silas sing hymns in a dungeon as an earthquake shakes every door open "
                "and chains fall off, while the jailer holds up a lamp."
            ),
            "prompt": "Write the late-night headline for THE PHILIPPI NIGHT EDITION!",
            "decoy": "PRISONERS' MIDNIGHT HYMN SING ROCKS THE JAIL, LITERALLY",
            "bot_answers": [
                "EVERY DOOR OPENS, NOBODY LEAVES; JAILER'S WHOLE FAMILY BAPTIZED",
                "CONCERT REVIEW: 'GREAT ACOUSTICS IN THE INNER PRISON'",
                "LOCAL JAILER ASKS 'WHAT MUST I DO TO BE SAVED?'",
            ],
            "scripture_note": (
                "Acts 16:25-26 - At midnight Paul and Silas were praying and singing hymns, and an earthquake opened "
                "all the doors."
            ),
            "announcer": (
                "Extra! Extra! A midnight earthquake in Philippi! Write the headline for The Philippi Night Edition!"
            ),
        },
        {
            "id": "ee_samson_gates",
            "scene": "samson_gates",
            "subprompt": (
                "SCENE: At midnight, Samson strolls up a hill carrying the entire city gate of Gaza, posts and all, "
                "on his shoulders while sleepy guards stare."
            ),
            "prompt": "Write the headline for THE GAZA CITY HALL REPORT!",
            "decoy": "CITY GATE REPORTED MISSING; LAST SEEN WALKING UPHILL",
            "bot_answers": [
                "STRONGMAN TAKES GATE, POSTS AND BAR; LEAVES TOWN VERY DRAFTY",
                "GAZA WAKES UP TO A VERY OPEN-CONCEPT CITY",
                "LOCALS ASK: WHY LOCK A GATE SOMEONE CAN CARRY?",
            ],
            "scripture_note": (
                "Judges 16:3 - At midnight Samson took hold of the doors of the city gate and the two posts, and "
                "carried them to the top of the hill."
            ),
            "announcer": (
                "Extra! Extra! The city gate of Gaza has gone for a walk! Write the headline for The Gaza City Hall "
                "Report!"
            ),
        },
    ],
    "sub_the_scroll": [
        {
            "id": "sts_noah_line",
            "scene": "noah_line",
            "speaker": "Noah",
            "subprompt": (
                "SCENE: Noah stands at the Ark's ramp with a clipboard as two giraffes, two elephants, and two very "
                "impatient skunks wait in line."
            ),
            "prompt": "What is Noah saying to the animals waiting in line?",
            "decoy": "Skunks, you're in the very back. The VERY back.",
            "bot_answers": [
                "Two of each, please! Rabbits, I SEE you trying to sneak in a third!",
                "Elephants on the bottom deck, and nobody sits next to the porcupines",
                "Welcome aboard! Today's forecast: forty days of rain",
            ],
            "scripture_note": (
                "Genesis 7:8-9 - Two and two, male and female, went into the ark with Noah, as God had commanded."
            ),
            "announcer": "Sub the Scroll! Noah is boarding the animals. What is he saying to the line?",
        },
        {
            "id": "sts_sarah_laughs",
            "scene": "sarah_laughs",
            "speaker": "Sarah",
            "subprompt": (
                "SCENE: Inside a tent, ninety-year-old Sarah covers her mouth trying not to laugh while three "
                "visitors outside tell Abraham she'll have a baby next year."
            ),
            "prompt": "What is Sarah whispering behind the tent flap?",
            "decoy": "A baby? At my age? I can't even remember where I put my sandals!",
            "bot_answers": [
                "Abraham, you'd better build a crib AND a rocking chair",
                "Are we sure they didn't mean the great-grandkids?",
                "I'm not laughing! This is just my very serious face!",
            ],
            "scripture_note": (
                "Genesis 18:12-14 - Sarah laughed to herself, and the Lord asked, 'Is anything too hard for the "
                "Lord?'"
            ),
            "announcer": "Sarah is listening behind the tent flap! What is she whispering?",
        },
        {
            "id": "sts_joseph_bowing",
            "scene": "joseph_bowing",
            "speaker": "Joseph",
            "subprompt": (
                "SCENE: Joseph, now governor of Egypt in royal robes, tries to keep a straight face as his ten older "
                "brothers bow low, not recognizing him."
            ),
            "prompt": "What is Joseph thinking while his brothers bow?",
            "decoy": "Wow, that dream about the bowing bundles of wheat is going EXACTLY as planned",
            "bot_answers": [
                "Don't laugh, don't laugh, they still don't recognize me",
                "Remember the coat, guys? Anybody? Nope? Okay",
                "Reuben looks older, Judah looks hungrier, and I look fabulous",
            ],
            "scripture_note": (
                "Genesis 42:8-9 - Joseph recognized his brothers, but they did not recognize him, and Joseph "
                "remembered his dreams."
            ),
            "announcer": "Joseph's brothers are bowing to him and have no idea who he is! What is Joseph thinking?",
        },
        {
            "id": "sts_pharaoh_snakes",
            "scene": "pharaoh_snakes",
            "speaker": "Pharaoh",
            "subprompt": (
                "SCENE: In Pharaoh's throne room, Aaron's staff has become a snake and is gulping down the magicians' "
                "snakes while they stare at their empty hands."
            ),
            "prompt": "What is Pharaoh muttering to his magicians?",
            "decoy": "You had ONE job, and it was 'don't let his snake eat your snakes'",
            "bot_answers": [
                "Can we just order new staffs from the catalog?",
                "I'm taking those out of your paychecks",
                "Okay, but can YOUR snake do that?",
            ],
            "scripture_note": (
                "Exodus 7:12 - The magicians' staffs became snakes too, but Aaron's staff swallowed up their staffs."
            ),
            "announcer": "Trouble in Pharaoh's throne room! What is Pharaoh muttering to his magicians?",
        },
        {
            "id": "sts_rahab_rope",
            "scene": "rahab_rope",
            "speaker": "Rahab",
            "subprompt": (
                "SCENE: Rahab lowers a scarlet cord from her window in the Jericho wall to two nervous spies who have "
                "been hiding under stalks of flax on her roof."
            ),
            "prompt": "What is Rahab saying to the two spies?",
            "decoy": "Next time you're spying, maybe don't ask the front desk for directions",
            "bot_answers": [
                "Hide in the flax! Yes, it's itchy. No, you can't complain",
                "Remember me and my red rope when you come back, okay?",
                "Head for the hills and wait three days. Trust me, I know this town",
            ],
            "scripture_note": (
                "Joshua 2:15-18 - Rahab let the spies down by a rope through her window, and they told her to tie a "
                "scarlet cord in it."
            ),
            "announcer": "The spies are escaping from Jericho! What is Rahab telling them?",
        },
        {
            "id": "sts_gideon_winepress",
            "scene": "gideon_winepress",
            "speaker": "Gideon",
            "subprompt": (
                "SCENE: Gideon hides in a winepress threshing wheat when an angel appears and calls him a mighty "
                "warrior, and Gideon looks around to see who it's talking to."
            ),
            "prompt": "What is Gideon saying to the angel?",
            "decoy": "Mighty warrior? Me? I'm hiding in a grape tub!",
            "bot_answers": [
                "I think you have the wrong address",
                "Can I get that in writing? Maybe on a fleece?",
                "Shh! I'm threshing wheat undercover!",
            ],
            "scripture_note": (
                "Judges 6:11-12 - The angel of the Lord appeared to Gideon and said, 'The Lord is with you, O mighty "
                "man of valor.'"
            ),
            "announcer": "An angel just called Gideon a mighty warrior! What is Gideon saying back?",
        },
        {
            "id": "sts_samuel_night",
            "scene": "samuel_night",
            "speaker": "Young Samuel",
            "subprompt": (
                "SCENE: Young Samuel in his pajamas stands beside old Eli's bed in the dark temple for the third time "
                "tonight, rubbing his sleepy eyes."
            ),
            "prompt": "What is Samuel saying to Eli on visit number three?",
            "decoy": "Okay, if it's not you calling me, then WHO is?",
            "bot_answers": [
                "You called? No? Okay, I'll just stand here until you do",
                "Eli, I promise I'm not making this up",
                "Third time's the charm! Are you SURE you didn't say my name?",
            ],
            "scripture_note": (
                "1 Samuel 3:8-10 - Eli realized the Lord was calling the boy, and Samuel answered, 'Speak, for your "
                "servant hears.'"
            ),
            "announcer": "It's the middle of the night and Samuel is at Eli's bed again! What is he saying?",
        },
        {
            "id": "sts_david_armor",
            "scene": "david_armor",
            "speaker": "Young David",
            "subprompt": (
                "SCENE: Young David wobbles in King Saul's oversized bronze armor, the helmet slipping over his eyes, "
                "while soldiers try very hard not to laugh."
            ),
            "prompt": "What is David saying in Saul's giant armor?",
            "decoy": "This is great, but I can't see, can't walk, and I'm pretty sure I'm facing the wrong way",
            "bot_answers": [
                "Thanks, King Saul, but I'll stick with my sling and my sandals",
                "Does this come in a kid's size?",
                "I sound like a pots-and-pans drawer when I walk",
            ],
            "scripture_note": (
                "1 Samuel 17:38-39 - Saul dressed David in his armor, but David said, 'I cannot go with these, for I "
                "have not tested them,' and took them off."
            ),
            "announcer": "David is trying on the king's armor! What is he saying?",
        },
        {
            "id": "sts_naaman_dips",
            "scene": "naaman_dips",
            "speaker": "Naaman",
            "subprompt": (
                "SCENE: A proud army commander in a fancy cape stands knee-deep in the muddy Jordan River, holding up "
                "five fingers while his servants cheer him on to dip again."
            ),
            "prompt": "What is Naaman grumbling between dips?",
            "decoy": "Two more dips. If this is a prank, Elisha is getting a very stern letter.",
            "bot_answers": [
                "Back home in Damascus our rivers are much cleaner, just saying",
                "Was that five or four? Don't make me start over!",
                "If this works, I'm never complaining about bath time again",
            ],
            "scripture_note": (
                "2 Kings 5:12,14 - Naaman grumbled about the Jordan, but he dipped seven times and his skin became "
                "like a little child's."
            ),
            "announcer": "Naaman is dipping in the Jordan River and he is NOT happy about it! What is he grumbling?",
        },
        {
            "id": "sts_esther_scepter",
            "scene": "esther_scepter",
            "speaker": "King Ahasuerus",
            "subprompt": (
                "SCENE: Queen Esther steps nervously into the throne room without an invitation as the king pauses, "
                "then smiles and holds out his golden scepter."
            ),
            "prompt": "What is the king saying as he holds out the scepter?",
            "decoy": "Queen Esther! Come in, come in. Is this about the banquet menu?",
            "bot_answers": [
                "Ask for anything, up to half my kingdom. Except my snack drawer",
                "You look radiant! Also, how did you get past security?",
                "Whatever it is, the answer is yes. Probably.",
            ],
            "scripture_note": (
                "Esther 5:2-3 - The king held out the golden scepter and said, 'What is your request? It shall be "
                "given you, even to half of my kingdom.'"
            ),
            "announcer": "Esther has walked into the throne room! What is the king saying?",
        },
        {
            "id": "sts_daniel_veggies",
            "scene": "daniel_veggies",
            "speaker": "Daniel",
            "subprompt": (
                "SCENE: In the Babylonian palace dining hall, Daniel and his three friends happily crunch vegetables "
                "while other young men stare at their rich royal feast."
            ),
            "prompt": "What is Daniel saying about his vegetable plate?",
            "decoy": "Ten-day challenge: veggies and water. Check back with us, cake-eaters!",
            "bot_answers": [
                "Pass the carrots! The king's pastries can wait",
                "We look healthier already. Must be the broccoli",
                "Salad for breakfast, salad for lunch, salad for victory",
            ],
            "scripture_note": (
                "Daniel 1:12,15 - Daniel asked for vegetables and water for ten days, and at the end they looked "
                "healthier than all the others."
            ),
            "announcer": "Daniel is eating vegetables at the king's feast! What is he saying about his plate?",
        },
        {
            "id": "sts_shepherds_manger",
            "scene": "shepherds_manger",
            "speaker": "The Youngest Shepherd",
            "subprompt": (
                "SCENE: Shepherds tiptoe into a Bethlehem stable, one clutching a lamb, and peek at the newborn baby "
                "in a manger while a cow watches curiously."
            ),
            "prompt": "What is the youngest shepherd whispering to the others?",
            "decoy": "Shh! Everybody act natural. Also, who left the sheep with the angels?",
            "bot_answers": [
                "I brought a lamb. Is that a good baby gift?",
                "The angel said manger, and they meant MANGER",
                "Tiptoe, everyone! The cow is giving us a look",
            ],
            "scripture_note": (
                "Luke 2:15-16 - The shepherds went with haste and found Mary and Joseph, and the baby lying in a "
                "manger."
            ),
            "announcer": "The shepherds have arrived at the stable! What is the youngest one whispering?",
        },
        {
            "id": "sts_john_snack",
            "scene": "john_snack",
            "speaker": "John the Baptist",
            "subprompt": (
                "SCENE: John the Baptist, wearing a scratchy camel-hair coat, offers a curious crowd a snack from a "
                "jar labeled LOCUSTS AND WILD HONEY."
            ),
            "prompt": "What is John the Baptist saying as he offers his snack?",
            "decoy": "Crunchy on the outside, sweet on the inside. Just like a changed heart!",
            "bot_answers": [
                "Try one! They're locally sourced. VERY locally",
                "The honey part is great. The locust part builds character",
                "Prepare the way of the Lord, and also your taste buds",
            ],
            "scripture_note": (
                "Matthew 3:4 - John wore a garment of camel's hair, and his food was locusts and wild honey."
            ),
            "announcer": "John the Baptist is sharing his favorite snack! What is he saying?",
        },
        {
            "id": "sts_nicodemus_night",
            "scene": "nicodemus_night",
            "speaker": "Nicodemus",
            "subprompt": (
                "SCENE: Late at night, Nicodemus tiptoes up to a house with a lantern, glancing around nervously "
                "before knocking to ask Jesus some questions."
            ),
            "prompt": "What is Nicodemus whispering at the door?",
            "decoy": "Psst, it's me, a very important Pharisee. Totally not here. Can I come in?",
            "bot_answers": [
                "I have just a few questions. Okay, maybe forty",
                "Born again? Like, how small do I have to get?",
                "Nobody saw me, right? Great. Do you have any tea?",
            ],
            "scripture_note": (
                "John 3:2,4 - Nicodemus came to Jesus by night and asked, 'How can a man be born when he is old?'"
            ),
            "announcer": "Someone is knocking on the door in the middle of the night! What is Nicodemus whispering?",
        },
        {
            "id": "sts_roof_hole",
            "scene": "roof_hole",
            "speaker": "The Homeowner",
            "subprompt": (
                "SCENE: Four friends lower a man on a mat through a big hole in the roof into a crowded house while "
                "dust falls on everyone and the homeowner stares at his ceiling."
            ),
            "prompt": "What is the homeowner saying about his roof?",
            "decoy": "I said 'come on in,' not 'come on THROUGH'!",
            "bot_answers": [
                "Who's paying for the roof tiles? Anyone? Hello?",
                "Well, at least now we have a skylight",
                "Next time, guys, there's a door. Right over there.",
            ],
            "scripture_note": (
                "Mark 2:3-4 - Unable to get near because of the crowd, four friends removed the roof and let down "
                "their friend on his mat."
            ),
            "announcer": "There's a brand new hole in the roof! What is the homeowner saying?",
        },
        {
            "id": "sts_rhoda_gate",
            "scene": "rhoda_gate",
            "speaker": "Rhoda",
            "subprompt": (
                "SCENE: A servant girl named Rhoda runs back into a packed prayer meeting, leaving the gate still "
                "locked while someone keeps knocking outside."
            ),
            "prompt": "What is Rhoda saying to the prayer group?",
            "decoy": "Our prayers worked! Peter's here! No, I didn't let him in. Why?",
            "bot_answers": [
                "Everybody stop praying for Peter, he's at the gate!",
                "I am NOT imagining it! That's definitely his knock",
                "He's outside! Somebody else open it, I'm too excited",
            ],
            "scripture_note": (
                "Acts 12:13-16 - Rhoda recognized Peter's voice and in her joy ran in without opening the gate, while "
                "Peter kept knocking."
            ),
            "announcer": "Somebody is knocking at the gate! What is Rhoda telling the prayer group?",
        },
        {
            "id": "sts_ethiopian_chariot",
            "scene": "ethiopian_chariot",
            "speaker": "The Ethiopian Official",
            "subprompt": (
                "SCENE: An official from Ethiopia rides in a chariot reading a scroll of Isaiah aloud as Philip jogs "
                "right alongside to keep up."
            ),
            "prompt": "What is the official saying to Philip, who is running beside his chariot?",
            "decoy": "You're keeping up with my chariot on foot? What did you have for breakfast?",
            "bot_answers": [
                "Perfect timing! Do you know who Isaiah is talking about?",
                "Hop in! There's water up ahead, just saying",
                "I've read this paragraph six times. Please help.",
            ],
            "scripture_note": (
                "Acts 8:30-31 - Philip asked, 'Do you understand what you are reading?' and he said, 'How can I, "
                "unless someone guides me?'"
            ),
            "announcer": "Philip is running alongside a chariot! What is the official saying?",
        },
        {
            "id": "sts_miriam_tambourine",
            "scene": "miriam_tambourine",
            "speaker": "Miriam",
            "subprompt": (
                "SCENE: On the far shore of the sea, Miriam leads the women in a joyful dance with tambourines while "
                "the men are still wringing out their robes."
            ),
            "prompt": "What is Miriam shouting to the crowd?",
            "decoy": "Grab a tambourine! If you can walk through a sea, you can dance!",
            "bot_answers": [
                "Sing to the Lord! And somebody find the drummer",
                "Big finish, everybody! Five, six, seven, eight!",
                "Dry ground deserves a dance party",
            ],
            "scripture_note": (
                "Exodus 15:20-21 - Miriam took a tambourine, and all the women went out after her with tambourines "
                "and dancing."
            ),
            "announcer": "It's a celebration on the shore! What is Miriam shouting?",
        },
        {
            "id": "sts_isaac_goatskin",
            "scene": "isaac_goatskin",
            "speaker": "Isaac",
            "subprompt": (
                "SCENE: Old Isaac, who can barely see, reaches out to feel Jacob's arms wrapped in hairy goatskins "
                "while Jacob nervously holds out a bowl of stew."
            ),
            "prompt": "What is Isaac saying as he feels those hairy arms?",
            "decoy": "Esau, have you been using a new hair conditioner?",
            "bot_answers": [
                "Son, why do you smell like a petting zoo?",
                "These are the hairiest arms I have ever felt. Well done, son",
                "You sound like Jacob, but you feel like a sweater",
            ],
            "scripture_note": (
                "Genesis 27:22 - Isaac felt him and said, 'The voice is Jacob's voice, but the hands are the hands of "
                "Esau.'"
            ),
            "announcer": "Isaac is checking which son is which! What is he saying about those arms?",
        },
        {
            "id": "sts_queen_sheba",
            "scene": "queen_sheba",
            "speaker": "The Queen of Sheba",
            "subprompt": (
                "SCENE: The Queen of Sheba stands amazed in Solomon's palace, staring at the feast on his table and "
                "his servants' fancy uniforms with her mouth open."
            ),
            "prompt": "What is the Queen of Sheba saying to Solomon?",
            "decoy": "I came with hard questions and I'm leaving with your recipe book",
            "bot_answers": [
                "Okay, the reviews undersold this place. Big time",
                "Who does your decorating? Asking for a palace",
                "They told me half, and honestly I didn't believe that half",
            ],
            "scripture_note": (
                "1 Kings 10:6-7 - The queen said, 'The report was true... and behold, the half was not told me.'"
            ),
            "announcer": "The Queen of Sheba is visiting Solomon's palace! What is she saying?",
        },
        {
            "id": "sts_tertius_scribe",
            "scene": "tertius_scribe",
            "speaker": "Tertius the Scribe",
            "subprompt": (
                "SCENE: Paul paces the room dictating a very long letter while his scribe Tertius shakes out a "
                "cramping hand beside a growing mountain of scrolls."
            ),
            "prompt": "What is Tertius thinking as Paul keeps dictating?",
            "decoy": "Sixteen chapters?! Paul, please, my hand has a hand cramp",
            "bot_answers": [
                "Can we just say 'grace and peace' and call it a day?",
                "He said 'finally' twenty minutes ago",
                "I'm sneaking my own greeting in at the end. He'll never notice",
            ],
            "scripture_note": "Romans 16:22 - 'I Tertius, who wrote this letter, greet you in the Lord.'",
            "announcer": "Paul is dictating another long letter! What is his scribe Tertius thinking?",
        },
        {
            "id": "sts_blind_man_sees",
            "scene": "blind_man_sees",
            "speaker": "The Man Born Blind",
            "subprompt": (
                "SCENE: A man walks back from the pool of Siloam, wiping mud from his eyes and seeing the world for "
                "the very first time as neighbors gasp."
            ),
            "prompt": "What is the man saying as he sees for the first time?",
            "decoy": "Whoa! So THAT'S what my neighbors look like!",
            "bot_answers": [
                "Is the sky always this blue?",
                "I can see! And I can see I need a haircut",
                "Who wants to hear what happened? It's a long story with mud in it",
            ],
            "scripture_note": (
                "John 9:7,25 - He went and washed and came back seeing. 'One thing I know, that though I was blind, "
                "now I see.'"
            ),
            "announcer": "A man born blind can see for the first time! What is he saying?",
        },
        {
            "id": "sts_ananias_gulp",
            "scene": "ananias_gulp",
            "speaker": "Ananias",
            "subprompt": (
                "SCENE: Ananias stands frozen in his doorway after God tells him to go visit Saul, the man everyone "
                "is afraid of, and he points nervously at himself."
            ),
            "prompt": "What is Ananias saying to the Lord?",
            "decoy": "Lord, are we talking about the SAME Saul? Because I've heard stories",
            "bot_answers": [
                "Could You send someone braver? Like, anyone else?",
                "Okay, I'll go. But I'm bringing a casserole as a peace offering",
                "Straight Street, house of Judas. Got it. Gulp.",
            ],
            "scripture_note": (
                "Acts 9:11-15 - Ananias protested that he had heard much about Saul, but the Lord said, 'Go, for he "
                "is a chosen instrument of mine.'"
            ),
            "announcer": "God just sent Ananias to visit Saul! What is Ananias saying?",
        },
        {
            "id": "sts_pastor_lion",
            "scene": "pastor_lion",
            "speaker": "The Pastor",
            "subprompt": (
                "SCENE: A pastor stands at the pulpit trying to give the announcements as a toddler in a lion costume "
                "roars from the front row and the whole church giggles."
            ),
            "prompt": "What is the pastor saying to keep the service going?",
            "decoy": "And that's a perfect introduction to today's sermon on Daniel!",
            "bot_answers": [
                "Looks like the Lion of Judah sent a representative",
                "Let's give our youngest choir member a hand",
                "I'll take that as an amen",
            ],
            "scripture_note": "Psalm 8:2 - Out of the mouths of babies and infants, you have established strength.",
            "announcer": "A tiny lion is roaring in the front row! What is the pastor saying?",
        },
        {
            "id": "sts_dad_minivan",
            "scene": "dad_minivan",
            "speaker": "Dad",
            "subprompt": (
                "SCENE: A dad sits in the driver's seat of a packed minivan in the driveway, checking his watch, "
                "while three kids search the yard for one missing shoe."
            ),
            "prompt": "What is Dad saying right now?",
            "decoy": "We are going to be on time for church, even if one of us goes barefoot",
            "bot_answers": [
                "Lord, grant me patience, and also that shoe",
                "Whoever finds the shoe gets to pick lunch",
                "Remember, everyone, we are joyful. We are JOYFUL.",
            ],
            "scripture_note": "James 1:19 - Let every person be quick to hear, slow to speak, slow to anger.",
            "announcer": "It's Sunday morning and Dad is waiting in the van! What is he saying?",
        },
    ],
    "truth_or_bluff": [
        {
            "id": "tob_methuselah_age",
            "prompt": (
                "According to Genesis 5, Methuselah lived longer than anyone else recorded in the Bible. Exactly how "
                "many years did he live? ______"
            ),
            "truth": "969 years",
            "truth_keywords": [
                "969",
                "nine hundred sixty-nine",
                "nine hundred and sixty-nine",
                "nine hundred sixty nine",
            ],
            "decoy": "930 years (and he looked great for his age)",
            "bot_answers": [
                "912 years",
                "1,001 years",
                "777 years",
            ],
            "scripture_note": (
                "Genesis 5:27 - All the days of Methuselah were 969 years, the longest life recorded in Scripture."
            ),
            "announcer": (
                "Truth or Bluff! How many birthdays did Methuselah celebrate? Write a believable fake number!"
            ),
        },
        {
            "id": "tob_tabitha_name",
            "prompt": (
                "In Acts 9, Peter raised a kind woman in Joppa who made clothes for widows. Her name was Tabitha. "
                "What was her other name? ______"
            ),
            "truth": "Dorcas",
            "truth_keywords": [
                "dorcas",
                "named dorcas",
                "called dorcas",
            ],
            "decoy": "Phoebe",
            "bot_answers": [
                "Lydia",
                "Junia",
                "Chloe",
            ],
            "scripture_note": (
                "Acts 9:36 - In Joppa there was a disciple named Tabitha, which translated means Dorcas, who was full "
                "of good works."
            ),
            "announcer": "Truth or Bluff! Tabitha had a second name in the Book of Acts. Can you bluff your family?",
        },
        {
            "id": "tob_esau_trade",
            "prompt": (
                "In Genesis 25, hungry Esau traded away his birthright to his brother Jacob. What did he get in "
                "exchange? ______"
            ),
            "truth": "Bread and a bowl of red lentil stew",
            "truth_keywords": [
                "lentil stew",
                "red stew",
                "bowl of stew",
                "lentil soup",
                "pottage of lentils",
            ],
            "decoy": "A roasted goat and a new hunting bow",
            "bot_answers": [
                "Three sheep and a silver cup",
                "A jar of honey and some figs",
                "His brother's best pair of sandals",
            ],
            "scripture_note": (
                "Genesis 25:34 - Jacob gave Esau bread and lentil stew, and he ate and drank, and went his way. So "
                "Esau despised his birthright."
            ),
            "announcer": "Truth or Bluff! What did Esau trade his whole birthright for? Write a sneaky bluff!",
        },
        {
            "id": "tob_joseph_price",
            "prompt": (
                "In Genesis 37, Joseph's brothers sold him to a group of traders headed for Egypt. How much did they "
                "sell him for? ______"
            ),
            "truth": "Twenty pieces of silver",
            "truth_keywords": [
                "twenty pieces",
                "20 pieces",
                "twenty shekels",
                "20 shekels",
                "twenty silver",
            ],
            "decoy": "Thirty pieces of silver",
            "bot_answers": [
                "Ten gold coins",
                "Two camels and a donkey",
                "Fifty shekels of bronze",
            ],
            "scripture_note": (
                "Genesis 37:28 - They drew Joseph out of the pit and sold him to the Ishmaelites for twenty shekels "
                "of silver."
            ),
            "announcer": (
                "Truth or Bluff! How much did Joseph's brothers sell him for? Write a believable fake price!"
            ),
        },
        {
            "id": "tob_jacob_rachel",
            "prompt": (
                "In Genesis 29, Jacob agreed to work for Laban so he could marry Rachel. How long did he first agree "
                "to work? ______"
            ),
            "truth": "Seven years (and it felt like only a few days)",
            "truth_keywords": [
                "seven years",
                "7 years",
                "fourteen years",
                "14 years",
            ],
            "decoy": "Three years of shepherding",
            "bot_answers": [
                "One full harvest season",
                "Forty days and forty nights",
                "Twelve months, one for each tribe",
            ],
            "scripture_note": (
                "Genesis 29:20 - Jacob served seven years for Rachel, and they seemed to him but a few days because "
                "of his love for her."
            ),
            "announcer": "Truth or Bluff! How long did Jacob agree to work to marry Rachel? Bluff your family!",
        },
        {
            "id": "tob_hezekiah_sign",
            "prompt": (
                "In 2 Kings 20, God gave sick King Hezekiah an amazing sign that he would get well. What happened? "
                "______"
            ),
            "truth": "The shadow on the stairway went backward ten steps",
            "truth_keywords": [
                "shadow went back",
                "went backward ten",
                "ten steps backward",
                "back ten steps",
                "shadow moved backward",
            ],
            "decoy": "A dove landed on his pillow holding a fig leaf",
            "bot_answers": [
                "The palace lamps relit themselves at midnight",
                "Rain fell on only the palace roof",
                "The temple bells rang three times on their own",
            ],
            "scripture_note": (
                "2 Kings 20:9-11 - Isaiah cried to the Lord, and He brought the shadow back ten steps on the stairway "
                "of Ahaz."
            ),
            "announcer": (
                "Truth or Bluff! God gave King Hezekiah an amazing sign. What was it? Write a convincing fake!"
            ),
        },
        {
            "id": "tob_peter_shadow",
            "prompt": (
                "In Acts 5, sick people were carried into the streets on beds and mats. What were they hoping for as "
                "Peter walked by? ______"
            ),
            "truth": "That at least Peter's shadow would fall on them",
            "truth_keywords": [
                "peter's shadow",
                "his shadow",
                "shadow would fall",
                "shadow fall on",
                "shadow passed over",
            ],
            "decoy": "That Peter would toss them a blessed loaf of bread",
            "bot_answers": [
                "That he would touch the hem of their cloaks",
                "That he would sprinkle them with Jordan water",
                "That he would stop and sing a psalm over them",
            ],
            "scripture_note": (
                "Acts 5:15 - They carried the sick into the streets so that as Peter came by, at least his shadow "
                "might fall on some of them."
            ),
            "announcer": "Truth or Bluff! What were the crowds hoping for as Peter walked down the street?",
        },
        {
            "id": "tob_paul_handkerchiefs",
            "prompt": (
                "Acts 19 says God did extraordinary miracles through Paul in Ephesus. What items were carried from "
                "Paul to the sick? ______"
            ),
            "truth": "Handkerchiefs and aprons that had touched his skin",
            "truth_keywords": [
                "handkerchiefs",
                "aprons",
                "handkerchiefs and aprons",
                "handkerchief",
            ],
            "decoy": "Little clay jars of anointing oil",
            "bot_answers": [
                "Pages torn from his letters",
                "Leather scraps from his tentmaking",
                "Small loaves of blessed bread",
            ],
            "scripture_note": (
                "Acts 19:11-12 - Even handkerchiefs or aprons that had touched Paul's skin were carried to the sick, "
                "and they were healed."
            ),
            "announcer": "Truth or Bluff! What unusual items were carried from Paul to the sick in Ephesus?",
        },
        {
            "id": "tob_ezekiel_scroll",
            "prompt": (
                "In Ezekiel 3, God told the prophet Ezekiel to eat something unusual. What was it, and how did it "
                "taste? ______"
            ),
            "truth": "A scroll, and it tasted as sweet as honey",
            "truth_keywords": [
                "ate a scroll",
                "eat the scroll",
                "eat this scroll",
                "sweet as honey",
                "the scroll",
            ],
            "decoy": "A bitter root, and it tasted like burnt bread",
            "bot_answers": [
                "A loaf of barley bread, and it tasted like dust",
                "A golden fig, and it tasted like rain",
                "A clay tablet, and it tasted salty",
            ],
            "scripture_note": (
                "Ezekiel 3:3 - 'Feed your belly with this scroll.' Then I ate it, and it was in my mouth as sweet as "
                "honey."
            ),
            "announcer": "Truth or Bluff! God gave Ezekiel something strange to eat. What was it? Bluff your family!",
        },
        {
            "id": "tob_temple_tax_coin",
            "prompt": (
                "In Matthew 17, Jesus told Peter how to get money to pay the temple tax for both of them. Where did "
                "the money come from? ______"
            ),
            "truth": "From the mouth of the first fish Peter caught",
            "truth_keywords": [
                "coin in its mouth",
                "fish's mouth",
                "fish with a coin",
                "mouth of a fish",
                "mouth of the fish",
            ],
            "decoy": "From under a stone at the Temple gate",
            "bot_answers": [
                "From Judas's money bag",
                "From selling Peter's old fishing net",
                "From a jar buried in Zacchaeus's yard",
            ],
            "scripture_note": (
                "Matthew 17:27 - 'Take the first fish that comes up, and when you open its mouth you will find a "
                "shekel. Give it for me and for yourself.'"
            ),
            "announcer": "Truth or Bluff! Where did Peter find the money for the temple tax? Write a sneaky bluff!",
        },
        {
            "id": "tob_jethro_advice",
            "prompt": (
                "In Exodus 18, Moses was judging the people from morning to evening. What advice did his "
                "father-in-law Jethro give him? ______"
            ),
            "truth": (
                "Choose capable leaders over groups of thousands, hundreds, fifties and tens to judge the small cases"
            ),
            "truth_keywords": [
                "rulers of thousands",
                "leaders over thousands",
                "small cases",
                "share the load",
                "judge the small",
            ],
            "decoy": "Only hold court on the Sabbath",
            "bot_answers": [
                "Build a taller seat so everyone could see him",
                "Let Aaron do all the talking",
                "Write the rules on stones so nobody needed to ask",
            ],
            "scripture_note": (
                "Exodus 18:17-22 - 'What you are doing is not good... Choose able men as chiefs of thousands, "
                "hundreds, fifties and tens.'"
            ),
            "announcer": "Truth or Bluff! Jethro gave Moses some very practical advice. What was it?",
        },
        {
            "id": "tob_first_plague",
            "prompt": "In Exodus 7, what was the very first of the ten plagues God sent on Egypt? ______",
            "truth": "The water of the Nile turned to blood",
            "truth_keywords": [
                "turned to blood",
                "water into blood",
                "nile to blood",
                "river into blood",
                "water to blood",
            ],
            "decoy": "Swarms of locusts covered the fields",
            "bot_answers": [
                "All the cows turned spotted overnight",
                "Every bread oven went cold",
                "Hail fell only on the palace",
            ],
            "scripture_note": (
                "Exodus 7:20 - Aaron struck the water in the Nile, and all the water turned into blood."
            ),
            "announcer": "Truth or Bluff! What was the very first plague on Egypt? Write a believable fake!",
        },
        {
            "id": "tob_judas_replacement",
            "prompt": (
                "In Acts 1, the apostles chose someone to take Judas's place among the twelve. Who was chosen, and "
                "how? ______"
            ),
            "truth": "Matthias, chosen by casting lots",
            "truth_keywords": [
                "matthias",
                "cast lots",
                "casting lots",
                "drew lots",
            ],
            "decoy": "Barnabas, chosen by a show of hands",
            "bot_answers": [
                "Stephen, chosen by Peter alone",
                "Silas, chosen by the oldest disciple",
                "Timothy, chosen by a vote of the crowd",
            ],
            "scripture_note": (
                "Acts 1:26 - They cast lots for them, and the lot fell on Matthias, and he was numbered with the "
                "eleven apostles."
            ),
            "announcer": "Truth or Bluff! Who took the twelfth spot among the apostles, and how?",
        },
        {
            "id": "tob_david_ark",
            "prompt": (
                "In 2 Samuel 6, when the Ark of the Covenant was brought into Jerusalem, what did King David do in "
                "front of it? ______"
            ),
            "truth": "He danced before the Lord with all his might",
            "truth_keywords": [
                "danced",
                "with all his might",
                "dancing before",
                "danced before",
            ],
            "decoy": "He played a brand-new song on his harp",
            "bot_answers": [
                "He laid his crown on the ground before it",
                "He rode ahead of it on a white mule",
                "He read every psalm he had written aloud",
            ],
            "scripture_note": (
                "2 Samuel 6:14 - David danced before the Lord with all his might, wearing a linen ephod."
            ),
            "announcer": "Truth or Bluff! What did King David do when the Ark came to Jerusalem? Bluff your family!",
        },
        {
            "id": "tob_saul_donkeys",
            "prompt": (
                "In 1 Samuel 9, young Saul met the prophet Samuel before becoming Israel's first king. What was Saul "
                "out searching for? ______"
            ),
            "truth": "His father's lost donkeys",
            "truth_keywords": [
                "lost donkeys",
                "father's donkeys",
                "missing donkeys",
                "the donkeys",
                "looking for donkeys",
            ],
            "decoy": "A runaway lamb for the family sacrifice",
            "bot_answers": [
                "A good spot to dig a new well",
                "His older brother who went hunting",
                "A blacksmith to fix his plow",
            ],
            "scripture_note": (
                "1 Samuel 9:3,20 - The donkeys of Kish, Saul's father, were lost, and Samuel told Saul, 'As for your "
                "donkeys, they have been found.'"
            ),
            "announcer": "Truth or Bluff! What was Saul searching for when he met Samuel? Write a believable fake!",
        },
        {
            "id": "tob_rebekah_sign",
            "prompt": (
                "In Genesis 24, Abraham's servant prayed for a sign to find a wife for Isaac. What did Rebekah do "
                "that showed she was the one? ______"
            ),
            "truth": "She offered to draw water for all of his camels too",
            "truth_keywords": [
                "water for his camels",
                "watered the camels",
                "water the camels",
                "water your camels",
                "camels to drink",
            ],
            "decoy": "She baked him bread before he even asked",
            "bot_answers": [
                "She guessed his name before he said it",
                "She sang the song he had prayed about",
                "She was wearing a blue veil",
            ],
            "scripture_note": (
                "Genesis 24:19 - When she had given him a drink, she said, 'I will draw water for your camels also, "
                "until they have finished drinking.'"
            ),
            "announcer": "Truth or Bluff! What did Rebekah do at the well that answered the servant's prayer?",
        },
        {
            "id": "tob_jesus_age",
            "prompt": "According to Luke 3, about how old was Jesus when He began His public ministry? ______",
            "truth": "About thirty years old",
            "truth_keywords": [
                "about thirty",
                "thirty years",
                "30 years",
                "age thirty",
                "age 30",
            ],
            "decoy": "About twenty-five years old",
            "bot_answers": [
                "Exactly forty years old",
                "About eighteen years old",
                "Thirty-three years old",
            ],
            "scripture_note": "Luke 3:23 - Jesus, when he began his ministry, was about thirty years of age.",
            "announcer": "Truth or Bluff! How old was Jesus when He started His ministry? Write a convincing fake!",
        },
        {
            "id": "tob_lydia_trade",
            "prompt": (
                "In Acts 16, Lydia became the first person in Philippi to believe Paul's message. What did she sell "
                "for a living? ______"
            ),
            "truth": "Purple cloth (she was a seller of purple goods)",
            "truth_keywords": [
                "purple",
                "seller of purple",
                "purple cloth",
                "purple dye",
            ],
            "decoy": "Spices and perfume",
            "bot_answers": [
                "Olive oil and lamps",
                "Sandals and leather belts",
                "Fine pottery from Corinth",
            ],
            "scripture_note": (
                "Acts 16:14 - A woman named Lydia, from the city of Thyatira, a seller of purple goods, was "
                "listening, and the Lord opened her heart."
            ),
            "announcer": "Truth or Bluff! What did Lydia sell for a living? Write a believable bluff!",
        },
        {
            "id": "tob_abigail_gift",
            "prompt": (
                "In 1 Samuel 25, wise Abigail rode out to meet David and his hungry men. What did she bring them? "
                "______"
            ),
            "truth": "A huge feast of bread, prepared sheep, roasted grain, raisin cakes and fig cakes",
            "truth_keywords": [
                "raisins",
                "cakes of figs",
                "fig cakes",
                "two hundred loaves",
                "clusters of raisins",
            ],
            "decoy": "Twelve baskets of fish and barley loaves",
            "bot_answers": [
                "A hundred jars of honey and a new tent",
                "Bags of gold and silver",
                "Fresh oxen and a cart full of melons",
            ],
            "scripture_note": (
                "1 Samuel 25:18 - Abigail took two hundred loaves, five prepared sheep, roasted grain, a hundred "
                "clusters of raisins and two hundred cakes of figs."
            ),
            "announcer": "Truth or Bluff! What did Abigail bring to David and his hungry men?",
        },
        {
            "id": "tob_still_small_voice",
            "prompt": (
                "In 1 Kings 19, after a great wind, an earthquake, and a fire, how did the Lord finally speak to "
                "Elijah? ______"
            ),
            "truth": "In a still small voice, a gentle whisper",
            "truth_keywords": [
                "still small voice",
                "small voice",
                "gentle whisper",
                "a whisper",
                "low whisper",
            ],
            "decoy": "Through a booming thunderclap",
            "bot_answers": [
                "Through a raven that could talk",
                "Through writing in the sand",
                "Through a dream about a ladder",
            ],
            "scripture_note": (
                "1 Kings 19:11-12 - The Lord was not in the wind, the earthquake, or the fire; and after the fire, a "
                "still small voice."
            ),
            "announcer": "Truth or Bluff! How did God speak to Elijah on the mountain? Write a believable fake!",
        },
        {
            "id": "tob_moses_face",
            "prompt": (
                "In Exodus 34, when Moses came down Mount Sinai after talking with God, something about him had "
                "changed. What was it? ______"
            ),
            "truth": "His face shone so brightly he covered it with a veil",
            "truth_keywords": [
                "face shone",
                "face was shining",
                "wore a veil",
                "put a veil",
                "face glowed",
            ],
            "decoy": "His hair had turned completely white",
            "bot_answers": [
                "His voice had become much deeper",
                "His staff had sprouted almond flowers",
                "He had grown two inches taller",
            ],
            "scripture_note": (
                "Exodus 34:29,33 - The skin of Moses' face shone because he had been talking with God, so he put a "
                "veil over his face."
            ),
            "announcer": "Truth or Bluff! What had changed about Moses when he came down the mountain?",
        },
        {
            "id": "tob_enoch_walk",
            "prompt": (
                "Genesis 5 tells us something unusual about Enoch, the father of Methuselah. What happened at the end "
                "of his life on earth? ______"
            ),
            "truth": "He walked with God, and God took him, so he never died",
            "truth_keywords": [
                "walked with god",
                "god took him",
                "never died",
                "did not die",
                "didn't die",
            ],
            "decoy": "He planted a forest that still stands today",
            "bot_answers": [
                "He sailed off and was never seen again",
                "He fell asleep for a hundred years",
                "He moved to the land of Nod",
            ],
            "scripture_note": (
                "Genesis 5:24 & Hebrews 11:5 - Enoch walked with God, and he was not, for God took him, so that he "
                "did not see death."
            ),
            "announcer": "Truth or Bluff! Something amazing happened to Enoch. What was it? Bluff your family!",
        },
        {
            "id": "tob_shortest_verse",
            "prompt": "In most English Bibles, the shortest verse is just two words long. What does it say? ______",
            "truth": "Jesus wept (John 11:35)",
            "truth_keywords": [
                "jesus wept",
                "john 11 35",
                "john 1135",
            ],
            "decoy": "God is (Psalm 46:1)",
            "bot_answers": [
                "Rejoice always",
                "Pray continually",
                "Love endures",
            ],
            "scripture_note": (
                "John 11:35 - 'Jesus wept.' Standing with grieving friends at Lazarus's tomb, Jesus showed His tender "
                "heart."
            ),
            "announcer": (
                "Truth or Bluff! What is the shortest verse in most English Bibles? Write a believable fake!"
            ),
        },
        {
            "id": "tob_matthew_job",
            "prompt": (
                "In Matthew 9, Jesus saw a man sitting at his tax booth and said, 'Follow me.' Which disciple was it? "
                "______"
            ),
            "truth": "Matthew (also called Levi)",
            "truth_keywords": [
                "matthew",
                "levi",
                "called levi",
                "named matthew",
            ],
            "decoy": "Thaddeus, the tax collector of Capernaum",
            "bot_answers": [
                "Bartholomew",
                "Philip of Bethsaida",
                "Simon the Zealot",
            ],
            "scripture_note": (
                "Matthew 9:9 - Jesus saw a man called Matthew sitting at the tax booth and said, 'Follow me.' And he "
                "rose and followed him."
            ),
            "announcer": "Truth or Bluff! Which disciple left his tax booth to follow Jesus?",
        },
        {
            "id": "tob_nebuchadnezzar_field",
            "prompt": (
                "In Daniel 4, proud King Nebuchadnezzar was driven away to live in the fields for a time. What did he "
                "do there? ______"
            ),
            "truth": "He ate grass like an ox, and his hair grew like eagles' feathers",
            "truth_keywords": [
                "ate grass",
                "eat grass",
                "grass like",
                "like an ox",
                "eagles feathers",
            ],
            "decoy": "He built a tiny palace out of mud bricks",
            "bot_answers": [
                "He herded goats for a farmer",
                "He wrote poems carved on tree bark",
                "He slept in a cave with a lion",
            ],
            "scripture_note": (
                "Daniel 4:33 - He was driven from among men and ate grass like an ox, till his hair grew as long as "
                "eagles' feathers."
            ),
            "announcer": (
                "Truth or Bluff! What did King Nebuchadnezzar do out in the fields? Write a believable fake!"
            ),
        },
    ],
    "survey_says": [
        {
            "id": "ss_church_snack",
            "prompt": "WE ASKED 100 KIDS: What is the best snack ever served after church?",
            "decoy": "Donut holes, because you can eat twelve and call it one donut",
            "bot_answers": [
                "Goldfish crackers by the fistful",
                "The cookies hidden behind the coffee pot",
                "Whatever the youth pastor isn't guarding",
            ],
            "scripture_note": "John 21:12 - Jesus said to them, 'Come and have breakfast.'",
            "announcer": (
                "Double-Point Lightning Finale! Survey Says: What is the best snack ever served after church?"
            ),
        },
        {
            "id": "ss_bible_pet",
            "prompt": "WE ASKED 100 FAMILIES: Which animal from the Bible would make the best family pet?",
            "decoy": "Balaam's donkey, because at least it would tell you when it's hungry",
            "bot_answers": [
                "One of Elijah's ravens: it brings breakfast!",
                "The big fish, if you have a REALLY big bathtub",
                "A lamb, as long as it doesn't wander off",
            ],
            "scripture_note": (
                "Genesis 1:25 - God made every creature that creeps on the ground, and God saw that it was good."
            ),
            "announcer": "Double Points on the line! Survey Says: Which Bible animal would make the best family pet?",
        },
        {
            "id": "ss_after_vbs",
            "prompt": "WE ASKED 100 PARENTS: What does your kid do for a whole week after Vacation Bible School?",
            "decoy": "Sings the theme song at 6 AM with full hand motions",
            "bot_answers": [
                "Asks if every day can be Bible School day",
                "Tries to teach the dog the memory verse",
                "Wears the T-shirt until it can stand up on its own",
            ],
            "scripture_note": "Psalm 96:1 - Oh sing to the Lord a new song; sing to the Lord, all the earth!",
            "announcer": (
                "Double-Point Finale! Survey Says: What does your kid do all week after Vacation Bible School?"
            ),
        },
        {
            "id": "ss_road_trip",
            "prompt": "WE ASKED 100 DADS: What is the most-asked question on a family road trip?",
            "decoy": "Are we there yet? (Asked forty times, like the wilderness)",
            "bot_answers": [
                "Can we stop at the next bathroom?",
                "Who touched my snacks?",
                "Can I pick the music now?",
            ],
            "scripture_note": (
                "Psalm 121:8 - The Lord will keep your going out and your coming in from this time forth and "
                "forevermore."
            ),
            "announcer": "Double Points! Survey Says: What is the most-asked question on a family road trip?",
        },
        {
            "id": "ss_thanksgiving_guest",
            "prompt": (
                "WE ASKED 100 CHURCHGOERS: Which Bible character would you most like to invite to Thanksgiving "
                "dinner?"
            ),
            "decoy": "Boaz, because he always leaves a little extra on purpose",
            "bot_answers": [
                "Peter, for the fish stories",
                "Solomon, to settle who gets the last slice of pie",
                "Martha, because she'd bring the whole kitchen",
            ],
            "scripture_note": (
                "Ruth 2:16 - Boaz told his reapers, 'Pull out some from the bundles for her and leave it for her to "
                "glean.'"
            ),
            "announcer": (
                "Lightning Finale for Double Points! Survey Says: Which Bible character would you invite to "
                "Thanksgiving dinner?"
            ),
        },
        {
            "id": "ss_long_sermon",
            "prompt": "WE ASKED 100 PASTORS: What is the surest sign the sermon is running a little long?",
            "decoy": "The kids have built a fort out of offering envelopes",
            "bot_answers": [
                "The organist starts tiptoeing toward the bench",
                "Even the stained-glass saints look hungry",
                "Someone's watch alarm goes off for lunch",
            ],
            "scripture_note": "Acts 20:7 - Paul talked with them and prolonged his speech until midnight.",
            "announcer": "Double-Point Finale! Survey Says: What is the surest sign the sermon is running long?",
        },
        {
            "id": "ss_choir_hardest",
            "prompt": "WE ASKED 100 CHOIR MEMBERS: What is the hardest part of singing in the church choir?",
            "decoy": "Finding your note after the pianist changes key without warning",
            "bot_answers": [
                "Not giggling when the soloist's voice cracks",
                "Standing up for all seven verses",
                "Remembering whether this is a 'repeat the chorus' week",
            ],
            "scripture_note": (
                "Colossians 3:16 - Singing psalms and hymns and spiritual songs, with thankfulness in your hearts to "
                "God."
            ),
            "announcer": (
                "Double Points on the line! Survey Says: What is the hardest part of singing in the church choir?"
            ),
        },
        {
            "id": "ss_hero_ability",
            "prompt": (
                "WE ASKED 100 KIDS: If you could have one Bible hero's ability for a day, which would you pick?"
            ),
            "decoy": "Samson's strength, to carry all the groceries in one trip",
            "bot_answers": [
                "Walking on water like Peter, for a little while anyway",
                "Daniel's lion-whispering, for the neighbor's dog",
                "Solomon's wisdom, for the math test",
            ],
            "scripture_note": "Philippians 4:13 - I can do all things through him who strengthens me.",
            "announcer": (
                "Lightning Finale for Double Points! Survey Says: Which Bible hero's ability would you pick for a "
                "day?"
            ),
        },
        {
            "id": "ss_bedtime_prayers",
            "prompt": "WE ASKED 100 PARENTS: What is the funniest thing kids pray for at bedtime?",
            "decoy": "That their brother turns into a nicer brother by morning",
            "bot_answers": [
                "That the goldfish has a good day tomorrow",
                "That broccoli gets un-invented",
                "Every person in the family, including the hamster's cousins",
            ],
            "scripture_note": (
                "Philippians 4:6 - In everything by prayer and supplication with thanksgiving let your requests be "
                "made known to God."
            ),
            "announcer": "Double-Point Finale! Survey Says: What is the funniest thing kids pray for at bedtime?",
        },
        {
            "id": "ss_potluck_staple",
            "prompt": "WE ASKED 100 CHURCH COOKS: What dish ALWAYS shows up at the church potluck?",
            "decoy": "Seventeen different kinds of potato salad",
            "bot_answers": [
                "Green bean casserole with crispy onions",
                "Mystery meatballs in a slow cooker",
                "Deviled eggs that vanish in four minutes",
            ],
            "scripture_note": (
                "1 Corinthians 10:31 - Whether you eat or drink, or whatever you do, do all to the glory of God."
            ),
            "announcer": "Double Points! Survey Says: What dish ALWAYS shows up at the church potluck?",
        },
        {
            "id": "ss_lost_and_found",
            "prompt": "WE ASKED 100 CHURCH CUSTODIANS: What is the most common thing in the church lost and found?",
            "decoy": "One small shoe. Never the pair. Always just one.",
            "bot_answers": [
                "Reading glasses sitting on top of a Bible",
                "A sweater nobody admits owning",
                "A half-eaten bag of Goldfish crackers",
            ],
            "scripture_note": "Luke 15:9 - 'Rejoice with me, for I have found the coin that I had lost.'",
            "announcer": (
                "Lightning Finale for Double Points! Survey Says: What is the most common thing in the church lost "
                "and found?"
            ),
        },
        {
            "id": "ss_bible_vacation",
            "prompt": "WE ASKED 100 TRAVELERS: Which place from the Bible would make the best family vacation spot?",
            "decoy": "The Garden of Eden, but everyone signs a no-fruit-picking waiver",
            "bot_answers": [
                "The Sea of Galilee for boating, life jackets required",
                "Mount Ararat, for the view and the boat museum",
                "Bethlehem at Christmas, for the star",
            ],
            "scripture_note": (
                "Genesis 2:8-9 - God planted a garden in Eden and made every tree that is pleasant to the sight and "
                "good for food."
            ),
            "announcer": "Double-Point Finale! Survey Says: Which Bible place would make the best family vacation?",
        },
        {
            "id": "ss_sunday_lunch",
            "prompt": "WE ASKED 100 FAMILIES: Where does your family go for lunch after church?",
            "decoy": "Wherever Dad says, which is the same place every single week",
            "bot_answers": [
                "Grandma's house, for the pot roast",
                "The restaurant where we see half the congregation",
                "Home, for leftovers and a nap",
            ],
            "scripture_note": "Psalm 34:8 - Oh, taste and see that the Lord is good!",
            "announcer": "Double Points on the line! Survey Says: Where does your family go for lunch after church?",
        },
        {
            "id": "ss_youth_game",
            "prompt": "WE ASKED 100 YOUTH PASTORS: What is the most popular youth group game of all time?",
            "decoy": "Anything involving a giant parachute and lots of screaming",
            "bot_answers": [
                "Dodgeball, with very good sportsmanship",
                "Capture the flag in the dark parking lot",
                "Four square at the church gym",
            ],
            "scripture_note": (
                "1 Timothy 4:12 - Let no one despise you for your youth, but set the believers an example."
            ),
            "announcer": (
                "Lightning Finale for Double Points! Survey Says: What is the most popular youth group game?"
            ),
        },
        {
            "id": "ss_pageant_mishap",
            "prompt": "WE ASKED 100 PARENTS: What goes wrong most often in the Christmas pageant?",
            "decoy": "A shepherd waves to Grandma right in the middle of the angel's big line",
            "bot_answers": [
                "Baby Jesus is a doll that keeps falling out of the manger",
                "A sheep costume loses its tail mid-scene",
                "The star of Bethlehem is held by a kid who gets bored",
            ],
            "scripture_note": (
                "Luke 2:10-11 - 'Behold, I bring you good news of great joy that will be for all the people.'"
            ),
            "announcer": "Double-Point Finale! Survey Says: What goes wrong most often in the Christmas pageant?",
        },
        {
            "id": "ss_easter_morning",
            "prompt": "WE ASKED 100 FAMILIES: What is the best part of Easter Sunday morning?",
            "decoy": "Seeing the church so full the ushers bring out folding chairs",
            "bot_answers": [
                "Singing 'He is risen!' really loud",
                "Sunrise service and then pancakes",
                "The family photo where nobody blinked",
            ],
            "scripture_note": "Matthew 28:6 - 'He is not here, for he has risen, as he said.'",
            "announcer": "Double Points! Survey Says: What is the best part of Easter Sunday morning?",
        },
        {
            "id": "ss_dog_name",
            "prompt": "WE ASKED 100 PET OWNERS: What is the best Bible name for a dog?",
            "decoy": "Caleb, because he's loyal and still full of energy at eighty-five",
            "bot_answers": [
                "Samson, for the big fluffy ones",
                "Goliath, for the tiny chihuahua",
                "Moses, because he's always leading you somewhere",
            ],
            "scripture_note": (
                "Joshua 14:10-11 - Caleb said, 'I am this day eighty-five years old. I am still as strong today as I "
                "was in the day Moses sent me.'"
            ),
            "announcer": "Lightning Finale for Double Points! Survey Says: What is the best Bible name for a dog?",
        },
        {
            "id": "ss_mom_verse",
            "prompt": "WE ASKED 100 MOMS: Which Bible verse do moms quote most often at home?",
            "decoy": "Children, obey your parents, for this is right (and also because I said so)",
            "bot_answers": [
                "Be kind to one another, especially in the back seat",
                "This is the day the Lord has made, now get out of bed",
                "Let all things be done decently and in order, starting with your room",
            ],
            "scripture_note": "Ephesians 6:1 - Children, obey your parents in the Lord, for this is right.",
            "announcer": "Double-Point Finale! Survey Says: Which Bible verse do moms quote most at home?",
        },
        {
            "id": "ss_late_excuse",
            "prompt": "WE ASKED 100 CHURCH GREETERS: What is the most common excuse for being late to church?",
            "decoy": "We were praying for patience and it took a while",
            "bot_answers": [
                "The dog ran off with the car keys",
                "Someone needed a new shirt after the pancake incident",
                "We had to go back for the Bible we left on the table",
            ],
            "scripture_note": "Proverbs 16:9 - The heart of man plans his way, but the Lord establishes his steps.",
            "announcer": (
                "Double Points on the line! Survey Says: What is the most common excuse for being late to church?"
            ),
        },
        {
            "id": "ss_bible_food",
            "prompt": "WE ASKED 100 KIDS: Which food from the Bible would you most like to taste?",
            "decoy": "Manna, to find out if it really tastes like honey wafers",
            "bot_answers": [
                "The fish Jesus cooked on the beach",
                "Esau's red stew, but not at the birthright price",
                "The bread from the boy's lunch that fed five thousand",
            ],
            "scripture_note": "Exodus 16:31 - The taste of manna was like wafers made with honey.",
            "announcer": (
                "Lightning Finale for Double Points! Survey Says: Which food from the Bible would you most like to "
                "taste?"
            ),
        },
        {
            "id": "ss_church_grandma",
            "prompt": "WE ASKED 100 GRANDKIDS: What is the best thing about sitting with Grandma at church?",
            "decoy": "The mint candy that magically appears from her purse during the sermon",
            "bot_answers": [
                "She knows every hymn without opening the book",
                "She introduces you to every single person",
                "Lunch afterward is always her treat",
            ],
            "scripture_note": (
                "2 Timothy 1:5 - I am reminded of your sincere faith, a faith that dwelt first in your grandmother "
                "Lois."
            ),
            "announcer": (
                "Double-Point Finale! Survey Says: What is the best thing about sitting with Grandma at church?"
            ),
        },
        {
            "id": "ss_bible_job",
            "prompt": "WE ASKED 100 BIBLE READERS: Which job from Bible times would you most like to have had?",
            "decoy": "Royal cupbearer at the palace, for the snacks",
            "bot_answers": [
                "Fisherman on the Sea of Galilee",
                "Shepherd, because the sheep don't talk back",
                "Carpenter's apprentice in Nazareth",
            ],
            "scripture_note": "Colossians 3:23 - Whatever you do, work heartily, as for the Lord and not for men.",
            "announcer": "Double Points! Survey Says: Which job from Bible times would you most like to have had?",
        },
        {
            "id": "ss_church_camp",
            "prompt": "WE ASKED 100 CAMPERS: What do you remember most about church camp?",
            "decoy": "The bug spray smell that lasted until Christmas",
            "bot_answers": [
                "Campfire songs and burnt marshmallows",
                "The cabin that pranked the counselors",
                "Canoe races on the lake",
            ],
            "scripture_note": "Mark 6:31 - 'Come away by yourselves to a desolate place and rest a while.'",
            "announcer": (
                "Lightning Finale for Double Points! Survey Says: What do you remember most about church camp?"
            ),
        },
        {
            "id": "ss_game_night",
            "prompt": "WE ASKED 100 FAMILIES: What causes the most arguments at family game night?",
            "decoy": "Grandpa making up a brand-new rule halfway through",
            "bot_answers": [
                "Someone 'accidentally' peeking at the cards",
                "Who gets to be the banker",
                "The dog walking right across the board",
            ],
            "scripture_note": "Romans 12:18 - If possible, so far as it depends on you, live peaceably with all.",
            "announcer": "Double-Point Finale! Survey Says: What causes the most arguments at family game night?",
        },
        {
            "id": "ss_best_sidekick",
            "prompt": "WE ASKED 100 BIBLE READERS: Who is the greatest sidekick in the whole Bible?",
            "decoy": "Aaron, Moses' official spokesman and backup staff carrier",
            "bot_answers": [
                "Jonathan, David's loyal best friend",
                "Elisha, who asked for a double portion",
                "Silas, the best duet partner in a jail cell",
            ],
            "scripture_note": "Proverbs 17:17 - A friend loves at all times, and a brother is born for adversity.",
            "announcer": "Double Points on the line! Survey Says: Who is the greatest sidekick in the whole Bible?",
        },
    ],
    "parable_remix": [
        {
            "id": "pr_lost_sheep_mall",
            "prompt": "Retell the Parable of the Lost Sheep as if it happened at a giant shopping mall.",
            "decoy": (
                "Mall security leaves ninety-nine shoppers at the food court to find one kid in the massage chairs"
            ),
            "bot_answers": [
                "Dad leaves the whole family at the pretzel stand to find Timmy in the toy store",
                "Security radios: 'One sheep spotted at the fountain, repeat, one sheep'",
                "Ninety-nine shoppers wait while Mom checks every single shoe store",
            ],
            "scripture_note": (
                "Luke 15:4-7 - There is more joy in heaven over one sinner who repents than over ninety-nine who need "
                "no repentance."
            ),
            "announcer": "Parable Remix! The Lost Sheep goes to the shopping mall! How does the story go now?",
        },
        {
            "id": "pr_sower_garden_show",
            "prompt": "Retitle the Parable of the Sower as an episode of a backyard gardening TV show.",
            "decoy": "Extreme Makeover: Soil Edition",
            "bot_answers": [
                "Seeds Gone Wild: The Thorny Truth",
                "Why Your Tomatoes Need Good Ground (and So Do You)",
                "Bird-Proofing Your Heart in Four Easy Steps",
            ],
            "scripture_note": (
                "Matthew 13:23 - The seed on good soil is the one who hears the word and understands it, and bears "
                "fruit."
            ),
            "announcer": (
                "Parable Remix! The Parable of the Sower is now a gardening show! What's the episode called?"
            ),
        },
        {
            "id": "pr_samaritan_flat_tire",
            "prompt": "Retell the Good Samaritan on a modern highway with a flat tire. Who finally stops to help?",
            "decoy": "A grandma in a minivan with jumper cables, snacks, and a spare tire",
            "bot_answers": [
                "A fan of the rival football team, wearing the other team's jersey",
                "A tow-truck driver who pays for the motel himself",
                "A kid on a bike who calls their mom, who calls everybody",
            ],
            "scripture_note": (
                "Luke 10:36-37 - 'Which of these proved to be a neighbor?' 'The one who showed him mercy.' 'You go, "
                "and do likewise.'"
            ),
            "announcer": "Parable Remix! The Good Samaritan meets a flat tire on the highway! Who stops to help?",
        },
        {
            "id": "pr_prodigal_arcade",
            "prompt": (
                "Retell the Prodigal Son as a kid who spent all his allowance at the arcade. What does Dad do when he "
                "comes home?"
            ),
            "decoy": "Throws a pizza party and gives him the good controller",
            "bot_answers": [
                "Runs to the end of the driveway in his slippers",
                "Hugs him before he finishes his apology speech",
                "Orders the biggest cake at the grocery store",
            ],
            "scripture_note": (
                "Luke 15:20 - While he was still a long way off, his father saw him and felt compassion, and ran and "
                "embraced him."
            ),
            "announcer": "Parable Remix! The Prodigal Son blew his allowance at the arcade! What does Dad do?",
        },
        {
            "id": "pr_bridesmaids_chargers",
            "prompt": "Retell the Parable of the Ten Bridesmaids and their lamps, but with phones and chargers.",
            "decoy": "Five packed chargers; five were stuck at one percent when the groom arrived",
            "bot_answers": [
                "Five brought power banks, five asked 'Can I borrow yours?'",
                "The wedding started while half of them hunted for an outlet",
                "Moral: keep your battery full and your heart ready",
            ],
            "scripture_note": (
                "Matthew 25:4,13 - The wise took oil in flasks with their lamps. Watch therefore, for you know "
                "neither the day nor the hour."
            ),
            "announcer": "Parable Remix! The Ten Bridesmaids now have phones and chargers! How does it go?",
        },
        {
            "id": "pr_mustard_seed_story",
            "prompt": "Retitle the Parable of the Mustard Seed as a modern small-start success story.",
            "decoy": "From Tiny Seed to Bird Hotel: A Backyard Startup Story",
            "bot_answers": [
                "Small Beginnings, Giant Branches",
                "The Smallest Seed That Became the Neighborhood's Favorite Tree",
                "Don't Judge a Seed by Its Size",
            ],
            "scripture_note": (
                "Matthew 13:31-32 - The mustard seed is the smallest of all seeds, but it grows into a tree so that "
                "birds make nests in its branches."
            ),
            "announcer": "Parable Remix! The Mustard Seed gets a modern success-story title! What is it?",
        },
        {
            "id": "pr_lost_remote",
            "prompt": "Retell the Parable of the Lost Coin, but it's the family's lost TV remote.",
            "decoy": "Mom sweeps the whole house, finds it in the couch, and throws a neighborhood party",
            "bot_answers": [
                "Everyone flips every cushion; Dad finds it in the fridge",
                "When it's found, the family cheers louder than at the big game",
                "She turns on every lamp and finally checks the dog's bed",
            ],
            "scripture_note": (
                "Luke 15:8-9 - She lights a lamp, sweeps the house, and seeks diligently until she finds it, then "
                "calls her friends to rejoice."
            ),
            "announcer": "Parable Remix! The Lost Coin is now the lost TV remote! How does the story go?",
        },
        {
            "id": "pr_talents_lemonade",
            "prompt": "Retell the Parable of the Talents as three kids given money to start lemonade stands.",
            "decoy": "One sells lemonade, one sells cookies too, and one buries his dollar in the sandbox",
            "bot_answers": [
                "Kid three kept the money under his pillow 'for safety'",
                "Two doubled their money; one was too scared to squeeze a lemon",
                "The boss says 'Well done!' to the kids who actually opened shop",
            ],
            "scripture_note": (
                "Matthew 25:21 - 'Well done, good and faithful servant. You have been faithful over a little; I will "
                "set you over much.'"
            ),
            "announcer": "Parable Remix! The Parable of the Talents goes to the lemonade stand! How does it end?",
        },
        {
            "id": "pr_vineyard_ice_cream",
            "prompt": (
                "Retell the Workers in the Vineyard as a family yard-work day where the late helpers get the same ice "
                "cream."
            ),
            "decoy": "Little brother helps for five minutes and gets the same sundae; the older kids are speechless",
            "bot_answers": [
                "Everyone gets a double scoop, no matter when they started raking",
                "Dad says, 'It's my ice cream, I can be generous with it'",
                "The last helper carried one leaf and still got sprinkles",
            ],
            "scripture_note": (
                "Matthew 20:14-15 - 'I choose to give to this last worker as I give to you. Do you begrudge my "
                "generosity?'"
            ),
            "announcer": (
                "Parable Remix! The Workers in the Vineyard are now raking leaves for ice cream! What happens?"
            ),
        },
        {
            "id": "pr_two_sons_rooms",
            "prompt": "Retell the Parable of the Two Sons as two kids asked to clean their rooms.",
            "decoy": "One says 'No way!' then cleans it; the other says 'Sure!' and plays video games",
            "bot_answers": [
                "Kid one grumbles, then organizes the closet by color",
                "Kid two says 'On it!' and is never seen again",
                "Mom asks who did what she asked, and everyone points at the grumbler",
            ],
            "scripture_note": (
                "Matthew 21:28-31 - One son said 'I will not' but afterward went; the other said 'I go, sir' but did "
                "not go."
            ),
            "announcer": "Parable Remix! The Two Sons have been told to clean their rooms! How does it go?",
        },
        {
            "id": "pr_pearl_collectors",
            "prompt": "Retitle the Parable of the Pearl of Great Price as a modern collectors' TV show.",
            "decoy": "The Man Who Sold Everything for One Pearl, Tonight at Eight",
            "bot_answers": [
                "Treasure Hunters: The Priceless Edition",
                "Extreme Bargain Hunters: Worth Every Penny",
                "Collector Sells Comic Books, Buys Kingdom",
            ],
            "scripture_note": (
                "Matthew 13:45-46 - A merchant found one pearl of great value, went and sold all that he had, and "
                "bought it."
            ),
            "announcer": "Parable Remix! The Pearl of Great Price is a TV show now! What's it called?",
        },
        {
            "id": "pr_banquet_birthday",
            "prompt": (
                "Retell the Parable of the Great Banquet as a birthday party. What excuses do the invited guests "
                "give?"
            ),
            "decoy": "Sorry, I just bought a new fantasy football team and need to go check on it",
            "bot_answers": [
                "Can't come, my goldfish has a dentist appointment",
                "I just got new shoes and have to break them in",
                "I'm busy reorganizing my sock drawer that night",
            ],
            "scripture_note": (
                "Luke 14:18-21 - They all began to make excuses, so the master said, 'Go out quickly and bring in the "
                "poor, the crippled, the blind and the lame.'"
            ),
            "announcer": (
                "Parable Remix! The Great Banquet is now a birthday party! What excuses are the guests giving?"
            ),
        },
        {
            "id": "pr_spelling_bee_giant",
            "prompt": "Retell David and Goliath as a school spelling bee.",
            "decoy": "A tiny first-grader faces the eighth-grade champion and spells 'Philistine' perfectly",
            "bot_answers": [
                "The giant word was 'onomatopoeia'; the small kid had faith and phonics",
                "Everyone said to borrow her brother's big glasses; she wore her own",
                "Five smooth flashcards were all she needed",
            ],
            "scripture_note": (
                "1 Samuel 17:45 - 'You come to me with a sword and a spear, but I come to you in the name of the Lord "
                "of hosts.'"
            ),
            "announcer": "Parable Remix! David and Goliath face off at the school spelling bee! How does it go?",
        },
        {
            "id": "pr_ruth_moving_day",
            "prompt": "Retell the story of Ruth and Naomi as a family moving to a brand new town.",
            "decoy": "Ruth refuses to let her mother-in-law move alone and packs the moving truck herself",
            "bot_answers": [
                "'Where you go, I go, and I'm driving'",
                "New town, new neighbors, and a kind farmer with extra groceries",
                "Ruth gets a job at the farmers market and meets the owner",
            ],
            "scripture_note": (
                "Ruth 1:16 - 'Where you go I will go, and where you lodge I will lodge. Your people shall be my "
                "people, and your God my God.'"
            ),
            "announcer": "Parable Remix! Ruth and Naomi are moving to a new town! How does the story go today?",
        },
        {
            "id": "pr_joseph_intern",
            "prompt": "Retell the story of Joseph as an office intern who ends up running the company.",
            "decoy": "Intern with a fancy jacket ends up as boss and feeds the whole town",
            "bot_answers": [
                "His brothers send him to a rival firm; he becomes CEO anyway",
                "He explains the boss's weird dreams and gets promoted to manager",
                "Years later, his brothers apply for jobs at his company",
            ],
            "scripture_note": (
                "Genesis 50:20 - 'You meant evil against me, but God meant it for good, to bring it about that many "
                "people should be kept alive.'"
            ),
            "announcer": "Parable Remix! Joseph is an office intern now! How does his story go?",
        },
        {
            "id": "pr_unforgiving_library",
            "prompt": "Retell the Parable of the Unforgiving Servant with library late fees.",
            "decoy": (
                "The librarian erases his ten-thousand-dollar fine; he then charges his sister a nickel for a pencil"
            ),
            "bot_answers": [
                "His whole fine was erased, then he billed his friend for borrowing a crayon",
                "Forgiven big, forgives small: that's the whole point, Kevin",
                "The librarian hears about it and is NOT impressed",
            ],
            "scripture_note": (
                "Matthew 18:33 - 'Should not you have had mercy on your fellow servant, as I had mercy on you?'"
            ),
            "announcer": "Parable Remix! The Unforgiving Servant owes the library! How does it go?",
        },
        {
            "id": "pr_rich_fool_storage",
            "prompt": "Retitle the Parable of the Rich Fool as a reality TV show about building bigger barns.",
            "decoy": "Bigger Barns: The Storage Unit Wars",
            "bot_answers": [
                "Extreme Collectors: Grain Silo Special",
                "Tiny House? No Thanks, I Need Three Barns",
                "My Stuff, My Barns, My Big Oops",
            ],
            "scripture_note": (
                "Luke 12:15 - 'Be on your guard against all covetousness, for one's life does not consist in the "
                "abundance of his possessions.'"
            ),
            "announcer": "Parable Remix! The Rich Fool gets his own reality show! What's it called?",
        },
        {
            "id": "pr_widow_puppy",
            "prompt": "Retell the Parable of the Persistent Widow as a kid asking for a puppy.",
            "decoy": "She asks every single day until Dad says yes just to get some sleep",
            "bot_answers": [
                "Puppy drawings taped to every door in the house",
                "Sticky notes on the coffee maker: 'Puppy? Please?'",
                "Day 47: Judge Dad finally rules in her favor",
            ],
            "scripture_note": (
                "Luke 18:1,5 - Jesus told them a parable to show that they ought always to pray and not lose heart."
            ),
            "announcer": "Parable Remix! The Persistent Widow wants a puppy! How does the story go?",
        },
        {
            "id": "pr_lamp_camping",
            "prompt": "Retell 'Don't hide your lamp under a basket' as a family camping trip.",
            "decoy": "One kid hides the only flashlight in the tent, and the whole family trips over the cooler",
            "bot_answers": [
                "Grandpa sets the lantern on a stump and finally everyone can find the s'mores",
                "Shine it bright, because Dad's trying to read the map",
                "Lantern under the sleeping bag? That's how you meet a raccoon",
            ],
            "scripture_note": (
                "Matthew 5:15-16 - No one lights a lamp and puts it under a basket, but on a stand. Let your light "
                "shine before others."
            ),
            "announcer": "Parable Remix! Don't hide your light, especially on a camping trip! How does it go?",
        },
        {
            "id": "pr_treasure_yard_sale",
            "prompt": (
                "Retell the Parable of the Hidden Treasure with a kid who finds something amazing at a yard sale."
            ),
            "decoy": "He sells his whole trading-card collection to buy a box with a treasure map inside",
            "bot_answers": [
                "He buys the whole yard sale just to get the one thing",
                "Joyfully trades everything he owns for one dusty old chest",
                "Mom asks why there are thirty lamps in the garage now",
            ],
            "scripture_note": (
                "Matthew 13:44 - A man found treasure hidden in a field, and in his joy he went and sold all that he "
                "had and bought that field."
            ),
            "announcer": "Parable Remix! The Hidden Treasure turns up at a yard sale! How does it go?",
        },
    ],
    "bible_bios": [
        {
            "id": "bb_noah_bio",
            "prompt": "Write Noah's social-media bio.",
            "decoy": "Boat builder. Zookeeper. Weather enthusiast. 600 years young.",
            "bot_answers": [
                "Big boats, bigger faith. DM me for rainbow photos",
                "Two of everything. One of me. Forty days of rain",
                "Not a sailor, just really good at following instructions",
            ],
            "scripture_note": (
                "Genesis 7:6 - Noah was six hundred years old when the flood of waters came upon the earth."
            ),
            "announcer": "Bible Bios! Noah just joined social media! Write his profile bio!",
        },
        {
            "id": "bb_moses_successor",
            "prompt": "Write the job listing for Moses' replacement as leader of Israel.",
            "decoy": "Must handle a million complaints a day and part seas on request. Bring your own staff.",
            "bot_answers": [
                "Requirements: forty years of wilderness experience and mountain hiking",
                "Ideal candidate is humble, patient, and okay with manna",
                "Remote work not available; this position is in the desert",
            ],
            "scripture_note": (
                "Joshua 1:9 - 'Be strong and courageous. Do not be frightened, for the Lord your God is with you "
                "wherever you go.'"
            ),
            "announcer": "Bible Bios! Israel is hiring a new leader to follow Moses! Write the job listing!",
        },
        {
            "id": "bb_goliath_gym",
            "prompt": "Write Goliath's gym-membership profile.",
            "decoy": "Six cubits tall. Loves heavy lifting. Do NOT bring a slingshot to the gym.",
            "bot_answers": [
                "Personal record: a 125-pound coat of armor",
                "Looking for a worthy opponent. Shepherds need not apply",
                "Favorite exercise: shouting across valleys",
            ],
            "scripture_note": (
                "1 Samuel 17:4-5 - Goliath was six cubits and a span tall, and his coat of mail weighed five thousand "
                "shekels of bronze."
            ),
            "announcer": "Bible Bios! Goliath signed up at the gym! Write his member profile!",
        },
        {
            "id": "bb_martha_bio",
            "prompt": "Write Martha of Bethany's social-media bio.",
            "decoy": "Host. Chef. Professional worrier. Ask me about my thirteen-course dinner.",
            "bot_answers": [
                "Busy, busy, busy. My sister? Not so much",
                "Hospitality is my love language",
                "I will feed you whether you're hungry or not",
            ],
            "scripture_note": (
                "Luke 10:41-42 - 'Martha, Martha, you are anxious and troubled about many things, but one thing is "
                "necessary.'"
            ),
            "announcer": "Bible Bios! Martha needs a profile bio! What does it say?",
        },
        {
            "id": "bb_john_baptist_helper",
            "prompt": "Write the job listing for John the Baptist's assistant.",
            "decoy": "Must love camel hair, river work and locust lunches. Waders optional.",
            "bot_answers": [
                "Duties include calling 'Prepare the way!' across the desert",
                "Benefits include unlimited wild honey",
                "Must be comfortable standing waist-deep in the Jordan all day",
            ],
            "scripture_note": (
                "Matthew 3:4-6 - John wore camel's hair, ate locusts and wild honey, and baptized people in the river "
                "Jordan."
            ),
            "announcer": "Bible Bios! John the Baptist is hiring an assistant! Write the job listing!",
        },
        {
            "id": "bb_solomon_headline",
            "prompt": "Write King Solomon's professional profile headline.",
            "decoy": "Wisest Man Alive | Temple Builder | 3,000 Proverbs and 1,005 Songs",
            "bot_answers": [
                "Chief Wisdom Officer of the Kingdom of Israel",
                "Consultant in Wisdom, Gold, and Very Large Building Projects",
                "Ask me anything. Seriously. Anything.",
            ],
            "scripture_note": (
                "1 Kings 4:29,32 - God gave Solomon wisdom and understanding; he spoke 3,000 proverbs, and his songs "
                "were 1,005."
            ),
            "announcer": "Bible Bios! King Solomon is updating his professional profile! Write his headline!",
        },
        {
            "id": "bb_young_david_resume",
            "prompt": "Write young David's résumé headline from before he became king.",
            "decoy": "Shepherd | Harpist | Giant-Defeater | Youngest of Eight",
            "bot_answers": [
                "Experienced in lion and bear removal",
                "Part-time musician, full-time sheep guy",
                "Delivers cheese to the front lines",
            ],
            "scripture_note": (
                "1 Samuel 17:34-35 - 'When there came a lion, or a bear, and took a lamb from the flock, I went after "
                "him and rescued it.'"
            ),
            "announcer": "Bible Bios! Young David is writing his résumé! What is his headline?",
        },
        {
            "id": "bb_paul_travel",
            "prompt": "Write the Apostle Paul's travel-blogger bio.",
            "decoy": "Three missionary journeys, three shipwrecks before Malta. Will travel for Jesus.",
            "bot_answers": [
                "Frequent sailor, occasional snake-shaker",
                "Tentmaker by trade, letter-writer by calling",
                "I've been to Rome, Corinth, and over a wall in a basket",
            ],
            "scripture_note": (
                "2 Corinthians 11:25-26 - 'Three times I was shipwrecked; a night and a day I was adrift at sea; on "
                "frequent journeys.'"
            ),
            "announcer": "Bible Bios! The Apostle Paul started a travel blog! Write his bio!",
        },
        {
            "id": "bb_samson_fitness",
            "prompt": "Write Samson's fitness-influencer bio.",
            "decoy": "Strongest man alive. Never had a haircut. Please don't ask why.",
            "bot_answers": [
                "Lifts city gates for fun",
                "Lion wrestler. Riddle lover. Nazirite since birth",
                "Learning to trust the right people. Working on it.",
            ],
            "scripture_note": (
                "Judges 13:5 - 'No razor shall come upon his head, for the child shall be a Nazirite to God from the "
                "womb.'"
            ),
            "announcer": "Bible Bios! Samson is now a fitness influencer! Write his profile bio!",
        },
        {
            "id": "bb_peter_fishing",
            "prompt": "Write Simon Peter's fishing-guide business listing.",
            "decoy": "Fishing trips daily. Nets may break from overfilling. Water-walking lessons extra.",
            "bot_answers": [
                "Caught 153 fish in one cast. Ask me how",
                "Former fisherman, now fishing for people",
                "Boat available, temper improving",
            ],
            "scripture_note": (
                "Luke 5:10 - Jesus said to Simon, 'Do not be afraid; from now on you will be catching men.'"
            ),
            "announcer": "Bible Bios! Simon Peter runs a fishing-guide business! Write his listing!",
        },
        {
            "id": "bb_nehemiah_builders",
            "prompt": "Write the slogan for Nehemiah's construction company.",
            "decoy": "Walls in 52 Days or Your Money Back",
            "bot_answers": [
                "We Build With a Mind to Work",
                "Ignoring Mockers Since Day One",
                "Rebuild, Rejoice, Repeat",
            ],
            "scripture_note": (
                "Nehemiah 6:15 - The wall was finished on the twenty-fifth day of the month Elul, in fifty-two days."
            ),
            "announcer": "Bible Bios! Nehemiah is starting a construction company! Write the slogan!",
        },
        {
            "id": "bb_deborah_headline",
            "prompt": "Write Deborah's professional profile headline.",
            "decoy": "Judge of Israel | Prophetess | Office Hours Under the Palm Tree",
            "bot_answers": [
                "Leading armies and settling disputes, palm tree optional",
                "Mother in Israel and excellent at giving directions",
                "Wise counsel available Monday through always",
            ],
            "scripture_note": (
                "Judges 4:4-5 - Deborah, a prophetess, was judging Israel, and she sat under the palm of Deborah "
                "where the people came to her for judgment."
            ),
            "announcer": "Bible Bios! Deborah needs a professional headline! What does it say?",
        },
        {
            "id": "bb_luke_bio",
            "prompt": "Write Luke's author bio for the back cover of his Gospel.",
            "decoy": "Physician. Historian. Wrote the most detailed travel journal of the early church.",
            "bot_answers": [
                "Doctor by trade, careful researcher by calling",
                "Author of two books, both addressed to my friend Theophilus",
                "Traveled with Paul. Took really good notes.",
            ],
            "scripture_note": "Colossians 4:14 - Luke the beloved physician greets you.",
            "announcer": "Bible Bios! Luke needs an author bio for the back cover! Write it!",
        },
        {
            "id": "bb_esther_bio",
            "prompt": "Write Queen Esther's social-media bio.",
            "decoy": "Orphan turned queen. Banquet planner. Here for such a time as this.",
            "bot_answers": [
                "Royal beauty treatments: twelve months. Courage: priceless",
                "Raised by cousin Mordecai, raising the bar for bravery",
                "Ask me about my two-banquet strategy",
            ],
            "scripture_note": (
                "Esther 2:7,17 - Mordecai raised his orphaned cousin Esther, and the king loved her and made her "
                "queen."
            ),
            "announcer": "Bible Bios! Queen Esther just made a profile! Write her bio!",
        },
        {
            "id": "bb_elijah_apprentice",
            "prompt": "Write the job listing for Elijah's apprentice prophet.",
            "decoy": "Must love mountain hikes, raven meals, and fire. Cloak provided after training.",
            "bot_answers": [
                "Benefits: a double portion upon completion",
                "Travel required, sometimes by whirlwind",
                "Must be willing to leave the family farm and twelve oxen behind",
            ],
            "scripture_note": (
                "1 Kings 19:19-21 - Elijah found Elisha plowing with twelve yoke of oxen and cast his cloak upon him, "
                "and Elisha followed him."
            ),
            "announcer": "Bible Bios! Elijah is hiring an apprentice! Write the job listing!",
        },
        {
            "id": "bb_abraham_bio",
            "prompt": "Write Abraham's social-media bio.",
            "decoy": "Father of many nations. Became a dad at 100. Still counting the stars.",
            "bot_answers": [
                "Moved from Ur. Left no forwarding address",
                "Tent life. Big family. Bigger promises",
                "Hospitality expert: angels have stayed at my place",
            ],
            "scripture_note": "Genesis 21:5 - Abraham was a hundred years old when his son Isaac was born to him.",
            "announcer": "Bible Bios! Father Abraham just joined social media! Write his bio!",
        },
        {
            "id": "bb_barnabas_headline",
            "prompt": "Write Barnabas's professional profile headline.",
            "decoy": "Chief Encouragement Officer | Sold a Field to Support the Church",
            "bot_answers": [
                "Son of Encouragement: It's Literally My Name",
                "Mentor to Paul and John Mark. Second chances available",
                "Team player. Generous giver. Excellent at introductions",
            ],
            "scripture_note": (
                "Acts 4:36-37 - Joseph, called Barnabas by the apostles (which means son of encouragement), sold a "
                "field and brought the money."
            ),
            "announcer": "Bible Bios! Barnabas needs a profile headline! What does it say?",
        },
        {
            "id": "bb_eden_gardener",
            "prompt": "Write the job listing for head gardener of the Garden of Eden.",
            "decoy": "Must name all animals, tend the garden, and avoid ONE specific tree. That's it.",
            "bot_answers": [
                "Great benefits, amazing views, no weeds (yet)",
                "Do NOT take advice from any talking snakes",
                "Free fruit, minus one tree. Read the fine print.",
            ],
            "scripture_note": (
                "Genesis 2:15-17 - God put the man in the garden to work it and keep it, and commanded him not to eat "
                "from one tree."
            ),
            "announcer": "Bible Bios! The Garden of Eden is hiring a head gardener! Write the job listing!",
        },
        {
            "id": "bb_priscilla_aquila",
            "prompt": "Write the business listing for Priscilla and Aquila's tent shop.",
            "decoy": "Tents made to order. Sound teaching included free. Ask about our house church.",
            "bot_answers": [
                "Quality tents, quality teaching. Apollos approves",
                "Family-owned, Spirit-led, waterproof guaranteed",
                "Now with a guest room for traveling apostles",
            ],
            "scripture_note": (
                "Acts 18:26 & Romans 16:5 - Priscilla and Aquila explained the way of God to Apollos, and a church "
                "met in their house."
            ),
            "announcer": "Bible Bios! Priscilla and Aquila are listing their tent shop! Write their listing!",
        },
        {
            "id": "bb_bezalel_craftsman",
            "prompt": "Write a job-listing line for Bezalel, the master craftsman who built the Tabernacle.",
            "decoy": "Master craftsman in gold, silver, bronze, stone and wood. Skill: divinely provided.",
            "bot_answers": [
                "Specialist in gold-covered furniture and very precise cubits",
                "Seeking apprentices good with tiny details and big tents",
                "My portfolio is literally the Tabernacle",
            ],
            "scripture_note": (
                "Exodus 31:2-5 - God filled Bezalel with skill, intelligence and craftsmanship to work in gold, "
                "silver, bronze, stone and wood."
            ),
            "announcer": "Bible Bios! Bezalel, the Tabernacle builder, is posting his job listing! Write it!",
        },
    ],
    "hymn_remix": [
        {
            "id": "hr_lost_keys",
            "prompt": "Write a hymn title for the moment Mom can't find the car keys on Sunday morning.",
            "decoy": "I Once Was Lost, Now I'm Found (in the Couch Cushions)",
            "bot_answers": [
                "Seek and Ye Shall Find (Under Yesterday's Mail)",
                "Great Is Thy Faithfulness, Spare Key",
                "Blessed Assurance, I Left Them in the Door",
            ],
            "scripture_note": "Luke 11:9 - Ask, and it will be given to you; seek, and you will find.",
            "announcer": "Hymn Remix! Mom can't find the car keys on Sunday morning! Write the hymn title!",
        },
        {
            "id": "hr_laundry_mountain",
            "prompt": "Write a worship-song title about the family's mountain of laundry.",
            "decoy": "How Great Thou Art at Folding Fitted Sheets",
            "bot_answers": [
                "Washed White as Snow (Except These Socks)",
                "Spin Cycle of Praise",
                "The Lonely Sock That Was Lost and Found",
            ],
            "scripture_note": "Psalm 51:7 - Wash me, and I shall be whiter than snow.",
            "announcer": (
                "Hymn Remix! The laundry pile is taller than the Tower of Babel! Write the worship song title!"
            ),
        },
        {
            "id": "hr_eat_veggies",
            "prompt": "Write one line of a hymn a parent sings to get the kids to eat their vegetables.",
            "decoy": "Daniel ate his veggies, so lift those peas on high!",
            "bot_answers": [
                "Broccoli, broccoli, little trees of grace",
                "O taste and see the carrots, they are good",
                "Every green bean is a blessing in disguise",
            ],
            "scripture_note": "Daniel 1:12 - 'Let us be given vegetables to eat and water to drink.'",
            "announcer": "Hymn Remix! It's dinnertime and nobody wants their vegetables! Sing us a line!",
        },
        {
            "id": "hr_traffic_jam",
            "prompt": "Write a hymn title for being stuck in traffic on the way to church.",
            "decoy": "Be Still, My Soul (Because the Car Certainly Is)",
            "bot_answers": [
                "When the Road Is Long and the Light Is Red",
                "Patience Like a River, Brake Lights Like a Sea",
                "It Is Well With My Soul, Even in Lane Three",
            ],
            "scripture_note": "Psalm 46:10 - 'Be still, and know that I am God.'",
            "announcer": (
                "Hymn Remix! The whole family is stuck in traffic on the way to church! Write the hymn title!"
            ),
        },
        {
            "id": "hr_bedtime_lullaby",
            "prompt": "Write one line of a lullaby hymn for a kid who will NOT go to sleep.",
            "decoy": "The Lord neither slumbers nor sleeps, but YOU, my child, absolutely must",
            "bot_answers": [
                "Close those eyes, the angels need a break too",
                "Count the stars like Abraham, but quietly",
                "One more glass of water, then amen for real",
            ],
            "scripture_note": "Psalm 121:4 - Behold, he who keeps Israel will neither slumber nor sleep.",
            "announcer": "Hymn Remix! It's way past bedtime and somebody is wide awake! Sing us a lullaby line!",
        },
        {
            "id": "hr_snow_day",
            "prompt": "Write a praise-song title for a surprise snow day with no school.",
            "decoy": "Showers of Blessing (the Frozen Kind)",
            "bot_answers": [
                "Whiter Than Snow, and Twelve Inches Deep",
                "This Is the Day That School Got Cancelled",
                "Let It Snow and Let It Glow",
            ],
            "scripture_note": "Job 37:6 - To the snow he says, 'Fall on the earth.'",
            "announcer": "Hymn Remix! It's a surprise snow day! Write the praise song title!",
        },
        {
            "id": "hr_dishwasher_turn",
            "prompt": "Write a hymn title about whose turn it is to unload the dishwasher.",
            "decoy": "Take My Plate and Let It Be (Put Away by Someone Else)",
            "bot_answers": [
                "A Mighty Fortress Is Our Cupboard",
                "Count Your Blessings, Then Count the Forks",
                "Clean Hearts, Clean Cups, Clean Hands",
            ],
            "scripture_note": "Galatians 6:2 - Bear one another's burdens.",
            "announcer": "Hymn Remix! Whose turn is it to unload the dishwasher? Write the hymn title!",
        },
        {
            "id": "hr_road_trip_hour_six",
            "prompt": "Write one line of a worship song for hour six of a family road trip.",
            "decoy": "How long, O Lord? Are we there yet? How long?",
            "bot_answers": [
                "Lead me to the next rest stop, I pray",
                "My cup runneth over, and so does the back seat",
                "Onward, Christian travelers, marching to the gas station",
            ],
            "scripture_note": "Psalm 13:1 - How long, O Lord? Will you forget me forever?",
            "announcer": "Hymn Remix! It's hour six of the family road trip! Sing us a line!",
        },
        {
            "id": "hr_grandpa_birthday",
            "prompt": "Write a hymn title for Grandpa's eightieth birthday party.",
            "decoy": "Ancient of Days (and Candles)",
            "bot_answers": [
                "Great Is Thy Faithfulness, and Thy Cake",
                "Eighty Years of Mercies New",
                "Gray Hair Is a Crown, Pass the Ice Cream",
            ],
            "scripture_note": "Proverbs 16:31 - Gray hair is a crown of glory; it is gained in a righteous life.",
            "announcer": "Hymn Remix! Grandpa is turning eighty! Write the hymn title for his party!",
        },
        {
            "id": "hr_new_puppy",
            "prompt": "Write one line of a praise song for the day the family brings home a new puppy.",
            "decoy": "All creatures great and small, especially this one chewing my shoe",
            "bot_answers": [
                "He made the stars, the seas, and this very wiggly tail",
                "Fearfully and wonderfully made, and also very fluffy",
                "Praise Him for the puppy, praise Him for the paper towels",
            ],
            "scripture_note": "Psalm 145:9 - The Lord is good to all, and his mercy is over all that he has made.",
            "announcer": "Hymn Remix! A new puppy just came home! Sing us a line of praise!",
        },
        {
            "id": "hr_first_school_day",
            "prompt": "Write a hymn title for a kid's very first day of school.",
            "decoy": "Be Not Afraid (of the Cafeteria)",
            "bot_answers": [
                "Lead Me Gently to Room Twelve",
                "I Will Not Be Shaken (by Long Division)",
                "Brave and Strong in My New Light-Up Shoes",
            ],
            "scripture_note": "Isaiah 41:10 - Fear not, for I am with you; be not dismayed, for I am your God.",
            "announcer": "Hymn Remix! It's the first day of school! Write the hymn title!",
        },
        {
            "id": "hr_spilled_milk",
            "prompt": "Write one line of a worship song for spilled milk at breakfast.",
            "decoy": "My cup overflows, and now so does the table",
            "bot_answers": [
                "There's power in the paper towels",
                "Mercies new every morning, and so are the spills",
                "Wipe it away, Lord, wipe it all away",
            ],
            "scripture_note": "Lamentations 3:22-23 - His mercies never come to an end; they are new every morning.",
            "announcer": "Hymn Remix! Somebody spilled the milk at breakfast! Sing us a line!",
        },
        {
            "id": "hr_grocery_trip",
            "prompt": "Write a hymn title for a family trip to the grocery store.",
            "decoy": "Our Daily Bread (and Snacks Not on the List)",
            "bot_answers": [
                "Shall We Gather at the Checkout?",
                "Give Us This Day Our Daily Coupons",
                "Cart of Ages, Wobbly Wheel",
            ],
            "scripture_note": "Matthew 6:11 - Give us this day our daily bread.",
            "announcer": "Hymn Remix! The whole family is going grocery shopping! Write the hymn title!",
        },
        {
            "id": "hr_rainy_saturday",
            "prompt": "Write a praise-song title for a rainy Saturday stuck indoors.",
            "decoy": "Forty Days? Lord, I Can't Even Do Four Hours",
            "bot_answers": [
                "There Shall Be Showers of Board Games",
                "Blessed Be the Blanket Fort",
                "Rain Down, but Please Not on the Picnic",
            ],
            "scripture_note": "Psalm 147:8 - He covers the heavens with clouds; he prepares rain for the earth.",
            "announcer": "Hymn Remix! It's raining all Saturday and everyone is stuck inside! Write the song title!",
        },
        {
            "id": "hr_first_haircut",
            "prompt": "Write a hymn title for a little kid's first haircut.",
            "decoy": "Samson Would Never (But Here We Are)",
            "bot_answers": [
                "Every Hair Is Numbered, Even the Ones on the Floor",
                "Snip, Snip, Hallelujah",
                "Lord, Keep Me Still Through the Clippers",
            ],
            "scripture_note": "Luke 12:7 - Even the hairs of your head are all numbered.",
            "announcer": "Hymn Remix! It's the first haircut! Write the hymn title!",
        },
        {
            "id": "hr_math_homework",
            "prompt": "Write one line of a worship song about math homework.",
            "decoy": "Teach us to number our days, and also these fractions",
            "bot_answers": [
                "Lord, I lift my pencil up",
                "You are bigger than my long division",
                "Count your blessings, carry the one",
            ],
            "scripture_note": "Psalm 90:12 - Teach us to number our days, that we may get a heart of wisdom.",
            "announcer": "Hymn Remix! It's math homework time! Sing us a line!",
        },
        {
            "id": "hr_monday_alarm",
            "prompt": "Write a hymn title for the 6 AM alarm on Monday morning.",
            "decoy": "Rise and Shine and Give God the Glory (in Ten More Minutes)",
            "bot_answers": [
                "Wake, O Sleeper (Seriously, Wake Up)",
                "Morning Has Broken, and So Has My Snooze Button",
                "Joy Comes in the Morning, Coffee Comes First",
            ],
            "scripture_note": "Psalm 30:5 - Weeping may tarry for the night, but joy comes with the morning.",
            "announcer": "Hymn Remix! The Monday morning alarm is ringing! Write the hymn title!",
        },
        {
            "id": "hr_moving_boxes",
            "prompt": "Write a hymn title for a family moving into a new house.",
            "decoy": "This World Is Not My Home, but This Box Says 'Kitchen'",
            "bot_answers": [
                "I Lift My Eyes Unto the Hills (and the New Neighborhood)",
                "Bless This House, Especially the Garage",
                "A Firm Foundation, Two Hundred Boxes",
            ],
            "scripture_note": "Psalm 127:1 - Unless the Lord builds the house, those who build it labor in vain.",
            "announcer": "Hymn Remix! The family is moving into a new house! Write the hymn title!",
        },
        {
            "id": "hr_christmas_photo",
            "prompt": "Write one line of a worship song for taking the family Christmas card photo.",
            "decoy": "Let everything that has breath praise the Lord, and SMILE",
            "bot_answers": [
                "Joy to the world, the photo's done, nobody blinked this year",
                "Peace on Earth, but not in this living room",
                "One more take, O Lord, just one more take",
            ],
            "scripture_note": "Psalm 150:6 - Let everything that has breath praise the Lord!",
            "announcer": "Hymn Remix! It's time for the family Christmas card photo! Sing us a line!",
        },
        {
            "id": "hr_cookie_jar",
            "prompt": "Write a hymn title for the kid caught with a hand in the cookie jar.",
            "decoy": "Just As I Am (Covered in Crumbs)",
            "bot_answers": [
                "Search Me, O God (but Maybe Not My Pockets)",
                "I Surrender All (the Cookies)",
                "Amazing Grace, How Sweet the Chocolate Chip",
            ],
            "scripture_note": "Psalm 139:23 - Search me, O God, and know my heart!",
            "announcer": "Hymn Remix! Somebody got caught with a hand in the cookie jar! Write the hymn title!",
        },
    ],
    "church_bulletin": [
        {
            "id": "cb_noah_volunteers",
            "prompt": (
                "Noah's congregation is putting out a bulletin. Write this week's volunteer sign-up announcement."
            ),
            "decoy": "Wanted: two volunteers of every kind for the Ark cleaning crew. Bring your own shovel.",
            "bot_answers": [
                "Rain-or-shine choir practice is now rain only",
                "Ark tours Saturday. Please do not feed the lions",
                "Lost: one dove. Last seen carrying an olive leaf",
            ],
            "scripture_note": "Genesis 6:22 - Noah did this; he did all that God commanded him.",
            "announcer": "Church Bulletin! Noah needs volunteers! Write the sign-up announcement!",
        },
        {
            "id": "cb_jericho_walkers",
            "prompt": "Write the bulletin announcement for Joshua's Jericho walking club.",
            "decoy": "Silent walking club meets daily. Trumpet section, please wait for day seven.",
            "bot_answers": [
                "Bring comfy sandals; we're doing seven laps Saturday",
                "Shouting practice postponed until further notice",
                "Parking near the city wall is NOT recommended this weekend",
            ],
            "scripture_note": (
                "Joshua 6:10 - Joshua commanded, 'You shall not shout or make your voice heard until the day I tell "
                "you to shout.'"
            ),
            "announcer": "Church Bulletin! Joshua's walking club needs members! Write the announcement!",
        },
        {
            "id": "cb_vbs_den",
            "prompt": "Write the bulletin announcement for this summer's Vacation Bible School theme: Daniel's Den.",
            "decoy": "Real lions will not be present. We promise. Mostly promise.",
            "bot_answers": [
                "Bring a flashlight, a buddy, and your bravest roar",
                "Snack this week: Daniel's vegetables. Sorry, kids",
                "Pickup is at 3 PM sharp at the den entrance",
            ],
            "scripture_note": (
                "Daniel 6:10 - Daniel got down on his knees three times a day and prayed and gave thanks before his "
                "God."
            ),
            "announcer": "Church Bulletin! Vacation Bible School is coming to Daniel's Den! Write the announcement!",
        },
        {
            "id": "cb_lost_found",
            "prompt": "Write the strangest item ever listed in the church bulletin's lost and found.",
            "decoy": "Found: one coat of many colors. Owner's brothers say it isn't theirs.",
            "bot_answers": [
                "Found: Elijah's cloak. Please claim before Elisha does",
                "Lost: one sandal, last seen near a burning bush",
                "Found: twelve baskets of leftover bread. Please take some home",
            ],
            "scripture_note": "Luke 15:6 - 'Rejoice with me, for I have found my sheep that was lost.'",
            "announcer": "Church Bulletin! Check out the lost and found section! What's the strangest item listed?",
        },
        {
            "id": "cb_choir_tryouts",
            "prompt": "Write the choir tryout announcement if King David were the new choir director.",
            "decoy": "Choir tryouts Thursday. Dancing in the aisles strongly encouraged.",
            "bot_answers": [
                "Harp players wanted. Must be able to calm a grumpy king",
                "All voices welcome; a joyful noise is in the job description",
                "All one hundred fifty psalms will be on the test",
            ],
            "scripture_note": "Psalm 150:4 - Praise him with tambourine and dance; praise him with strings and pipe!",
            "announcer": "Church Bulletin! King David is directing the choir! Write the tryout announcement!",
        },
        {
            "id": "cb_parking_notice",
            "prompt": "Write a bulletin announcement about the church parking lot that nobody saw coming.",
            "decoy": "Chariot parking is now in the back. Camels, please use the overflow lot.",
            "bot_answers": [
                "The last shall be first, but NOT in the parking lot",
                "Please do not leave donkeys tied to the minivans",
                "Valet parking is now available for visiting Magi",
            ],
            "scripture_note": "Matthew 20:16 - So the last will be first, and the first last.",
            "announcer": "Church Bulletin! There's a surprising new parking lot announcement! What does it say?",
        },
        {
            "id": "cb_endless_bake_sale",
            "prompt": "Write the church bake sale announcement if the widow of Zarephath were in charge.",
            "decoy": "Bake sale all year! Our flour jar and oil jug just will not run out.",
            "bot_answers": [
                "Bread available daily. Visiting prophets eat first",
                "Somehow we're never out of muffins. Don't ask, just enjoy",
                "Small batch, big miracle, endless refills",
            ],
            "scripture_note": (
                "1 Kings 17:16 - The jar of flour was not spent, neither did the jug of oil become empty, according "
                "to the word of the Lord."
            ),
            "announcer": "Church Bulletin! The widow of Zarephath is running the bake sale! Write the announcement!",
        },
        {
            "id": "cb_pastor_vacation",
            "prompt": "Write the bulletin note explaining that the pastor is on vacation this week.",
            "decoy": "Pastor is on a forty-day fishing retreat. Please do not call unless the boat sinks.",
            "bot_answers": [
                "Pastor is resting like it's the seventh day; please let him",
                "Pastor is touring the seven churches of Revelation by bus",
                "This week's guest preacher promises to finish before lunch",
            ],
            "scripture_note": (
                "Genesis 2:2 - On the seventh day God finished his work that he had done, and he rested."
            ),
            "announcer": "Church Bulletin! The pastor is on vacation! Write the bulletin note!",
        },
        {
            "id": "cb_peter_breakfast",
            "prompt": "Write the men's breakfast announcement if Simon Peter were cooking.",
            "decoy": "Men's breakfast on the beach at dawn. Fish provided. Bring your own toast.",
            "bot_answers": [
                "Bacon is NOT on the menu. Fish is. Always fish",
                "Meet at the boat ramp; bring a net and your appetite",
                "Breakfast is cooked over a charcoal fire. Seconds are free",
            ],
            "scripture_note": (
                "John 21:9 - When they got out on land, they saw a charcoal fire in place, with fish laid out on it, "
                "and bread."
            ),
            "announcer": "Church Bulletin! Simon Peter is cooking the men's breakfast! Write the announcement!",
        },
        {
            "id": "cb_nursery_reeds",
            "prompt": "Write the church nursery announcement if Pharaoh's daughter were running it.",
            "decoy": "Nursery now offers floating basket naps. Lifeguards on duty.",
            "bot_answers": [
                "Lost a baby in the reeds? Check with us first",
                "Helpers wanted; must be good at peeking from behind tall plants",
                "We found three babies this morning. Please claim yours",
            ],
            "scripture_note": (
                "Exodus 2:10 - She named him Moses, 'Because,' she said, 'I drew him out of the water.'"
            ),
            "announcer": "Church Bulletin! Pharaoh's daughter is running the nursery! Write the announcement!",
        },
        {
            "id": "cb_too_much_giving",
            "prompt": "Write the building-fund update if Moses were collecting gifts for the Tabernacle.",
            "decoy": "PLEASE STOP GIVING. We have more than enough. Signed, the very overwhelmed finance team.",
            "bot_answers": [
                "Gold, silver and goat hair still welcome. Mostly goat hair",
                "Building fund 100 percent funded. Committee in shock",
                "Thank you! Please take your extra earrings back",
            ],
            "scripture_note": (
                "Exodus 36:5-7 - 'The people bring much more than enough,' so Moses told them to stop, for they had "
                "more than enough."
            ),
            "announcer": "Church Bulletin! Moses has an update on the Tabernacle building fund! Write it!",
        },
        {
            "id": "cb_disciples_small_group",
            "prompt": "Write the small-group announcement if the twelve disciples were hosting a Bible study.",
            "decoy": "Small group meets Thursday. Peter will talk first. Thomas will have questions.",
            "bot_answers": [
                "Bring your own boat; we're meeting on the lake",
                "James and John will be arguing over the best seats",
                "Matthew will be collecting the snack money",
            ],
            "scripture_note": "Matthew 18:20 - 'Where two or three are gathered in my name, there am I among them.'",
            "announcer": "Church Bulletin! The twelve disciples are hosting a small group! Write the announcement!",
        },
        {
            "id": "cb_midnight_lockin",
            "prompt": "Write the youth group lock-in announcement if Paul and Silas were the chaperones.",
            "decoy": "Youth lock-in Friday! Midnight hymn sing included. Doors may open on their own.",
            "bot_answers": [
                "Bring a sleeping bag and earplugs; earthquakes possible",
                "Chaperones will be singing all night. ALL night",
                "Pizza at midnight, prayer at midnight, everything at midnight",
            ],
            "scripture_note": (
                "Acts 16:25 - About midnight Paul and Silas were praying and singing hymns to God, and the prisoners "
                "were listening."
            ),
            "announcer": "Church Bulletin! Paul and Silas are chaperoning the youth lock-in! Write the announcement!",
        },
        {
            "id": "cb_river_baptisms",
            "prompt": "Write the baptism announcement if John the Baptist were in charge of scheduling.",
            "decoy": "Baptisms in the Jordan River this Sunday. Locust snacks served afterward.",
            "bot_answers": [
                "Dress code: camel hair encouraged",
                "No pool. We're doing this the old-fashioned way, in the river",
                "Bring a towel and a ready heart",
            ],
            "scripture_note": (
                "Matthew 3:5-6 - Jerusalem and all Judea went out to him and were baptized by him in the river "
                "Jordan."
            ),
            "announcer": "Church Bulletin! John the Baptist is scheduling baptisms! Write the announcement!",
        },
        {
            "id": "cb_sermon_series",
            "prompt": "Write the title of the pastor's surprising new sermon series announced in the bulletin.",
            "decoy": "Leviticus: The Director's Cut (Twelve Weeks)",
            "bot_answers": [
                "Begats: Learning to Pronounce Everyone in First Chronicles",
                "Balaam's Donkey: Lessons From the Back Seat",
                "Numbers: Yes, There Will Be Math",
            ],
            "scripture_note": "2 Timothy 3:16 - All Scripture is breathed out by God and profitable for teaching.",
            "announcer": "Church Bulletin! The pastor is starting a surprising new sermon series! What's it called?",
        },
        {
            "id": "cb_coffee_bar",
            "prompt": "Write the bulletin announcement for the church's brand new coffee bar.",
            "decoy": "New coffee bar opens Sunday! Our espresso is so strong it could part a sea.",
            "bot_answers": [
                "Try the Jacob's Ladder latte: twelve shots, stacked to heaven",
                "Decaf available at the early service. Just kidding.",
                "Free refills, just like the widow's oil",
            ],
            "scripture_note": "Psalm 23:5 - My cup overflows.",
            "announcer": "Church Bulletin! The church is opening a coffee bar! Write the announcement!",
        },
        {
            "id": "cb_pageant_casting",
            "prompt": "Write the Christmas pageant casting call that nobody saw coming.",
            "decoy": "Now casting: one very calm donkey and three kids who can walk slowly carrying gifts.",
            "bot_answers": [
                "Wanted: a shepherd who will not wave at Grandma",
                "Angel roles open. Must say 'Fear not!' without giggling",
                "Real sheep auditions have been moved outside after last year",
            ],
            "scripture_note": (
                "Luke 2:8-10 - The angel said to the shepherds, 'Fear not, for behold, I bring you good news of great "
                "joy.'"
            ),
            "announcer": "Church Bulletin! The Christmas pageant is casting! Write the casting call!",
        },
        {
            "id": "cb_elijah_forecast",
            "prompt": "Write the church picnic weather update if Elijah were the forecaster.",
            "decoy": "No rain for three and a half years. Picnic is ON. Bring sunscreen.",
            "bot_answers": [
                "Small cloud spotted, about the size of a hand. Bring umbrellas",
                "Fire expected on the mountain; do not bring paper plates",
                "Forecast: dry, dry, dry, then a downpour after prayer",
            ],
            "scripture_note": (
                "James 5:17-18 - Elijah prayed that it might not rain, and for three years and six months it did not "
                "rain. Then he prayed again, and heaven gave rain."
            ),
            "announcer": "Church Bulletin! Elijah has the weather forecast for the church picnic! Write the update!",
        },
        {
            "id": "cb_magi_visitors",
            "prompt": "Write the 'Welcome, Visitors!' announcement for the Sunday the Magi showed up at church.",
            "decoy": "Welcome, travelers from the East! Camel parking is behind the fellowship hall.",
            "bot_answers": [
                "Gifts of gold accepted at the offering. Myrrh, please see the pastor",
                "Please fill out a visitor card and include your star's coordinates",
                "Our greeters will show you to a seat near the nativity scene",
            ],
            "scripture_note": (
                "Matthew 2:1-2 - Wise men from the east came to Jerusalem, saying, 'We saw his star when it rose and "
                "have come to worship him.'"
            ),
            "announcer": "Church Bulletin! The Magi have arrived at church! Write the welcome announcement!",
        },
        {
            "id": "cb_zacchaeus_dinner",
            "prompt": "Write the bulletin note about this week's fellowship dinner hosted by Zacchaeus.",
            "decoy": "Dinner at Zacchaeus's house! He's paying everyone back four times over. Bring a big bag.",
            "bot_answers": [
                "Seating is limited; shorter guests get the front row",
                "Tree climbing before dinner for anyone who can't see",
                "Tax refunds will be served with dessert",
            ],
            "scripture_note": "Luke 19:8 - 'If I have defrauded anyone of anything, I restore it fourfold.'",
            "announcer": "Church Bulletin! Zacchaeus is hosting the fellowship dinner! Write the bulletin note!",
        },
    ],
}
