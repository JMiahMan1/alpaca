import json
from pathlib import Path

import pytest

from llm_benchmark_suite import LLMModelBenchmark


@pytest.mark.parametrize("test_id", ["bash_backup_rotate", "bash_health_loop", "bash_csv_sums", "bash_top_cpu"])
@pytest.mark.parametrize("shebang", ["#!/bin/bash", "#!/usr/bin/env bash", "#!/usr/local/bin/bash -e"])
def test_bash_verifier_accepts_bash_shebangs(test_id, shebang):
    response = f"```bash\n{shebang}\ntar backup; for x in files; do curl url; sleep 1; done\nawk sum; ps cpu\n```"
    assert LLMModelBenchmark()._verify_functional_response(test_id, response)


@pytest.mark.parametrize(
    "code,expected",
    [
        ('print "1,1,2,3,5,8,13"', True),
        ('print "0 1 1 2 3 5 8"', True),
        ("print 1\nprint 1\nprint 2\nprint 3\nprint 5\nprint 8", True),
        ('print "1,1,2,3,5,9"', False),
        ('print "1,1,2,3"', False),
        ('print "2,3,5,8,13,21"', False),
        ("for n = 1 to 10\nprint fib(n)\nnext n", True),
    ],
)
def test_basic_fibonacci_numeric_sequence(code, expected):
    assert LLMModelBenchmark()._verify_functional_response("bas_fibonacci", f"```basic\n{code}\n```") is expected


VOXEL = """
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
blocks = {(0, 0, 0): "grass", (0, -1, 0): "dirt", (0, -2, 0): "stone"}
screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
gluPerspective(45, 800 / 600, 0.1, 100)
angle = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_DELETE:
            blocks.pop((0, 0, 0), None)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_INSERT:
            blocks[(0, 0, 0)] = "grass"
    angle += 0.1
    glRotatef(angle, 0, 1, 0)
    glBegin(GL_QUADS)
    for x, y, z in blocks:
        for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1)):
            glVertex3f(x + dx, y + dy, z)
    glEnd()
    pygame.display.flip()
"""


JOKES = """1. **Setup:** Why did I buy a tiny ladder for my herb garden?
   **Punchline:** I wanted to take my thyme to the next level.

2. **Setup:** Why did I put wheels on my filing cabinet?
   **Punchline:** I wanted to take my records for a spin.

3. **Setup:** I named my vacuum cleaner Alibi.
   **Punchline:** It always covers my tracks.
"""

WIREFRAME = """┌───────────────────────────┐
│ HEADER: Create account    │
│ FORM                      │
│ Name     [____________]   │
│ Email    [____________]   │
│ Password [____________]   │
│ PRIMARY CTA               │
│ [Create account]          │
└───────────────────────────┘"""

SCOREBOARD = """
import json
score = 0
with open("high_scores.json") as handle:
    high_scores = json.load(handle)
def finish_run():
    name = input("Initials: ")
    entry = {"initials": name, "score": score}
    high_scores.append(entry)
    high_scores.sort(key=lambda row: row["score"], reverse=True)
    with open("high_scores.json", "w") as handle:
        json.dump(high_scores[:5], handle)
    with open("/tmp/alpaca_score.json", "w") as handle:
        json.dump(entry, handle)
def new_game():
    global score
    score = 0
"""

GAMES = {
    "retro_temple_run": """
lanes = [200, 400, 600]
if keys[pygame.K_LEFT]: lane = max(0, lane - 1)
if keys[pygame.K_RIGHT]: lane = min(2, lane + 1)
if keys[pygame.K_UP]: jump_velocity = 10
speed = 27 + elapsed * 0.65
for obstacle in obstacles:
    obstacle.z -= speed * dt
    if obstacle.lane == lane and jump_height < 1:
        finish_run()
""",
    "retro_donkey_kong": """
floors = [640, 530, 420, 310, 200, 90]
ladder_x = [500, 100, 480, 130, 470]
if keys[pygame.K_UP] and on_ladder:
    player.y -= climb_speed * dt
for barrel in barrels:
    barrel.x += barrel.vx * dt
    if barrel.rect.colliderect(player): finish_run()
if player.colliderect(princess): finish_run()
for y in floors:
    pygame.draw.line(screen, RED, (40, y), (600, y), 8)
""",
    "retro_mario": """
platforms = [pygame.Rect(0, 500, 400, 20)]
flag = pygame.Rect(600, 300, 10, 200)
vx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 200
if keys[pygame.K_SPACE] and grounded:
    vy = -650
vy = min(950.0, vy + 1800.0 * dt)
x += vx * dt
y += vy * dt
if y > 600 or player.colliderect(flag):
    finish_run()
""",
    "retro_arcade_loop": """
lives = 3
wave = 1
ship.velocity += thrust * direction(angle) * dt
ship.position += ship.velocity * dt
wrap(ship.position)
if keys[pygame.K_SPACE]: bullets.append(Bullet(ship.position, angle))
for asteroid in asteroids:
    if touching(bullet, asteroid):
        split(asteroid)
        score += 100
if not asteroids:
    wave += 1
    spawn_wave(wave)
pygame.draw.polygon(screen, GREEN, ship.vertices, 1)
""",
    "retro_echo_dolphin": """
sea_top = 60
if keys[pygame.K_LEFT]: dolphin.velocity.x -= 100 * dt
if keys[pygame.K_RIGHT]: dolphin.velocity.x += 100 * dt
if keys[pygame.K_UP]: dolphin.velocity.y -= 100 * dt
if keys[pygame.K_DOWN]: dolphin.velocity.y += 100 * dt
dolphin.position += dolphin.velocity * dt
for reef in reefs:
    reef.x -= scroll_speed * dt
    if dolphin.rect.colliderect(reef): finish_run()
""",
}

FPS = """
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
gluPerspective(70, 800 / 600, 0.1, 100)
yaw += (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]) * 90 * dt
if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
    projectiles.append([camera.x, camera.z, math.sin(yaw), -math.cos(yaw)])
for p in projectiles:
    p[0] += p[2] * dt * 20
    p[1] += p[3] * dt * 20
    if abs(p[0] - target.x) < 0.5 and abs(p[1] - target.z) < 0.5:
        target.alive = False
draw_walls()
draw_target_cube()
draw_crosshair()
"""


@pytest.fixture
def benchmark():
    return LLMModelBenchmark.__new__(LLMModelBenchmark)


@pytest.fixture(scope="module")
def prompts():
    config = json.loads((Path(__file__).resolve().parents[1] / "benchmark_tests.json").read_text())
    return {test["id"]: test for group in config.values() if isinstance(group, list) for test in group}


@pytest.mark.parametrize("test_id", GAMES)
def test_retro_prompt_requirements_without_optional_design_features(benchmark, prompts, test_id):
    response = GAMES[test_id] + SCOREBOARD
    assert benchmark._verify_functional_response(prompts[test_id], response)
    assert benchmark._verify_functional_response(test_id, response)
    assert not benchmark._verify_functional_response(prompts[test_id], GAMES[test_id])


@pytest.mark.parametrize(
    ("test_id", "required", "replacement"),
    [
        ("retro_temple_run", "jump", "hop"),
        ("retro_temple_run", "lane", "path"),
        ("retro_temple_run", "obstacle", "coin"),
        ("retro_donkey_kong", "ladder", "rope"),
        ("retro_donkey_kong", "barrel", "ball"),
        ("retro_donkey_kong", "princess", "marker"),
        ("retro_mario", "vy = min(950.0, vy + 1800.0 * dt)", "vy = 0"),
        ("retro_mario", "flag", "decoration"),
        ("retro_arcade_loop", "split(asteroid)", "asteroids.remove(asteroid)"),
        ("retro_arcade_loop", "bullets", "sparks"),
        ("retro_arcade_loop", "pygame.K_SPACE", "ambient_effect"),
        ("retro_arcade_loop", "lives", "attempts"),
        ("retro_echo_dolphin", "reef", "pearl"),
        ("retro_echo_dolphin", "colliderect", "contains"),
    ],
)
def test_retro_missing_core_mechanic(benchmark, test_id, required, replacement):
    response = GAMES[test_id].replace(required, replacement) + SCOREBOARD
    if required == "bullets":
        response = response.replace("Bullet(", "Spark(").replace("touching(bullet,", "touching(spark,")
    assert not benchmark._verify_functional_response(test_id, response)


@pytest.mark.parametrize("layout", [WIREFRAME, WIREFRAME.replace("│", "|").replace("─", "-")])
def test_wireframe_glyphs_not_meta_keywords(benchmark, prompts, layout):
    assert benchmark._verify_functional_response(prompts["uiux_wireframe"], layout)


def test_wireframe_box_description(benchmark):
    response = "Header region: Sign up\nForm box: Name input, Email input, Password input\nAction region: primary CTA button Create account"
    assert benchmark._verify_functional_response("uiux_wireframe", response)


@pytest.mark.parametrize("field", ["Name", "Email", "Password", "PRIMARY CTA\n│ [Create account]"])
def test_wireframe_requires_all_fields(benchmark, field):
    response = WIREFRAME.replace(field, "")
    if field.startswith("PRIMARY"):
        response = WIREFRAME.replace("PRIMARY CTA", "").replace("[Create account]", "")
    assert not benchmark._verify_functional_response("uiux_wireframe", response)


@pytest.mark.parametrize(
    "response",
    [
        "A wireframe has name, email, password and a primary CTA button. Sign up.",
        "Header: Sign up\nForm: Name Email Password\nFooter: copyright",
        "[Name] [Email] [Password] [Create account]",
    ],
)
def test_wireframe_requires_labeled_layout_and_action(benchmark, response):
    assert not benchmark._verify_functional_response("uiux_wireframe", response)


@pytest.mark.parametrize(
    "response",
    [JOKES, JOKES.replace("**Setup:** ", "").replace("**Punchline:** ", "")],
)
def test_three_jokes_without_because_or_dad_keywords(benchmark, prompts, response):
    assert benchmark._verify_functional_response(prompts["life_dad_joke"], response)


@pytest.mark.parametrize(
    "response",
    [
        JOKES.split("\n\n3.")[0],
        JOKES + "\n4. Why did the lamp stop? It was light headed.",
        "1. Setup: Why did the clock stop?\n2. Setup: Why did the chair move?\n3. Setup: Why did the plant wilt?",
        "1. Setup: Punchline:\n2. Setup: Punchline:\n3. Setup: Punchline:",
        "A dad joke needs a setup and punchline. Why? Because it should make you groan.",
        JOKES.replace("thyme", "fucking thyme"),
    ],
)
def test_jokes_require_three_complete_clean_pairs(benchmark, response):
    assert not benchmark._verify_functional_response("life_dad_joke", response)


@pytest.mark.parametrize(
    "condition",
    ["i < n", "i<n", "i <= n - 1"],
)
def test_offbyone_correct_bound_with_explanation(benchmark, prompts, condition):
    response = f"for (int i = 0; {condition}; i++) {{ dest[i] = src[i]; }}\nValid indices are 0 through n - 1."
    assert benchmark._verify_functional_response(prompts["debug_offbyone"], response)


@pytest.mark.parametrize(
    "response",
    [
        "The off-by-one error is in the array loop boundary. Fix the index bug.",
        "for (int i=0; i<=n; i++) { dest[i]=src[i]; } Valid indices are 0 through n - 1.",
        "for (int i=0; i<n+1; i++) { dest[i]=src[i]; } Valid indices are 0 through n - 1.",
        "for (int i=0; i<n; i++) { dest[i]=src[i]; }",
    ],
)
def test_offbyone_needs_correction_and_explanation(benchmark, response):
    assert not benchmark._verify_functional_response("debug_offbyone", response)


@pytest.mark.parametrize("increment", ["i++;", "++i;", "i += 1;", "i = i + 1;"])
def test_loop_increment_without_fix_keyword(benchmark, prompts, increment):
    response = f"while (i < n) {{ work(); {increment} }}\nAdvancing i toward n lets the loop terminate."
    assert benchmark._verify_functional_response(prompts["debug_infinite_loop"], response)


@pytest.mark.parametrize(
    "response",
    [
        "Fix the infinite loop: update its condition so it terminates.",
        "while (i < n) { work(); i--; } This will terminate the loop.",
        "while (i < n) { work(); i += 0; } This will terminate the loop.",
        "while (i < n) { work(); n++; } This will terminate the loop.",
        "while (i < n) { work(); i++; }",
    ],
)
def test_loop_requires_progress_and_explanation(benchmark, response):
    assert not benchmark._verify_functional_response("debug_infinite_loop", response)


@pytest.mark.parametrize("update", ["vy += 0.6", "vy = vy + 0.6", "vy = min(950, vy + 1800 * dt)"])
def test_mario_numeric_vertical_acceleration(benchmark, update):
    response = GAMES["retro_mario"].replace("vy = min(950.0, vy + 1800.0 * dt)", update) + SCOREBOARD
    assert benchmark._verify_functional_response("retro_mario", response)


@pytest.mark.parametrize("update", ["vx += 0.6", "vy = 0.6", "vy += 0", 'label = "vy += 0.6"'])
def test_mario_rejects_non_acceleration(benchmark, update):
    response = GAMES["retro_mario"].replace("vy = min(950.0, vy + 1800.0 * dt)", update) + SCOREBOARD
    assert not benchmark._verify_functional_response("retro_mario", response)


@pytest.mark.parametrize("shooting", [FPS, FPS.replace("projectiles", "bullets")])
def test_fps_projectiles_without_raycast_or_hitscan(benchmark, prompts, shooting):
    assert benchmark._verify_functional_response(prompts["game_fps"], shooting)


def test_fps_hitscan_remains_valid(benchmark):
    response = (
        FPS[: FPS.index("if event.type")]
        + """
if keys[pygame.K_SPACE]:
    target.alive = not hitscan(camera, yaw, target)
draw_walls()
draw_target_cube()
draw_crosshair()
"""
    )
    assert benchmark._verify_functional_response("game_fps", response)


@pytest.mark.parametrize("required", ["projectiles", "target", "crosshair", "walls", "gluPerspective"])
def test_fps_missing_scene_or_shooting_requirement(benchmark, required):
    assert not benchmark._verify_functional_response("game_fps", FPS.replace(required, "missing"))


@pytest.mark.parametrize("required", ["glRotatef", "gluPerspective", "grass", "glVertex3f"])
def test_voxel_requires_rotating_3d_chunk(benchmark, required):
    response = VOXEL.replace(required, "missing")
    if required == "grass":
        response = response.replace("dirt", "missing")
    if required == "glVertex3f":
        response = response.replace("GL_QUADS", "missing")
    assert not benchmark._verify_functional_response("game_minecraft_voxel", response)


def test_voxel_mutable_rotating_chunk_without_gravity(benchmark):
    assert benchmark._verify_functional_response("game_minecraft_voxel", VOXEL)
    static = VOXEL.replace("blocks.pop((0, 0, 0), None)", "selected = (0, 0, 0)").replace(
        'blocks[(0, 0, 0)] = "grass"', "selected = (0, 0, 0)"
    )
    assert not benchmark._verify_functional_response("game_minecraft_voxel", static)
