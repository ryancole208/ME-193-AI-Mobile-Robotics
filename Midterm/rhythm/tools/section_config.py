"""
Tunables for tools/suggest_sections.py (the [2,4] section suggester).

Every value here can be edited freely; --target and --phrase on the command line override
TARGET_HARD_FRACTION and PHRASE_BARS for one run.
"""

from pathlib import Path

RHYTHM = Path(__file__).resolve().parent.parent
GODOT = RHYTHM.parent / "Godot Game" / "pingpong-game"

# -- where things are ----------------------------------------------------------------------
CHARTS_DIR = GODOT / "charts"                  # read only, never written
MUSIC_DIR = GODOT / "audio" / "music"          # the chart's "song" field is a file name in here
OUT_DIR = RHYTHM / "tools" / "suggested_charts"

# -- selection -----------------------------------------------------------------------------
TARGET_HARD_FRACTION = 0.33     # roughly this share of the playable bars becomes hard
PHRASE_BARS = 8                 # phrases are this many bars, counted from start_bar
MIN_SECTION_BARS = 4            # shortest hard section; a shorter leftover at the end joins
                                # the phrase before it
INTRO_EASY_BARS = 8             # the first this-many playable bars always stay easy
SPLIT_HALVES = True             # allow a 4-bar section when a phrase's halves differ a lot
SPLIT_DIFF = 0.75               # ...by more than this (in score units, i.e. song std devs)

HARD_PATTERN = [2, 4]           # pattern written into the suggested sections

# -- energy score (per bar) ----------------------------------------------------------------
# Both features are z-normalised over the playable bars of each song, so a song is only ever
# compared with itself. score = W_RMS * z(rms dB) + W_ONSET * z(percussive onset strength)
W_RMS = 0.5
W_ONSET = 0.5
ANALYSIS_SR = 22050             # audio is mixed to mono and resampled to this (Hz)
HOP_LENGTH = 512                # onset envelope hop (samples)
USE_PERCUSSIVE = True           # onset strength on the HPSS percussive part (else the full mix)

# -- terminal output -----------------------------------------------------------------------
GRAPH_WIDTH = 40                # characters for the loudest bar
GRAPH_CHAR = "█"
GRAPH_FALLBACK_CHAR = "#"       # used if the console can't print GRAPH_CHAR
