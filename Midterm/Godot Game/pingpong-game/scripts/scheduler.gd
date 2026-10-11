extends Node

@export var conductor: Conductor
var hits: Array[Dictionary] = []
@export var ball: Node2D
@export var opponent_spot: Vector2 = Vector2(192, 100)
@export var player_spot: Vector2 = Vector2(192, 195)
var flight: int = 0
@export var hit_height: float = 12.0
@export var arc_height: float = 16.0
@export var bounce_at: float = 0.6

func _ready() -> void:
	build_hits()

func _process(_delta: float) -> void:
	var now: float = conductor.song_position
	while flight < hits.size() - 2 and now >= hits[flight + 1]["time"]:
		flight += 1
	var a: Dictionary = hits[flight]
	var b: Dictionary = hits[flight + 1]
	var t: float = clampf(inverse_lerp(a["time"], b["time"], now), 0.0, 1.0)
	var from: Vector2 = opponent_spot if a["by"] == "opponent" else player_spot
	var to: Vector2 = player_spot if a["by"] == "opponent" else opponent_spot
	ball.position = from.lerp(to, t)
	var height: float
	if t < bounce_at:
		var u: float = t / bounce_at
		height = lerpf(hit_height, 0.0, u) + 4.0 * arc_height * u * (1.0 - u)
	else:
		var u: float = (t - bounce_at) / (1.0 - bounce_at)
		height = lerpf(0.0, hit_height, u) + 2.0 * arc_height * u * (1.0 - u)
	ball.get_node("Sprite").position.y = -height

func pattern_for_bar(bar: int) -> Array:
	for section in conductor.chart["sections"]:
		if bar >= section["bars"][0] and bar <= section["bars"][1]:
			return section["pattern"]
	return conductor.chart["pattern"]
func build_hits() -> void:
	hits.clear()
	var first_bar: int = int(conductor.chart["start_bar"])
	var last_bar: int = int(conductor.chart["end_bar"])
	for bar in range(first_bar, last_bar + 1):
		var start: float = conductor.bar_to_beat(bar)
		if pattern_for_bar(bar).size() == 1:
			hits.append({"beat": start + 1.0, "by": "opponent"})
			hits.append({"beat": start + 3.0, "by": "player"})
		else:
			for i in range(4):
				hits.append({"beat": start + i, "by": "opponent" if i % 2 == 0 else "player"})
	for h in hits:
		h["time"] = conductor.beat_to_seconds(h["beat"])
