extends Node

@export var conductor: Conductor
var hits: Array[Dictionary] = []
@export var ball: Node2D
@export var opponent_spot: Vector2 = Vector2(192, 100)
@export var player_spot: Vector2 = Vector2(192, 195)
var flight: int = 0
@export var player_hit_height: float = 12.0
@export var opponent_hit_height: float = 24.0
@export var arc_height: float = 16.0
@export var bounce_at: float = 0.6
@export var player_x_range: Vector2 = Vector2(96, 288)
@export var opponent_x_range: Vector2 = Vector2(144, 240)
@export var opponent: Node2D
signal judged(grade: String, offset_ms: float)
@export var perfect_window_ms: float = 50.0
@export var good_window_ms: float = 120.0
@export var miss_grace_ms: float = 120.0
@export var paddle: Node2D
@export var aim_tolerance_px: float = 15.0



func _ready() -> void:
	build_hits()

func _process(_delta: float) -> void:
	var now: float = conductor.song_position
	var late_limit: float = (good_window_ms + miss_grace_ms) / 1000.0
	for i in range(maxi(flight - 2, 0), mini(flight + 2, hits.size())):
		var h: Dictionary = hits[i]
		if h["by"] == "player" and not h["judged"] and now > h["time"] + late_limit:
			h["judged"] = true
			h["grade"] = "Miss"
			judged.emit("Miss", 0.0)
			print("Miss")
	while flight < hits.size() - 2 and now >= hits[flight + 1]["time"]:
		flight += 1
	var a: Dictionary = hits[flight]
	var b: Dictionary = hits[flight + 1]
	var t: float = clampf(inverse_lerp(a["time"], b["time"], now), 0.0, 1.0)
	if a["by"] == "player" and not a["judged"]:
		t = 0.0
	var from: Vector2 = spot_for(a)
	var to: Vector2 = spot_for(b)
	ball.position = from.lerp(to, t)
	var height: float
	var h_from: float = opponent_hit_height if a["by"] == "opponent" else player_hit_height
	var h_to: float = opponent_hit_height if b["by"] == "opponent" else player_hit_height
	if t < bounce_at:
		var u: float = t / bounce_at
		height = lerpf(h_from, 0.0, u) + 4.0 * arc_height * u * (1.0 - u)
	else:
		var u: float = (t - bounce_at) / (1.0 - bounce_at)
		height = lerpf(0.0, h_to, u) + 2.0 * arc_height * u * (1.0 - u)
	var sprite: Sprite2D = ball.get_node("Sprite")
	var depth: float = inverse_lerp(opponent_spot.y, player_spot.y, ball.position.y)
	height *= lerpf(0.5, 1.0, depth)
	sprite.position.y = -height
	sprite.frame = clampi(roundi(depth * 4.0), 0, 4)
	if a["by"] == "opponent":
		opponent.position.x = a["x"]
	else:
		var prev_x: float = hits[flight - 1]["x"] if flight > 0 else b["x"]
		opponent.position.x = lerpf(prev_x, b["x"], t)
	var missed: bool = a.get("grade", "") == "Miss" \
		or (not a["judged"] and now > a["time"] + good_window_ms / 1000.0)
	ball.visible = a["by"] == "opponent" or not missed

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
	var rng := RandomNumberGenerator.new()
	rng.seed = hash(conductor.chart["title"])
	for h in hits:
		h["time"] = conductor.beat_to_seconds(h["beat"])
		var r: Vector2 = player_x_range if h["by"] == "player" else opponent_x_range
		h["x"] = rng.randf_range(r.x, r.y)
		h["judged"] = false

func spot_for(hit: Dictionary) -> Vector2:
	var y: float = opponent_spot.y if hit["by"] == "opponent" else player_spot.y
	return Vector2(hit["x"], y)


func _on_paddle_swung() -> void:
	var now: float = conductor.song_position
	var best: int = -1
	var best_offset: float = INF
	for i in range(maxi(flight - 2, 0), mini(flight + 3, hits.size())):
		if hits[i]["by"] != "player" or hits[i]["judged"]:
			continue
		var offset: float = now - hits[i]["time"]
		if absf(offset) < absf(best_offset):
			best = i
			best_offset = offset
	var ms: float = best_offset * 1000.0
	if best == -1 or absf(ms) > good_window_ms:
		return
	hits[best]["judged"] = true
	var grade: String = "Perfect" if absf(ms) <= perfect_window_ms else "Good"
	var blade_x: float = paddle.to_global(Vector2(0, -16)).x
	if absf(blade_x - hits[best]["x"]) > aim_tolerance_px:
		grade = "Miss"
	hits[best]["grade"] = grade
	judged.emit(grade, ms)
	print(grade, " ", roundi(ms))
