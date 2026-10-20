extends Control
const SWING_DELAY_MS := 1500
var opened_msec: int = 0
const GRADES := [
	[95.0, "S", "FFEC27"],
	[85.0, "A", "00E436"],
	[70.0, "B", "29ADFF"],
	[50.0, "C", "FFA300"],
	[0.0, "D", "FF004D"],
]

func show_grade(accuracy: float) -> void:
	for g in GRADES:
		if accuracy >= g[0]:
			$Center/Grade.text = g[1]
			$Center/Grade.modulate = Color(g[2])
		
			$Center/Grade.pivot_offset = $Center/Grade.size / 2.0
			$Center/Grade.scale = Vector2(2.0, 2.0)
			create_tween().tween_property($Center/Grade, "scale", Vector2.ONE, 0.3) \
				.set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
		
			return

func _ready() -> void:
	var r: Dictionary = GameState.results
	$Center/SongTitle.text = r["title"]
	$Center/Stats.text = "Accuracy %.1f%%\nPerfect %d  Good %d  Miss %d\nBest streak %d" % [
		r["accuracy"], r["counts"]["Perfect"], r["counts"]["Good"],
		r["counts"]["Miss"], r["best_streak"]]
	if r["full_run"] and GameState.submit_score(r["title"], r["accuracy"]):
		$Center/Stats.text += "\nNEW BEST!"
	opened_msec = Time.get_ticks_msec()
	Bridge.swing_received.connect(_on_swing)
	show_grade(r["accuracy"])

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_accept"):
		Fader.go("res://scenes/SongSelect.tscn")

func _on_swing(_age_s: float, _strength: float) -> void:
	if Time.get_ticks_msec() - opened_msec < SWING_DELAY_MS:
		return
	Fader.go("res://scenes/SongSelect.tscn")
