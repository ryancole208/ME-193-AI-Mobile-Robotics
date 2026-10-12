extends Control
const SWING_DELAY_MS := 1500
var opened_msec: int = 0

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

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_accept"):
		Fader.go("res://scenes/SongSelect.tscn")

func _on_swing(_age_s: float, _strength: float) -> void:
	if Time.get_ticks_msec() - opened_msec < SWING_DELAY_MS:
		return
	Fader.go("PASTE song_select PATH")
