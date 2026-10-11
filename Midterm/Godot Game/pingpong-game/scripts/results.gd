extends Control

func _ready() -> void:
	var r: Dictionary = GameState.results
	$Center/SongTitle.text = r["title"]
	$Center/Stats.text = "Accuracy %.1f%%\nPerfect %d  Good %d  Miss %d\nBest streak %d" % [
		r["accuracy"], r["counts"]["Perfect"], r["counts"]["Good"],
		r["counts"]["Miss"], r["best_streak"]]
	if r["full_run"] and GameState.submit_score(r["title"], r["accuracy"]):
		$Center/Stats.text += "\nNEW BEST!"

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_accept"):
		get_tree().change_scene_to_file("res://scenes/SongSelect.tscn")
