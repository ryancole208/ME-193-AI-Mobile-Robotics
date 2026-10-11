extends Node
var results: Dictionary = {}

var chart_path: String = "res://charts/Theme_for_Harold_var_3.chart.json"
const SAVE_PATH := "user://scores.cfg"
var scores := ConfigFile.new()

func _ready() -> void:
	scores.load(SAVE_PATH)

func high_score(title: String) -> float:
	return scores.get_value("high_scores", title, -1.0)

func submit_score(title: String, accuracy: float) -> bool:
	if accuracy <= high_score(title):
		return false
	scores.set_value("high_scores", title, accuracy)
	scores.save(SAVE_PATH)
	return true
