extends Node
var results: Dictionary = {}

var chart_path: String = "res://charts/Theme_for_Harold_var_3.chart.json"
const SAVE_PATH := "user://scores.cfg"
var scores := ConfigFile.new()
var input_offset_ms: float = 0.0
var music_volume: float = 1.0
var sfx_volume: float = 1.0

func _ready() -> void:
	scores.load(SAVE_PATH)
	input_offset_ms = scores.get_value("settings", "input_offset_ms", 0.0)
	music_volume = scores.get_value("settings", "music_volume", 1.0)
	sfx_volume = scores.get_value("settings", "sfx_volume", 1.0)
	apply_volumes()

func high_score(title: String) -> float:
	return scores.get_value("high_scores", title, -1.0)

func submit_score(title: String, accuracy: float) -> bool:
	if accuracy <= high_score(title):
		return false
	scores.set_value("high_scores", title, accuracy)
	scores.save(SAVE_PATH)
	return true
func save_input_offset(ms: float) -> void:
	input_offset_ms = ms
	scores.set_value("settings", "input_offset_ms", ms)
	scores.save(SAVE_PATH)

func apply_volumes() -> void:
	AudioServer.set_bus_volume_db(AudioServer.get_bus_index("Music"), linear_to_db(music_volume))
	AudioServer.set_bus_volume_db(AudioServer.get_bus_index("SFX"), linear_to_db(sfx_volume))

func save_volumes() -> void:
	scores.set_value("settings", "music_volume", music_volume)
	scores.set_value("settings", "sfx_volume", sfx_volume)
	scores.save(SAVE_PATH)
