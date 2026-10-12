extends Control
const CALIBRATION_CHART := "res://charts/Cut_and_Dry.chart.json"

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_accept"):
		get_tree().change_scene_to_file("res://scenes/SongSelect.tscn")
	elif event.is_action_pressed("credits"):
		get_tree().change_scene_to_file("res://scenes/Credits.tscn")
	elif event.is_action_pressed("options"):
		get_tree().change_scene_to_file("res://scenes/Options.tscn")
	elif event.is_action_pressed("calibrate"):
		GameState.chart_path = CALIBRATION_CHART
		get_tree().change_scene_to_file("res://scenes/calibration.tscn")

func _ready() -> void:
	Bridge.swing_received.connect(_on_swing)

func _on_swing(_age_s: float, _strength: float) -> void:
	get_tree().change_scene_to_file("res://scenes/SongSelect.tscn")
