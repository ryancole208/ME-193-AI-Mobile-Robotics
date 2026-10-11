extends Control

func _ready() -> void:
	var lines: Array[String] = ["CREDITS", "", "Game by Ryan Cole", "", "MUSIC"]
	for file in DirAccess.get_files_at("res://charts"):
		if file.ends_with(".json"):
			var chart: Dictionary = JSON.parse_string(
				FileAccess.get_file_as_string("res://charts/" + file))
			lines.append(chart["attribution"]["credit"])
			lines.append("")
	lines.append("Font: Pixel Operator by Jayvee Enaguas (CC0)")
	$Scroll/Text.text = "\n".join(lines)

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_accept") or event.is_action_pressed("ui_cancel"):
		get_tree().change_scene_to_file("res://scenes/Title.tscn")
