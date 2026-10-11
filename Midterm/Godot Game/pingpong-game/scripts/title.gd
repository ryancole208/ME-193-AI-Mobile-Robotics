extends Control

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_accept"):
		get_tree().change_scene_to_file("res://scenes/SongSelect.tscn")
	elif event.is_action_pressed("credits"):
		get_tree().change_scene_to_file("res://scenes/Credits.tscn")
