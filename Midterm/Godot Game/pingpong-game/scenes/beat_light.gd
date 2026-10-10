extends ColorRect

func _process(delta: float) -> void:
	modulate.a = move_toward(modulate.a, 0.0, delta * 6.0)


func _on_conductor_beat(beat_number: int, beat_in_bar: int) -> void:
	modulate.a = 1.0
	color = Color("FFEC27") if beat_in_bar == 0 else Color.WHITE
