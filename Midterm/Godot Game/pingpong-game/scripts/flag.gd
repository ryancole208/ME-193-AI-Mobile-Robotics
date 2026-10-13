extends Sprite2D

func _ready() -> void:
	var conductor: Conductor = get_tree().get_first_node_in_group("conductor")
	conductor.beat.connect(_on_beat)

func _on_beat(beat_number: int, _beat_in_bar: int) -> void:
	frame = posmod(beat_number, 2)
