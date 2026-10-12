extends Sprite2D

@export var conductor: Conductor

func _ready() -> void:
	conductor.beat.connect(_on_beat)

func _on_beat(beat_number: int, _beat_in_bar: int) -> void:
	frame = posmod(beat_number, 2)
