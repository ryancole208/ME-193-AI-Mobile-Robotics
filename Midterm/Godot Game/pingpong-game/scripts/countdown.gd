extends Label

@export var conductor: Conductor
@onready var click: AudioStreamPlayer = $Click

func _ready() -> void:
	hide()
	conductor.beat.connect(_on_beat)

func _on_beat(beat_number: int, _beat_in_bar: int) -> void:
	var left: int = conductor.music_start_beat - beat_number
	if left > 0:
		text = str(left)
		show()
		click.play()
	else:
		hide()
