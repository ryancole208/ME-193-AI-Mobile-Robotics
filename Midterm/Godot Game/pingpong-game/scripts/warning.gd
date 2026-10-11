extends Label

var time_left: float = 0.0

func _ready() -> void:
	visible = false

func _process(delta: float) -> void:
	time_left -= delta
	visible = time_left > 0.0 and fmod(time_left, 0.25) > 0.125

func _on_scheduler_cue() -> void:
	time_left = 1.2
