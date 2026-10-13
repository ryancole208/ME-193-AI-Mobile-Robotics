extends Sprite2D

@export var distance_px: float = 6.0
@export var period_s: float = 8.0
var start_x: float = 0.0
var time: float = 0.0

func _ready() -> void:
	start_x = position.x

func _process(delta: float) -> void:
	time += delta
	position.x = roundf(start_x + sin(time * TAU / period_s) * distance_px)
