extends Sprite2D

@export var step_px: float = 6.0
@export var center_x: float = 192.0
@export var blade_offset: Vector2 = Vector2(-4, 5)
var last_x: float = 0.0
var walked: float = 0.0

func _ready() -> void:
	last_x = global_position.x

func _process(_delta: float) -> void:
	var x: float = global_position.x
	var moved: float = absf(x - last_x)
	last_x = x
	if moved > 0.01:
		walked += moved
		frame = int(walked / step_px) % 2
	else:
		walked = 0.0
		frame = 0
	flip_h = x < center_x
	offset = Vector2(-blade_offset.x if flip_h else blade_offset.x, blade_offset.y)
