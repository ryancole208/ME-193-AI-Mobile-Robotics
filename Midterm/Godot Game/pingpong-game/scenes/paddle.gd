extends Sprite2D

@export var speed: float = 300.0

func _process(delta: float) -> void:
	var direction: float = Input.get_axis("move_left", "move_right")
	position.x += direction * speed * delta
	position.x = clampf(position.x, 0.0, get_viewport_rect().size.x)
	if Input.is_action_just_pressed("swing"):
		print("Swing!")
