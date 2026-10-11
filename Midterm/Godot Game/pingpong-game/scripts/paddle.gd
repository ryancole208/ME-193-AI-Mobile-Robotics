extends Sprite2D

@export var speed: float = 300.0
@export var min_x: float = 72.0
@export var max_x: float = 312.0
@export var yellow_texture: Texture2D
@export var black_texture: Texture2D
var yellow_facing: bool = true
@export var max_roll_degrees: float = 45.0

func _process(delta: float) -> void:
	var direction: float = Input.get_axis("move_left", "move_right")
	position.x += direction * speed * delta
	position.x = clampf(position.x, min_x, max_x)
	if Input.is_action_just_pressed("swing"):
		print("Swing!")
	if Input.is_action_just_pressed("flip_face"):
		set_yellow_facing(not yellow_facing)
	var roll_input: float = Input.get_axis("roll_left", "roll_right")
	if roll_input != 0.0:
		set_roll(rotation_degrees + roll_input * 90.0 * delta)

func set_yellow_facing(value: bool) -> void:
	yellow_facing = value
	texture = yellow_texture if value else black_texture

func set_roll(degrees: float) -> void:
	rotation_degrees = clampf(degrees, -max_roll_degrees, max_roll_degrees)
