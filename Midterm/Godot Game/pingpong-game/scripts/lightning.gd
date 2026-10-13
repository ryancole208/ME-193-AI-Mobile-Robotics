extends Sprite2D

@export var every_bars: int = 2
@export var flash_time: float = 0.12
var downbeats: int = 0

func _ready() -> void:
	visible = false
	var conductor: Conductor = get_tree().get_first_node_in_group("conductor")
	conductor.beat.connect(_on_beat)

func _on_beat(_beat_number: int, beat_in_bar: int) -> void:
	if beat_in_bar != 0:
		return
	downbeats += 1
	if downbeats % every_bars == 0:
		flash()

func flash() -> void:
	visible = true
	await get_tree().create_timer(flash_time).timeout
	visible = false
