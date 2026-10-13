extends Node2D


# Called when the node enters the scene tree for the first time.
const LOOKS := {
	"retro": {
		"backdrop": preload("res://art/background_retro.png"),
		"table": preload("res://art/table_retro.png"),
		"opponent": preload("res://art/opponent_retro.png"),
		"paddle_yellow": preload("res://art/paddle_retro_yellow.png"),
		"paddle_black": preload("res://art/paddle_retro_black.png"),
	},
	"future": {
		"backdrop": preload("res://art/background_future.png"),
		"table": preload("res://art/table_future.png"),
		"opponent": preload("res://art/opponent_future.png"),
		"paddle_yellow": preload("res://art/paddle_future_yellow.png"),
		"paddle_black": preload("res://art/paddle_future_black.png"),
	},
}
const FANTASY_ONLY := ["FlagLight", "FlagDark", "CloudsSun", "CloudsDark", "Lightning"]

func _ready() -> void:
	var look_name: String = $Conductor.chart.get("theme", "fantasy")
	if not LOOKS.has(look_name):
		return
	var look: Dictionary = LOOKS[look_name]
	$Stage/Backdrop.texture = look["backdrop"]
	for node_name in FANTASY_ONLY:
		var node: Node = $Stage.get_node_or_null(node_name)
		if node:
			node.queue_free()
	$Table/TableArt.texture = look["table"]
	var opponent: Sprite2D = get_tree().get_first_node_in_group("opponent")
	opponent.texture = look["opponent"]
	$paddle.yellow_texture = look["paddle_yellow"]
	$paddle.black_texture = look["paddle_black"]
	$paddle.set_yellow_facing($paddle.yellow_facing)

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		Fader.go("res://scenes/SongSelect.tscn")
